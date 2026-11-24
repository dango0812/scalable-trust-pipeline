"""LoRA fine-tuning of an open model on gold tokens and retrieval-augmented examples.

Training from scratch is out of reach for one team (Llama 3 used about 16,000 GPUs), so a
public base model is adapted with LoRA (Hu et al., 2021): small low-rank matrices next to
every linear layer are trained while the base weights stay frozen. The result is an adapter
of a few megabytes that serving stacks (transformers, vLLM, TGI) load on top of the base.

Each optimizer step mixes two kinds of micro-batches:

- knowledge: fixed-length windows of gold tokens (gold/tokens.py), the plain next-token
  objective over trusted text;
- answering: RAFT examples (data.py), a prompt of numbered sources and a question, with the
  loss on the answer tokens only, so the model learns to answer from sources, cite them
  and say so when they do not hold the answer.

``rag_ratio`` sets the share of answering micro-batches.

Fault tolerance: every ``checkpoint_every`` steps the adapter, the optimizer state and the
step are written to a new directory that is renamed into place when complete, so a crash
never leaves a half-written checkpoint. A restarted run resumes from the latest one. The
batch of step ``s`` is drawn from a generator seeded with ``(seed, s)`` and dropout is
seeded the same way, so a resumed run produces exactly the weights an uninterrupted run
would. The learning rate is a pure function of the step (linear warmup, cosine decay).

Every ``log_every`` steps one JSON line goes to ``train_log.jsonl`` (loss, learning rate,
tokens, tokens per second). The finished adapter, the tokenizer and a manifest of the
configuration, data and versions are written to ``output_dir``.
"""

import dataclasses
import json
import math
import shutil
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from trustpipe.gold.tokens import TokenWindows
from trustpipe.train.data import RagExample
from trustpipe.train.prompt import render

IGNORE = -100
MANIFEST_FORMAT = "trustpipe.training/1"
KEEP_CHECKPOINTS = 2


@dataclass(frozen=True, slots=True)
class TrainConfig:
    base_model: str
    output_dir: Path
    steps: int = 1000
    batch_size: int = 4
    grad_accum: int = 4
    seq_len: int = 1024
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1
    rag_ratio: float = 0.5
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    checkpoint_every: int = 200
    log_every: int = 10
    seed: int = 0
    device: str | None = None

    def learning_rate_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / self.warmup_steps
        span = max(1, self.steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / span)
        floor = self.learning_rate * self.min_lr_ratio
        return floor + (self.learning_rate - floor) * 0.5 * (1 + math.cos(math.pi * progress))


@dataclass(frozen=True, slots=True)
class TrainResult:
    output_dir: Path
    steps: int
    resumed_from: int
    final_loss: float
    tokens: int
    seconds: float


Sequence = tuple[list[int], list[int]]  # (input ids, labels)


def encode_example(tokenizer: Any, example: RagExample, seq_len: int) -> Sequence:
    """Prompt and answer tokens; the loss covers the answer and the end-of-text token."""
    prompt = tokenizer(
        render(tokenizer, example.question, example.sources), add_special_tokens=False
    )
    answer = tokenizer(example.target, add_special_tokens=False)
    prompt_ids: list[int] = list(prompt["input_ids"])
    if not tokenizer.chat_template and tokenizer.bos_token_id is not None:
        prompt_ids.insert(0, tokenizer.bos_token_id)
    answer_ids: list[int] = [*answer["input_ids"], tokenizer.eos_token_id]
    answer_ids = answer_ids[:seq_len]
    prompt_ids = (
        prompt_ids[len(prompt_ids) - (seq_len - len(answer_ids)) :]
        if seq_len > len(answer_ids)
        else []
    )
    return prompt_ids + answer_ids, [IGNORE] * len(prompt_ids) + answer_ids


def _collate(rows: list[Sequence], pad: int) -> dict[str, Any]:
    import torch

    width = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (row_ids, row_labels) in enumerate(rows):
        ids[i, : len(row_ids)] = torch.tensor(row_ids)
        labels[i, : len(row_labels)] = torch.tensor(row_labels)
        mask[i, : len(row_ids)] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class _Batches:
    """The micro-batches of a step, a pure function of (seed, step)."""

    def __init__(
        self, config: TrainConfig, windows: TokenWindows | None, answers: list[Sequence], pad: int
    ) -> None:
        if (windows is None or len(windows) == 0) and not answers:
            raise ValueError("nothing to train on: no token windows and no answering examples")
        self.config = config
        self.windows = windows if windows is not None and len(windows) > 0 else None
        self.answers = answers
        self.pad = pad

    def step(self, step: int) -> list[dict[str, Any]]:
        generator = np.random.default_rng([self.config.seed, step])
        batches = []
        windows = self.windows
        for _ in range(self.config.grad_accum):
            if windows is None or (self.answers and generator.random() < self.config.rag_ratio):
                picks = generator.integers(0, len(self.answers), self.config.batch_size)
                rows = [self.answers[int(i)] for i in picks]
            else:
                picks = generator.integers(0, len(windows), self.config.batch_size)
                tokens = [[int(t) for t in windows[int(i)]] for i in picks]
                rows = [(window, window) for window in tokens]
            batches.append(_collate(rows, self.pad))
        return batches


def _checkpoints(directory: Path) -> list[tuple[int, Path]]:
    """Complete checkpoints (renamed into place) as (step, path), oldest first."""
    names = [p for p in directory.glob("step-*") if p.is_dir()]
    found = [(int(p.name[5:]), p) for p in names if p.name[5:].isdigit()]
    return sorted(found)


def _save_checkpoint(model: Any, optimizer: Any, directory: Path, step: int, tokens: int) -> None:
    import torch

    final = directory / f"step-{step:07d}"
    partial = directory / f"{final.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    model.save_pretrained(partial)
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    (partial / "trainer.json").write_text(
        json.dumps({"step": step, "tokens": tokens}), encoding="utf-8"
    )
    partial.rename(final)
    for _, old in _checkpoints(directory)[:-KEEP_CHECKPOINTS]:
        shutil.rmtree(old)


def _load_model(config: TrainConfig, device: str) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model: Any = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=dtype)
    model.to(device)
    torch.manual_seed(config.seed)
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return tokenizer, get_peft_model(model, lora)


def train(
    config: TrainConfig,
    *,
    windows: TokenWindows | None = None,
    examples: list[RagExample] | None = None,
    data: dict[str, Any] | None = None,
) -> TrainResult:
    """Fine-tune, resuming from the latest checkpoint in ``output_dir`` if there is one."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    started = time.perf_counter()
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_model(config, device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    answers = [encode_example(tokenizer, e, config.seq_len) for e in examples or []]
    batches = _Batches(config, windows, answers, pad)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    checkpoints = config.output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    start, tokens = 0, 0
    if found := _checkpoints(checkpoints):
        start, latest = found[-1]
        set_peft_model_state_dict(model, load_file(str(latest / "adapter_model.safetensors")))
        optimizer.load_state_dict(torch.load(latest / "optimizer.pt", weights_only=True))
        tokens = json.loads((latest / "trainer.json").read_text(encoding="utf-8"))["tokens"]
    model.train()
    loss_value = float("nan")
    log = (config.output_dir / "train_log.jsonl").open("a", encoding="utf-8")
    with log:
        for step in range(start, config.steps):
            torch.manual_seed(config.seed * 1_000_003 + step)
            rate = config.learning_rate_at(step)
            for group in optimizer.param_groups:
                group["lr"] = rate
            losses = []
            for batch in batches.step(step):
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                (loss / config.grad_accum).backward()
                losses.append(float(loss.detach()))
                tokens += int((batch["labels"] != IGNORE).sum())
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_value = float(np.mean(losses))
            done = step + 1
            if done % config.log_every == 0 or done == config.steps:
                elapsed = time.perf_counter() - started
                line = {
                    "step": done,
                    "loss": round(loss_value, 6),
                    "lr": rate,
                    "tokens": tokens,
                    "seconds": round(elapsed, 3),
                }
                log.write(json.dumps(line) + "\n")
                log.flush()
            if done % config.checkpoint_every == 0 or done == config.steps:
                _save_checkpoint(model, optimizer, checkpoints, done, tokens)
    model.save_pretrained(config.output_dir / "adapter")
    tokenizer.save_pretrained(config.output_dir / "tokenizer")
    seconds = time.perf_counter() - started
    result = TrainResult(config.output_dir, config.steps, start, loss_value, tokens, seconds)
    _write_manifest(config, model, result, data or {}, len(answers), len(windows) if windows else 0)
    return result


def _write_manifest(
    config: TrainConfig,
    model: Any,
    result: TrainResult,
    data: dict[str, Any],
    answers: int,
    windows: int,
) -> None:
    settings = {
        k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(config).items()
    }
    manifest = {
        "format": MANIFEST_FORMAT,
        "base_model": config.base_model,
        "base_revision": str(
            getattr(model.base_model.model.config, "_commit_hash", None) or "local"
        ),
        "config": settings,
        "data": {**data, "answering_examples": answers, "token_windows": windows},
        "result": {
            "steps": result.steps,
            "resumed_from": result.resumed_from,
            "final_loss": result.final_loss,
            "tokens": result.tokens,
        },
        "versions": {
            name: version(name)
            for name in ("scalable-trust-pipeline", "torch", "transformers", "peft")
        },
    }
    path = config.output_dir / "train_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
