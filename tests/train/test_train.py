import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from corpus import english, korean
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from trustpipe.gold.policy import GoldPolicy
from trustpipe.gold.stages import build_gold
from trustpipe.gold.tokens import TokenWindows, tokenize_snapshot
from trustpipe.index.build import build_index
from trustpipe.index.embed import HashingEmbedder
from trustpipe.index.search import Retriever
from trustpipe.train import lora
from trustpipe.train.data import QAPair, RagExample, load_qa, rag_examples
from trustpipe.train.lora import TrainConfig, _Batches, encode_example, train
from trustpipe.train.prompt import (
    NO_ANSWER_EN,
    NO_ANSWER_KO,
    SYSTEM,
    Source,
    citations,
    no_answer,
    render,
    user_message,
)

Lander = Callable[[list[tuple[str, str]]], Path]
PRIMARY = "https://primary.example.org/registry"
AGENCY = "https://agency.example.org/korean"


class Plain:
    chat_template: str | None = None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        raise AssertionError("a tokenizer without a template is never asked to apply one")


class Chat(Plain):
    chat_template: str | None = "template"

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "|".join(f"<{m['role']}>{m['content']}" for m in conversation) + "<assistant>"


SOURCES = (
    Source("passage one", "https://a.example.org/", "primary"),
    Source("two", "https://b.example.org/", "reviewed"),
)


def test_the_prompt_numbers_sources_and_names_the_no_answer_sentence() -> None:
    message = user_message("신뢰 등급은?", SOURCES)
    assert message.startswith("Sources:\n\n[1] https://a.example.org/ (tier: primary)\npassage one")
    assert "[2] https://b.example.org/ (tier: reviewed)" in message
    assert message.endswith(f"No-answer sentence: {NO_ANSWER_KO}")
    assert "(no sources)" in user_message("What?", ())
    assert (no_answer("What is it?"), no_answer("무엇인가?")) == (NO_ANSWER_EN, NO_ANSWER_KO)


def test_prompts_use_the_chat_template_when_the_model_has_one() -> None:
    plain = render(Plain(), "What?", SOURCES)
    assert plain.startswith(f"### System\n{SYSTEM}")
    assert plain.endswith("### Assistant\n")
    chat = render(Chat(), "What?", SOURCES)
    assert chat.startswith(f"<system>{SYSTEM}|<user>Sources:")
    assert chat.endswith("<assistant>")


def test_citations_resolve_valid_numbers_once_in_order() -> None:
    assert citations("see [2] and [1], again [2], not [3] or [0]", SOURCES) == [
        SOURCES[1],
        SOURCES[0],
    ]
    assert citations("no citation", SOURCES) == []


def test_qa_files_are_read_strictly(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    good = {"question": "q", "answer": "a", "source_uri": PRIMARY}
    path.write_text(json.dumps(good) + "\n\n" + json.dumps(good) + "\n", encoding="utf-8")
    assert load_qa(path) == [QAPair("q", "a", PRIMARY)] * 2
    path.write_text(json.dumps({"question": "q", "answer": " "}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: missing answer, source_uri"):
        load_qa(path)


@pytest.fixture
def lake_index(lake: Lander) -> tuple[Path, str, Retriever]:
    root = lake(
        [
            (PRIMARY, english(21, 150)),
            (AGENCY, korean(22, 150)),
            ("https://mirror.example.net/other", english(23, 150)),
        ]
    )
    snapshot = build_gold(root, GoldPolicy(version="test")).snapshot
    return root, snapshot, Retriever(build_index(root, snapshot, HashingEmbedder()).directory)


def test_rag_examples_place_the_oracle_or_withhold_it(
    lake_index: tuple[Path, str, Retriever],
) -> None:
    _, _, retriever = lake_index
    pairs = [QAPair(english(21, 150).split("\n")[0], "the registry", PRIMARY)] * 20
    examples = rag_examples(pairs, retriever, sources=2, oracle_rate=0.5, seed=3)
    with_oracle = [e for e in examples if e.oracle is not None]
    without = [e for e in examples if e.oracle is None]
    assert with_oracle
    assert without
    for example in with_oracle:
        assert example.oracle is not None
        assert example.sources[example.oracle - 1].source_uri == PRIMARY
        assert example.target == f"the registry [{example.oracle}]"
    for example in without:
        assert PRIMARY not in {s.source_uri for s in example.sources}
        assert example.target == NO_ANSWER_EN
    assert examples == rag_examples(pairs, retriever, sources=2, oracle_rate=0.5, seed=3)


def test_an_oracle_that_retrieval_misses_is_still_found(
    lake_index: tuple[Path, str, Retriever],
) -> None:
    _, _, retriever = lake_index
    pair = QAPair("completely unrelated words zzz", "answer", AGENCY)
    [example] = rag_examples([pair], retriever, sources=1, oracle_rate=1.0)
    assert example.sources[0].source_uri == AGENCY
    with pytest.raises(ValueError, match="is in the index"):
        rag_examples([QAPair("q", "a", "https://missing.example.org/")], retriever)
    with pytest.raises(ValueError, match="oracle_rate"):
        rag_examples([pair], retriever, oracle_rate=2.0)


def tiny_model(directory: Path) -> Path:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special = ["<unk>", "<pad>", "<s>", "</s>"]
    corpus = [english(21, 400), korean(22, 400), SYSTEM, user_message("q", SOURCES), NO_ANSWER_EN]
    tokenizer.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=special))
    fast = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    fast.save_pretrained(directory)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=fast.pad_token_id,
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(directory)
    return directory


@pytest.fixture
def setup(tmp_path: Path, lake_index: tuple[Path, str, Retriever]) -> dict[str, Any]:
    root, snapshot, retriever = lake_index
    model = tiny_model(tmp_path / "base")
    tokens = tokenize_snapshot(root, snapshot, model / "tokenizer.json", eos_token="</s>")
    pairs = [QAPair(english(21, 150).split("\n")[1], "the registry", PRIMARY)] * 4
    return {
        "model": model,
        "windows": TokenWindows(root, tokens.manifest_key, 32),
        "examples": rag_examples(pairs, retriever, sources=2, oracle_rate=0.5),
    }


def config(setup: dict[str, Any], out: Path, **overrides: Any) -> TrainConfig:
    settings: dict[str, Any] = {
        "steps": 4,
        "batch_size": 2,
        "grad_accum": 2,
        "seq_len": 48,
        "learning_rate": 1e-2,
        "warmup_steps": 1,
        "checkpoint_every": 1,
        "log_every": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        **overrides,
    }
    return TrainConfig(base_model=str(setup["model"]), output_dir=out, **settings)


def adapter(out: Path) -> dict[str, Any]:
    return load_file(str(out / "adapter" / "adapter_model.safetensors"))


def test_training_writes_an_adapter_logs_and_a_manifest(
    tmp_path: Path, setup: dict[str, Any]
) -> None:
    out = tmp_path / "run"
    result = train(
        config(setup, out), windows=setup["windows"], examples=setup["examples"], data={"qa": "x"}
    )
    assert (result.steps, result.resumed_from) == (4, 0)
    assert result.final_loss > 0
    assert result.tokens > 0
    assert (out / "adapter" / "adapter_config.json").exists()
    assert (out / "tokenizer" / "tokenizer.json").exists()
    lines = [
        json.loads(line)
        for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [line["step"] for line in lines] == [2, 4]
    checkpoints = sorted(p.name for p in (out / "checkpoints").iterdir())
    assert checkpoints == ["step-0000003", "step-0000004"]  # only the latest ones are kept
    manifest = json.loads((out / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_revision"] == "local"
    assert manifest["data"] == {
        "qa": "x",
        "answering_examples": 4,
        "token_windows": len(setup["windows"]),
    }
    assert set(manifest["versions"]) == {"scalable-trust-pipeline", "torch", "transformers", "peft"}


def test_a_resumed_run_ends_with_the_weights_of_an_uninterrupted_run(
    tmp_path: Path, setup: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    whole = tmp_path / "whole"
    train(config(setup, whole), windows=setup["windows"], examples=setup["examples"])

    crashed = tmp_path / "crashed"
    save = lora._save_checkpoint

    def save_then_crash(
        model: Any, optimizer: Any, directory: Path, step: int, tokens: int
    ) -> None:
        save(model, optimizer, directory, step, tokens)
        if step == 2:
            raise RuntimeError("the machine went down")

    monkeypatch.setattr(lora, "_save_checkpoint", save_then_crash)
    with pytest.raises(RuntimeError, match="went down"):
        train(config(setup, crashed), windows=setup["windows"], examples=setup["examples"])
    monkeypatch.setattr(lora, "_save_checkpoint", save)
    resumed = train(config(setup, crashed), windows=setup["windows"], examples=setup["examples"])
    assert resumed.resumed_from == 2
    first, second = adapter(whole), adapter(crashed)
    assert first.keys() == second.keys()
    assert all(bool((first[k] == second[k]).all()) for k in first)


def test_training_on_one_kind_of_data(tmp_path: Path, setup: dict[str, Any]) -> None:
    only_windows = train(config(setup, tmp_path / "w", steps=1), windows=setup["windows"])
    assert only_windows.tokens == 2 * 2 * 33
    only_answers = train(
        config(setup, tmp_path / "a", steps=2, checkpoint_every=5), examples=setup["examples"]
    )
    assert only_answers.tokens > 0
    with pytest.raises(ValueError, match="nothing to train on"):
        _Batches(config(setup, tmp_path / "n"), None, [], 0)


def test_long_prompts_are_cut_from_the_left_and_answers_kept(setup: dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(setup["model"]))
    example = RagExample("q", SOURCES, "the answer [1]", 1)
    ids, labels = encode_example(tokenizer, example, 12)
    assert len(ids) == len(labels) == 12
    answer = [t for t in labels if t != -100]
    assert answer[-1] == tokenizer.eos_token_id
    assert ids[-len(answer) :] == answer
    full_ids, _ = encode_example(tokenizer, example, 4096)
    assert full_ids[0] == tokenizer.bos_token_id
    tiny_ids, tiny_labels = encode_example(tokenizer, example, 2)
    assert tiny_ids == tiny_labels
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    templated, _ = encode_example(tokenizer, example, 4096)
    assert templated[0] != tokenizer.bos_token_id


def test_the_learning_rate_warms_up_then_decays() -> None:
    schedule = TrainConfig(
        "m", Path("o"), steps=110, warmup_steps=10, learning_rate=1.0, min_lr_ratio=0.1
    )
    assert schedule.learning_rate_at(0) == pytest.approx(0.1)
    assert schedule.learning_rate_at(9) == pytest.approx(1.0)
    assert schedule.learning_rate_at(60) == pytest.approx(0.55)
    assert schedule.learning_rate_at(110) == pytest.approx(0.1)
