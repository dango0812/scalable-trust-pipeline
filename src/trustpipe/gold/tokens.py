"""Token files: gold documents as flat arrays of token ids, the fastest form to train on.

Training reads tokens, not text. Tokenizing once, ahead of training, turns every epoch into
plain sequential reads of a memory-mapped array: no parsing, no tokenizer in the training
loop, and any window of tokens is one slice. This is the layout of Megatron-LM's indexed
datasets (``.bin`` + ``.idx``), which NeMo Curator also exports to.

Per gold part and tokenizer:

    part-<id>.bin          token ids, little-endian uint16 (vocabularies up to 65,536)
                           or uint32, every document followed by the end-of-text token
    part-<id>.idx.parquet  per document: doc_id, offset of its first token, length

The tokenizer is named by the SHA-256 of its ``tokenizer.json``; changing the tokenizer
gives new files instead of overwriting old ones. ``TokenWindows`` serves fixed-length
training windows from all parts of one snapshot without loading them into memory.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from tokenizers import Tokenizer

from trustpipe import __version__
from trustpipe.gold.stages import BUILD_STAGE
from trustpipe.gold.tables import GOLD_DOCUMENTS, GOLD_TOKEN_INDEX
from trustpipe.identity.ids import typed_hash
from trustpipe.lake.layout import gold_tokens_keys, snapshot_key
from trustpipe.lake.receipts import (
    Receipt,
    commit,
    is_committed,
    load_receipt,
    verify_file,
)
from trustpipe.lake.storage import (
    lake_path,
    read_table,
    write_bytes,
    write_json,
    write_table,
)
from trustpipe.runtime.executor import PartReport, PermanentError, RunSummary, TaskEvent, run_stage

TOKENS_STAGE = "gold.tokens"
TOKENS_FORMAT = "trustpipe.tokens/1"


def tokenizer_id(path: Path) -> str:
    """SHA-256 (hex) of a tokenizer.json file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens_part_id(tokenizer: str, gold_part: str) -> str:
    return typed_hash("tokens-part", tokenizer.encode(), gold_part.encode()).hex()[:16]


def _dtype(vocab_size: int) -> np.dtype[Any]:
    return np.dtype("<u2") if vocab_size <= 1 << 16 else np.dtype("<u4")


def tokens_part(root: Path, gold_part: str, tokenizer_path: Path, eos_token: str) -> Receipt:
    """Tokenize the documents of one gold part and commit the token and index files."""
    build = load_receipt(root, BUILD_STAGE, gold_part)
    documents_file = build.outputs[0]
    verify_file(root, documents_file)
    documents = read_table(
        root, documents_file.key, schema=GOLD_DOCUMENTS, columns=["doc_id", "text"]
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id(eos_token)
    if eos is None:
        raise PermanentError(f"the tokenizer has no token {eos_token!r}")
    dtype = _dtype(tokenizer.get_vocab_size())
    texts = [str(text) for text in documents.column("text").to_pylist()]
    encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
    pieces = [np.asarray([*e.ids, eos], dtype=dtype) for e in encoded]
    lengths = np.asarray([len(p) for p in pieces], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]) if len(pieces) else lengths
    tokens = np.concatenate(pieces) if pieces else np.zeros(0, dtype=dtype)
    snapshot = build.transform["snapshot"]
    identity = tokenizer_id(tokenizer_path)
    bin_key, idx_key = gold_tokens_keys(snapshot, identity, gold_part)
    index = pa.Table.from_arrays(
        [
            documents.column("doc_id"),
            pa.array(offsets, pa.int64()),
            pa.array(lengths, pa.int32()),
        ],
        schema=GOLD_TOKEN_INDEX,
    )
    outputs = (
        write_bytes(root, bin_key, tokens.tobytes(), create_only=False),
        write_table(index, root, idx_key, create_only=False),
    )
    receipt = Receipt(
        stage=TOKENS_STAGE,
        part_id=tokens_part_id(identity, gold_part),
        inputs=(documents_file.key,),
        outputs=outputs,
        transform={
            "trustpipe": __version__,
            "snapshot": snapshot,
            "tokenizer_sha256": identity,
            "dtype": dtype.str,
            "eos_token_id": str(eos),
            "vocab_size": str(tokenizer.get_vocab_size()),
        },
        counters={"documents": documents.num_rows, "tokens": int(tokens.size)},
    )
    commit(root, receipt, create_only=False)
    return receipt


def tokens_manifest_key(snapshot: str, tokenizer: str) -> str:
    return snapshot_key(snapshot).removesuffix(".json") + f"/tokens-{tokenizer[:12]}.json"


@dataclass(frozen=True, slots=True)
class TokenSet:
    manifest_key: str
    tokens: int
    documents: int
    run: RunSummary


def _task(root: Path, tokenizer_path: Path, eos_token: str, gold_part: str) -> PartReport:
    receipt = tokens_part(root, gold_part, tokenizer_path, eos_token)
    return PartReport(
        tuple(o.key for o in receipt.outputs),
        receipt.counters["documents"],
        receipt.counters["tokens"],
        0,
        sum(o.bytes for o in receipt.outputs),
        counters=dict(receipt.counters),
    )


def tokenize_snapshot(
    root: Path,
    snapshot: str,
    tokenizer_path: Path,
    *,
    eos_token: str,
    workers: int = 1,
    on_event: Callable[[TaskEvent], None] | None = None,
) -> TokenSet:
    """Tokenize every gold part of a snapshot and write the token manifest."""
    manifest = json.loads(lake_path(root, snapshot_key(snapshot)).read_text(encoding="utf-8"))
    gold_parts = [entry["gold_part"] for entry in manifest["parts"]]
    identity = tokenizer_id(tokenizer_path)
    dtype = _dtype(Tokenizer.from_file(str(tokenizer_path)).get_vocab_size())
    run = run_stage(
        TOKENS_STAGE,
        partial(_task, root, tokenizer_path, eos_token),
        gold_parts,
        workers=workers,
        is_done=lambda part: is_committed(root, TOKENS_STAGE, tokens_part_id(identity, part)),
        on_event=on_event,
    )
    if run.failed:
        raise RuntimeError(f"gold.tokens failed for parts {list(run.failed_keys)}")
    receipts = [load_receipt(root, TOKENS_STAGE, tokens_part_id(identity, p)) for p in gold_parts]
    tokens = sum(r.counters["tokens"] for r in receipts)
    documents = sum(r.counters["documents"] for r in receipts)
    document: dict[str, Any] = {
        "format": TOKENS_FORMAT,
        "snapshot": snapshot,
        "tokenizer_sha256": identity,
        "dtype": dtype.str,
        "eos_token": eos_token,
        "tokens": tokens,
        "documents": documents,
        "parts": [
            {
                "tokens": r.outputs[0].key,
                "index": r.outputs[1].key,
                "sha256": r.outputs[0].sha256,
                "count": r.counters["tokens"],
            }
            for r in receipts
        ],
    }
    key = tokens_manifest_key(snapshot, identity)
    write_json(root, key, document, create_only=False)
    return TokenSet(key, tokens, documents, run)


class TokenWindows:
    """Fixed-length windows of ``length + 1`` tokens over every part of a token set.

    Window ``i`` of a part starts at ``i * length``, so consecutive windows share one token
    (the target of the last position is the input of the next window's first). Windows do
    not cross parts. Files are memory-mapped: only the windows read are paged in.
    """

    def __init__(self, root: Path, manifest_key: str, length: int) -> None:
        if length < 1:
            raise ValueError("window length must be positive")
        manifest = json.loads(lake_path(root, manifest_key).read_text(encoding="utf-8"))
        self.length = length
        self.dtype = np.dtype(manifest["dtype"])
        self._parts: list[npt.NDArray[Any]] = []
        self._starts: list[int] = []
        total = 0
        for part in manifest["parts"]:
            count = max(0, (part["count"] - 1) // length)
            if count == 0:
                continue
            path = lake_path(root, part["tokens"])
            self._parts.append(np.memmap(path, dtype=self.dtype, mode="r"))
            self._starts.append(total)
            total += count
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> npt.NDArray[np.int64]:
        if not 0 <= index < self._total:
            raise IndexError(index)
        part = int(np.searchsorted(self._starts, index, side="right")) - 1
        start = (index - self._starts[part]) * self.length
        return np.asarray(self._parts[part][start : start + self.length + 1], dtype=np.int64)
