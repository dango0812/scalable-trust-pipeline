"""Build the retrieval index of a gold snapshot.

The index is a directory of plain files, so it can be copied as it is to a server or into a
delivery bundle and opened there without the lake:

    chunks.parquet      every passage with its provenance (source URI, tier, license, offsets)
    dense.faiss         passage vectors (dense.py)
    lexical.npz         BM25 postings and document lengths (lexical.py)
    vocabulary.parquet  BM25 terms
    manifest.json       snapshot, embedder spec, counts and the SHA-256 of every file

Row ``i`` of chunks.parquet is vector ``i`` of the dense index and document ``i`` of the
lexical index. The directory lives in the lake under ``index/snapshot=<id>/embedder=<id>/``
and is committed by an ``index.build`` receipt, like every other stage output.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from trustpipe import __version__
from trustpipe.gold.tables import GOLD_CHUNKS
from trustpipe.identity.ids import typed_hash
from trustpipe.index.dense import EXACT_LIMIT, DenseIndex
from trustpipe.index.embed import Embedder
from trustpipe.index.lexical import LexicalIndex
from trustpipe.lake.layout import index_key, snapshot_key
from trustpipe.lake.receipts import Receipt, commit, is_committed, load_receipt, verify_file
from trustpipe.lake.storage import (
    WrittenFile,
    describe,
    lake_path,
    read_table,
    write_json,
    write_table,
)

STAGE = "index.build"
MANIFEST_FORMAT = "trustpipe.index/1"
CHUNK_COLUMNS = [
    "chunk_id",
    "doc_id",
    "ordinal",
    "char_start",
    "char_end",
    "source_uri",
    "tier",
    "tier_rank",
    "license",
    "text",
]


def embedder_id(spec: str) -> str:
    return hashlib.sha256(spec.encode()).hexdigest()


def index_id(snapshot: str, spec: str) -> str:
    return typed_hash("index", snapshot.encode(), spec.encode()).hex()[:16]


@dataclass(frozen=True, slots=True)
class BuiltIndex:
    index_id: str
    directory: Path
    chunks: int
    dense_kind: str
    receipt: Receipt


def _snapshot_chunks(root: Path, snapshot: str) -> pa.Table:
    manifest = json.loads(lake_path(root, snapshot_key(snapshot)).read_text(encoding="utf-8"))
    tables = []
    for part in manifest["parts"]:
        chunks = next(o for o in part["outputs"] if "/chunks/" in o["key"])
        written = WrittenFile(chunks["key"], chunks["bytes"], chunks["sha256"], chunks["rows"])
        verify_file(root, written)
        tables.append(read_table(root, written.key, schema=GOLD_CHUNKS, columns=CHUNK_COLUMNS))
    if not tables:
        return GOLD_CHUNKS.empty_table().select(CHUNK_COLUMNS)
    return pa.concat_tables(tables)


def build_index(
    root: Path, snapshot: str, embedder: Embedder, *, exact_limit: int = EXACT_LIMIT
) -> BuiltIndex:
    """Embed and index every chunk of a snapshot; an index already committed is reused."""
    spec = embedder.spec
    identity = index_id(snapshot, spec)
    embedder_hash = embedder_id(spec)
    directory = lake_path(root, index_key(snapshot, embedder_hash, "manifest.json")).parent
    if is_committed(root, STAGE, identity):
        receipt = load_receipt(root, STAGE, identity)
        return BuiltIndex(
            identity, directory, receipt.counters["chunks"], receipt.transform["dense"], receipt
        )
    chunks = _snapshot_chunks(root, snapshot)
    texts = [str(t) for t in chunks.column("text").to_pylist()]
    dense = DenseIndex.build(embedder.embed_passages(texts), exact_limit=exact_limit)
    lexical = LexicalIndex.build(texts)
    outputs = [
        write_table(
            chunks, root, index_key(snapshot, embedder_hash, "chunks.parquet"), create_only=False
        )
    ]
    directory.mkdir(parents=True, exist_ok=True)
    dense.save(directory / "dense.faiss")
    lexical.save(directory)
    outputs += [
        describe(root, index_key(snapshot, embedder_hash, name))
        for name in ("dense.faiss", "lexical.npz", "vocabulary.parquet")
    ]
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "index_id": identity,
        "snapshot": snapshot,
        "embedder": spec,
        "dimensions": embedder.dimensions,
        "dense": dense.kind,
        "chunks": len(texts),
        "files": {Path(o.key).name: o.sha256 for o in outputs},
    }
    outputs.append(
        write_json(
            root, index_key(snapshot, embedder_hash, "manifest.json"), manifest, create_only=False
        )
    )
    receipt = Receipt(
        stage=STAGE,
        part_id=identity,
        inputs=(snapshot_key(snapshot),),
        outputs=tuple(outputs),
        transform={"trustpipe": __version__, "embedder": spec, "dense": manifest["dense"]},
        counters={"chunks": len(texts), "terms": len(lexical.vocabulary)},
    )
    commit(root, receipt, create_only=False)
    return BuiltIndex(identity, directory, len(texts), manifest["dense"], receipt)
