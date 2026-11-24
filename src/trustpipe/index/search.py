"""Hybrid retrieval: dense and BM25 results fused by reciprocal rank, with provenance.

Each retriever returns its best ``pool`` passages; the two rankings are fused with
reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009):
``score = sum(1 / (60 + rank))`` over the rankings a passage appears in. RRF needs no
score calibration between cosine similarities and BM25 scores, and a passage found by both
retrievers rises to the top.

A ``min_tier`` filter keeps only passages from sources trusted at least that much, so a
server can answer from primary and institutional sources only. Every hit carries the
source URI, tier, license and character offsets of its passage, which is what a citation
needs, plus the raw cosine similarity, which a server can threshold to decide that the
index holds no evidence for a question.

``evaluate`` measures recall@k and mean reciprocal rank on questions whose answer source
is known, the numbers a retrieval change should be judged by.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

from trustpipe.identity.ids import to_text
from trustpipe.index.dense import DenseIndex
from trustpipe.index.embed import Embedder, embedder_from_spec
from trustpipe.index.lexical import LexicalIndex
from trustpipe.lake.schema import TrustTier

RRF_K = 60


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    source_uri: str
    tier: str
    license: str
    char_start: int
    char_end: int
    score: float  # fused reciprocal-rank score
    cosine: float | None  # None when only the keyword search found the passage
    bm25: float | None  # None when only the dense search found the passage


class Retriever:
    """Opens an index directory (in the lake or in a delivery bundle) and searches it."""

    def __init__(self, directory: Path, embedder: Embedder | None = None) -> None:
        self.manifest: dict[str, Any] = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.embedder = embedder or embedder_from_spec(self.manifest["embedder"])
        if self.embedder.spec != self.manifest["embedder"]:
            raise ValueError("the embedder differs from the one the index was built with")
        self._rows: list[dict[str, Any]] = pq.read_table(directory / "chunks.parquet").to_pylist()
        self._dense = DenseIndex.load(directory / "dense.faiss")
        self._lexical = LexicalIndex.load(directory)

    def __len__(self) -> int:
        return len(self._rows)

    def search(
        self, query: str, k: int = 5, *, min_tier: TrustTier | None = None, pool: int = 50
    ) -> list[Hit]:
        if k < 1 or not self._rows:
            return []
        dense = self._dense.search(self.embedder.embed_queries([query])[0], max(pool, k))
        lexical = self._lexical.search(query, max(pool, k))
        fused: dict[int, float] = {}
        for ranking in (dense, lexical):
            for rank, (row, _) in enumerate(ranking, start=1):
                fused[row] = fused.get(row, 0.0) + 1.0 / (RRF_K + rank)
        cosine, bm25 = dict(dense), dict(lexical)
        minimum = -1 if min_tier is None else TrustTier(min_tier).rank
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        hits: list[Hit] = []
        for row, score in ordered:
            chunk = self._rows[row]
            if chunk["tier_rank"] < minimum:
                continue
            hits.append(
                Hit(
                    chunk_id=to_text(chunk["chunk_id"]),
                    doc_id=to_text(chunk["doc_id"]),
                    text=chunk["text"],
                    source_uri=chunk["source_uri"],
                    tier=chunk["tier"],
                    license=chunk["license"],
                    char_start=chunk["char_start"],
                    char_end=chunk["char_end"],
                    score=score,
                    cosine=cosine.get(row),
                    bm25=bm25.get(row),
                )
            )
            if len(hits) == k:
                break
        return hits


@dataclass(frozen=True, slots=True)
class Evaluation:
    questions: int
    k: int
    recall: float  # share of questions whose expected source is among the top k
    mrr: float  # mean of 1 / rank of the expected source (0 when it is not found)


def evaluate(retriever: Retriever, cases: Iterable[tuple[str, str]], k: int = 5) -> Evaluation:
    """Score retrieval on (question, expected source URI) pairs."""
    reciprocal: list[float] = []
    for question, source in cases:
        uris = [hit.source_uri for hit in retriever.search(question, k)]
        reciprocal.append(1.0 / (uris.index(source) + 1) if source in uris else 0.0)
    if not reciprocal:
        return Evaluation(0, k, 0.0, 0.0)
    found = cast(float, np.mean([r > 0 for r in reciprocal]))
    return Evaluation(len(reciprocal), k, float(found), float(np.mean(reciprocal)))
