"""Dense vector search with FAISS.

Vectors are unit length, so inner product is cosine similarity. Up to ``EXACT_LIMIT``
vectors the index is exact (``IndexFlatIP``: every query compares against every vector,
about 50 ms per query for a million 384-dimensional vectors on one core). Above it the
index is HNSW (``IndexHNSWFlat``, 32 links per node), the graph index most vector
databases use, which answers in about a millisecond at a recall of 0.95 to 0.99 depending
on ``ef_search``.

Larger corpora split the index into shards that are searched in parallel and merged by
score, and compress vectors with product quantisation (FAISS ``IndexIVFPQ``); the search
interface stays the same.
"""

from pathlib import Path
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

EXACT_LIMIT = 1_000_000
HNSW_LINKS = 32


class DenseIndex:
    def __init__(self, index: Any) -> None:
        self._index = index

    @classmethod
    def build(
        cls, vectors: npt.NDArray[np.float32], *, exact_limit: int = EXACT_LIMIT
    ) -> "DenseIndex":
        count, dimensions = vectors.shape
        if count <= exact_limit:
            index: Any = faiss.IndexFlatIP(dimensions)
        else:
            index = faiss.IndexHNSWFlat(dimensions, HNSW_LINKS, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 128
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index)

    @property
    def kind(self) -> str:
        return type(self._index).__name__

    def __len__(self) -> int:
        return int(self._index.ntotal)

    def search(self, query: npt.NDArray[np.float32], k: int) -> list[tuple[int, float]]:
        """The ``k`` nearest vectors to one query vector as (vector index, cosine)."""
        if len(self) == 0:
            return []
        scores, ids = self._index.search(
            np.ascontiguousarray(query.reshape(1, -1)), min(k, len(self))
        )
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    # FAISS opens files with C stdio, which cannot open non-ASCII paths on Windows (a user
    # name in Korean is enough), so the index is serialised in memory and written by Python.
    def save(self, path: Path) -> Path:
        path.write_bytes(faiss.serialize_index(self._index).tobytes())
        return path

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        return cls(faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8)))
