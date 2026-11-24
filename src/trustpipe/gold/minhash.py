"""MinHash signatures and locality-sensitive hashing for near-duplicate detection.

Two documents are near duplicates when the sets of their word n-grams ("shingles") overlap
strongly (Jaccard similarity). Comparing every pair is quadratic, so each document gets a
signature instead: for each of ``bands * rows`` hash functions, the minimum hash over its
shingles. The probability that two signatures agree at one position equals the Jaccard
similarity of the two shingle sets. The signature is cut into ``bands`` bands of ``rows``
values; documents that agree on every value of at least one band become candidates, which
makes the probability of being matched ``1 - (1 - s^rows)^bands`` for similarity ``s``.

The construction follows datatrove (used for FineWeb): text normalised before shingling,
hash functions ``(a * x + b) mod (2^61 - 1)`` with ``a`` and ``b`` drawn from a seeded
generator, 64-bit arithmetic. Shingle hashes are built from per-word BLAKE2b hashes, so
they are stable across processes and platforms (Python's ``hash`` is salted per process).

Each band is stored as one 64-bit hash of its ``rows`` values, so the dedup stage only
compares integers.
"""

import hashlib
import re
import unicodedata
from functools import cache, lru_cache

import numpy as np
import numpy.typing as npt

MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_NOT_WORD = re.compile(r"[\W_]+")
_BLOCK = 4096  # shingles per block, so huge documents do not allocate huge matrices


def normalise(text: str) -> list[str]:
    """Words for shingling: NFKC, case-folded, punctuation and symbols removed."""
    simple = _NOT_WORD.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    return simple.split()


@lru_cache(maxsize=1 << 18)
def _word_hash(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")


def shingle_hashes(words: list[str], n: int) -> npt.NDArray[np.uint64]:
    """64-bit hashes of the word n-grams; one shingle of all words if there are fewer."""
    if not words:
        return np.zeros(0, dtype=np.uint64)
    hashes = np.fromiter((_word_hash(w) for w in words), dtype=np.uint64, count=len(words))
    if len(hashes) < n:
        n = len(hashes)
    count = len(hashes) - n + 1
    combined = np.zeros(count, dtype=np.uint64)
    # Order-sensitive combination: h = sum(word_i * 31^(n-1-i)) mod 2^64.
    for offset in range(n):
        multiplier = np.uint64(pow(31, n - 1 - offset, 1 << 64))
        combined += hashes[offset : offset + count] * multiplier
    return combined


@cache
def _permutations(hashes: int, seed: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    generator = np.random.RandomState(seed)  # the legacy generator datatrove seeds
    a = generator.randint(1, int(MERSENNE_PRIME), size=hashes, dtype=np.uint64)
    b = generator.randint(0, int(MERSENNE_PRIME), size=hashes, dtype=np.uint64)
    return a, b


def signature(shingles: npt.NDArray[np.uint64], hashes: int, seed: int) -> npt.NDArray[np.uint64]:
    """The MinHash signature: per hash function, the minimum over all shingles."""
    a, b = _permutations(hashes, seed)
    result = np.full(hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(shingles), _BLOCK):
        block = shingles[start : start + _BLOCK, np.newaxis]
        values = (block * a + b) % MERSENNE_PRIME
        np.minimum(result, values.min(axis=0), out=result)
    return result


def band_hashes(values: npt.NDArray[np.uint64], bands: int, rows: int) -> list[int]:
    """One 64-bit hash per band of ``rows`` signature values."""
    data = values.astype("<u8").tobytes()
    width = rows * 8
    return [
        int.from_bytes(
            hashlib.blake2b(data[i * width : (i + 1) * width], digest_size=8).digest(), "little"
        )
        for i in range(bands)
    ]


def document_bands(text: str, *, ngram: int, bands: int, rows: int, seed: int) -> list[int]:
    """MinHash band hashes of a document."""
    shingles = shingle_hashes(normalise(text), ngram)
    return band_hashes(signature(shingles, bands * rows, seed), bands, rows)


class UnionFind:
    """Disjoint sets over 0..n-1 with path halving and union by size."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def candidate_groups(band_columns: npt.NDArray[np.uint64]) -> UnionFind:
    """Join documents that share all values of any band; rows are documents."""
    count, bands = band_columns.shape
    sets = UnionFind(count)
    for band in range(bands):
        keys = band_columns[:, band]
        order = np.argsort(keys, kind="stable")
        ordered = keys[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        ends = np.r_[starts[1:], count]
        shared = ends - starts > 1  # only groups of two or more need a union
        for start, end in zip(starts[shared].tolist(), ends[shared].tolist(), strict=True):
            first = int(order[start])
            for other in order[start + 1 : end].tolist():
                sets.union(first, other)
    return sets
