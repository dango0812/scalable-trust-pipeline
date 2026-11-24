"""BM25 keyword search over passages.

Dense vectors find paraphrases; exact terms (names, numbers, statute ids, rare words) are
found more reliably by keyword matching. Hybrid retrieval runs both and fuses the results
(search.py).

Terms: text is NFKC-normalised and case-folded and split into words. Latin words are terms
as they are. Words containing Hangul, kana or CJK ideographs are cut into overlapping
character bigrams ("학습데이터를" -> 학습, 습데, 데이, 이터, 터를), the standard analyzer for
CJK text without a morphological analyzer: particles attached to a word still leave its stem
bigrams matchable.

Scoring is Lucene's BM25: idf = ln(1 + (N - df + 0.5) / (df + 0.5)), k1 = 1.2, b = 0.75.

The index is stored as compressed sparse rows per term (``indptr``, ``docs``, ``freqs``)
plus document lengths in one ``.npz`` file and the vocabulary in Parquet. A query only
touches the postings of its own terms.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

K1 = 1.2
B = 0.75
_WORD = re.compile(r"\w+")
_CJK = re.compile(r"[ᄀ-ᇿ぀-ヿ㄰-㆏㐀-鿿가-힣]")


def terms(text: str) -> list[str]:
    """Index terms of a text: Latin words as they are, CJK words as character bigrams."""
    found: list[str] = []
    for word in _WORD.findall(unicodedata.normalize("NFKC", text).casefold()):
        if _CJK.search(word) and len(word) > 1:
            found += [word[i : i + 2] for i in range(len(word) - 1)]
        else:
            found.append(word)
    return found


@dataclass(frozen=True, slots=True)
class LexicalIndex:
    vocabulary: dict[str, int]
    indptr: npt.NDArray[np.int64]
    docs: npt.NDArray[np.int32]
    freqs: npt.NDArray[np.int32]
    lengths: npt.NDArray[np.int32]

    @classmethod
    def build(cls, texts: list[str]) -> "LexicalIndex":
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(len(texts), dtype=np.int32)
        for doc, text in enumerate(texts):
            counts = Counter(terms(text))
            lengths[doc] = sum(counts.values())
            for term, count in counts.items():
                postings.setdefault(term, []).append((doc, count))
        vocabulary = {term: i for i, term in enumerate(sorted(postings))}
        sizes = np.array([len(postings[t]) for t in vocabulary], dtype=np.int64)
        indptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        flat = [pair for term in vocabulary for pair in postings[term]]
        docs = np.array([d for d, _ in flat], dtype=np.int32)
        freqs = np.array([f for _, f in flat], dtype=np.int32)
        return cls(vocabulary, indptr, docs, freqs, lengths)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """The ``k`` best documents for a query as (document index, BM25 score)."""
        count = len(self.lengths)
        if count == 0:
            return []
        scores = np.zeros(count, dtype=np.float64)
        average = float(self.lengths.mean()) or 1.0
        norm = K1 * (1 - B + B * self.lengths / average)
        for term in set(terms(query)):
            index = self.vocabulary.get(term)
            if index is None:
                continue
            start, end = int(self.indptr[index]), int(self.indptr[index + 1])
            docs, freqs = self.docs[start:end], self.freqs[start:end]
            idf = np.log(1 + (count - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += idf * freqs * (K1 + 1) / (freqs + norm[docs])
        hits = np.flatnonzero(scores > 0)
        top = hits[np.argsort(-scores[hits], kind="stable")[:k]]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, directory: Path) -> tuple[Path, Path]:
        arrays = directory / "lexical.npz"
        vocabulary = directory / "vocabulary.parquet"
        np.savez_compressed(
            arrays, indptr=self.indptr, docs=self.docs, freqs=self.freqs, lengths=self.lengths
        )
        pq.write_table(pa.table({"term": list(self.vocabulary)}), vocabulary, compression="zstd")
        return arrays, vocabulary

    @classmethod
    def load(cls, directory: Path) -> "LexicalIndex":
        with np.load(directory / "lexical.npz") as arrays:
            indptr, docs = arrays["indptr"], arrays["docs"]
            freqs, lengths = arrays["freqs"], arrays["lengths"]
        names = cast(list[str], pq.read_table(directory / "vocabulary.parquet")["term"].to_pylist())
        return cls({term: i for i, term in enumerate(names)}, indptr, docs, freqs, lengths)
