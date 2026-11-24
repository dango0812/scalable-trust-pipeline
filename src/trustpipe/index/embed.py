"""Text embedders for dense retrieval.

An embedder maps texts to unit-length float32 vectors, so the inner product of two vectors
is their cosine similarity. Two implementations share one interface:

- ``HashingEmbedder``: character n-gram feature hashing. No model, no download, fully
  deterministic and fast; it measures surface overlap, so it is a baseline and the
  embedder of tests and torch-less installs, not a semantic model.
- ``TransformerEmbedder``: a Hugging Face encoder with mean pooling over the attention
  mask, as E5 and most sentence-embedding models are used. Queries and passages get the
  prefixes the model was trained with ("query: " and "passage: " for E5). Needs the
  ``cpu`` or ``gpu`` extra (PyTorch).

Every embedder has a ``spec``: compact JSON that names it completely (kind, model, revision,
dimensions, prefixes). The index records it, and ``embedder_from_spec`` rebuilds the same
embedder when the index is loaded, so queries are always embedded like the passages were.
"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]
_SPACE = re.compile(r"\s+")


class Embedder(Protocol):
    @property
    def spec(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_passages(self, texts: list[str]) -> Vectors: ...

    def embed_queries(self, texts: list[str]) -> Vectors: ...


def _normalised(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of character n-grams (n = 2..4) into ``dimensions`` slots."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 8:
            raise ValueError("a hashing embedder needs at least 8 dimensions")
        self._dimensions = dimensions

    @property
    def spec(self) -> str:
        return _spec(kind="hashing", ngrams=[2, 4], dimensions=self._dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> npt.NDArray[np.float32]:
        vector = np.zeros(self._dimensions, dtype=np.float32)
        simple = " " + _SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip() + " "
        for n in (2, 3, 4):
            for start in range(len(simple) - n + 1):
                digest = hashlib.blake2b(simple[start : start + n].encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                vector[value % self._dimensions] += 1.0 if value >> 63 else -1.0
        return vector

    def embed_passages(self, texts: list[str]) -> Vectors:
        vectors = [self._vector(t) for t in texts]
        if not vectors:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        return _normalised(np.stack(vectors))

    def embed_queries(self, texts: list[str]) -> Vectors:
        return self.embed_passages(texts)


class TransformerEmbedder:
    """A Hugging Face encoder with mean pooling; needs PyTorch (the cpu or gpu extra)."""

    def __init__(
        self,
        model: str,
        *,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: Any = AutoTokenizer.from_pretrained(model)
        self._model: Any = AutoModel.from_pretrained(model).to(self.device).eval()
        self._revision = str(getattr(self._model.config, "_commit_hash", None) or "local")
        self._dimensions = int(self._model.config.hidden_size)

    @property
    def spec(self) -> str:
        return _spec(
            kind="transformer",
            model=self.model_name,
            revision=self._revision,
            dimensions=self._dimensions,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, texts: list[str]) -> Vectors:
        import torch

        batches: list[Vectors] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self._tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            batches.append(pooled.float().cpu().numpy())
        if not batches:
            return np.zeros((0, self._dimensions), dtype=np.float32)
        return _normalised(np.concatenate(batches))

    def embed_passages(self, texts: list[str]) -> Vectors:
        return self._embed([self.passage_prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> Vectors:
        return self._embed([self.query_prefix + t for t in texts])


def _spec(**fields: object) -> str:
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def embedder_from_spec(spec: str, *, model_path: Path | None = None) -> Embedder:
    """Rebuild the embedder an index was built with (``model_path`` overrides the model)."""
    fields = json.loads(spec)
    if fields["kind"] == "hashing":
        return HashingEmbedder(int(fields["dimensions"]))
    if fields["kind"] == "transformer":
        return TransformerEmbedder(
            str(model_path or fields["model"]),
            query_prefix=fields["query_prefix"],
            passage_prefix=fields["passage_prefix"],
        )
    raise ValueError(f"unknown embedder kind {fields['kind']!r}")
