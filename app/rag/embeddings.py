"""Text embedders.

* ``FastEmbedEmbedder`` - open-source BAAI/bge-small-en-v1.5 (ONNX, CPU, no torch)
  served through ``fastembed``; the default.
* ``LexicalEmbedder``   - dependency-free hashed TF vectors; used offline, in tests,
  or when the model cannot be downloaded.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.query_embed(text)), dtype=np.float32))[0]


_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = {"the", "a", "an", "of", "and", "or", "is", "are", "to", "in", "on", "for", "with", "what", "which",
         "does", "do", "how", "has", "have", "it", "its", "this", "that", "samsung", "galaxy", "phone", "phones"}


class LexicalEmbedder:
    """Feature-hashed unigram+bigram vectors with sublinear TF. Deterministic, no fitting."""

    def __init__(self, dims: int = 4096):
        self.dims = dims
        self.name = f"lexical-hash-{dims}"

    def _vector(self, text: str) -> np.ndarray:
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts: dict[int, float] = {}
        for feature in features:
            index = int(hashlib.md5(feature.encode()).hexdigest()[:8], 16) % self.dims
            counts[index] = counts.get(index, 0.0) + 1.0
        vector = np.zeros(self.dims, dtype=np.float32)
        for index, count in counts.items():
            vector[index] = 1.0 + math.log(count)
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vector(t) for t in texts])) if texts else np.zeros((0, self.dims), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vector(text)[None, :])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    backend = settings.embedding_backend
    if backend in ("auto", "fastembed"):
        try:
            return FastEmbedEmbedder(settings.embedding_model)
        except Exception as exc:  # missing package, no network for first download, ...
            if backend == "fastembed":
                raise
            log.warning("fastembed unavailable (%s); falling back to lexical embeddings", exc)
    return LexicalEmbedder()


def reset_embedder() -> None:
    get_embedder.cache_clear()
