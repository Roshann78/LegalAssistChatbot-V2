"""Embedders: a dependency-free hashing embedder and an optional sentence-transformers one."""

from __future__ import annotations

import hashlib
import math
import re

from legal_rag.retrieval.base import Embedder

_TOKEN = re.compile(r"[a-z0-9]+")


_STOP = frozenset(
    "a an and are as at be by does for from has have how in is it its of on or that the to was "
    "were what when where which who whom why will with say about give me must".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with common stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class HashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder (signed, L2-normalised).

    Lightweight stand-in so tests/CI/eval smoke runs need no model download.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def is_ready(self) -> bool:
        return True

    def get_status(self) -> str:
        return f"ready ({self.name})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder(Embedder):
    """Real model adapter; imports sentence-transformers lazily (optional extra)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def is_ready(self) -> bool:
        return True

    def get_status(self) -> str:
        return f"ready ({self.name})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def build_embedder(name: str) -> Embedder:
    """`hashing[-dim]` -> HashingEmbedder, anything else -> sentence-transformers."""
    if name.startswith("hashing"):
        dim = int(name.split("-")[1]) if "-" in name else 512
        return HashingEmbedder(dim)
    return SentenceTransformerEmbedder(name)
