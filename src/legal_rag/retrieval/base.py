"""Base interfaces and placeholders for retrieval and embedding components."""

from __future__ import annotations

from abc import ABC, abstractmethod

from legal_rag.retrieval.types import Chunk, Filters, SearchResult


class Embedder(ABC):
    """Abstract interface for dense embedding models."""

    name: str = "embedder"

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if embedder model is loaded and ready."""

    @abstractmethod
    def get_status(self) -> str:
        """Return human-readable status for readiness checks."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised embeddings, one per text."""
        raise NotImplementedError


class VectorStore(ABC):
    """Abstract interface for vector database storage and search."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if vector database is connected and ready."""

    @abstractmethod
    def get_status(self) -> str:
        """Return human-readable status for readiness checks."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    def delete_by_doc_id(self, doc_id: str) -> int:
        raise NotImplementedError

    def search(
        self,
        dense: list[float] | None,
        query_text: str | None,
        filters: Filters | None,
        k: int,
    ) -> list[SearchResult]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def list_docs(self) -> list[str]:
        raise NotImplementedError


class PlaceholderEmbedder(Embedder):
    """Placeholder embedder (readiness tests only)."""

    def __init__(self, ready: bool = True) -> None:
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready

    def get_status(self) -> str:
        return "ready (placeholder)" if self._ready else "not_ready (placeholder)"


class PlaceholderVectorStore(VectorStore):
    """Placeholder vector store (readiness tests only)."""

    def __init__(self, ready: bool = True) -> None:
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready

    def get_status(self) -> str:
        return "ready (placeholder)" if self._ready else "not_ready (placeholder)"

    def count(self) -> int:
        return 0

    def list_docs(self) -> list[str]:
        return []
