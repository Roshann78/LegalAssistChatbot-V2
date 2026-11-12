"""Build the configured vector store (memory or Qdrant). Qdrant client creation lives here."""

from __future__ import annotations

from legal_rag.config.settings import Settings
from legal_rag.retrieval.base import Embedder, VectorStore
from legal_rag.retrieval.memory_store import InMemoryVectorStore


def build_store(settings: Settings, embedder: Embedder) -> VectorStore:
    if settings.vector_store_backend == "memory":
        store = InMemoryVectorStore.load(settings.index_path)
        store.path = settings.index_path
        return store
    from qdrant_client import QdrantClient

    from legal_rag.retrieval.qdrant_store import QdrantVectorStore

    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url)
    else:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(settings.qdrant_path))
    dim = len(embedder.embed(["dimension probe"])[0])
    return QdrantVectorStore(client, settings.qdrant_collection, dim)
