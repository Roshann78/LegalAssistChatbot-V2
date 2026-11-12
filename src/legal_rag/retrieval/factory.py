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


def build_retriever(settings: Settings, embedder: Embedder, store: VectorStore):
    """Retriever fully configured from settings (shared by the API and the eval runner)."""
    from legal_rag.retrieval.hybrid import Retriever
    from legal_rag.retrieval.rerank import build_reranker

    return Retriever(
        embedder,
        store,
        mode=settings.retrieval_mode,
        candidates=settings.retrieval_candidates,
        per_doc_cap=settings.retrieval_per_doc_cap,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        reranker=build_reranker(settings.rerank_strategy, settings.rerank_model),
        rerank_top_n=settings.rerank_top_n,
        parent_expand=settings.parent_expand,
        parent_max_chars=settings.parent_max_chars,
    )
