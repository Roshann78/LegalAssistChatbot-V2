"""Reranking behind an interface: candidates in, reordered scored passages out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from legal_rag.retrieval.embedder import tokenize
from legal_rag.retrieval.types import SearchResult


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        """Return the top-k results reordered by relevance, with `score` replaced."""


class LexicalReranker(Reranker):
    """Dependency-free reranker: query-term coverage of text + section label.

    A cheap stand-in for tests/CI; it is NOT expected to match a cross-encoder.
    """

    name = "lexical"

    def rerank(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        q = set(tokenize(query))
        if not q:
            return results[:k]
        scored = []
        for r in results:
            terms = set(tokenize(f"{r.title} {r.section or ''} {r.text}"))
            scored.append(replace(r, score=len(q & terms) / len(q)))
        scored.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored[:k]


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder; imported lazily (optional extra)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self.name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], k: int) -> list[SearchResult]:
        if not results:
            return []
        scores = self._model.predict([(query, r.text) for r in results])
        ranked = sorted(
            zip(scores, results, strict=True), key=lambda t: (-float(t[0]), t[1].chunk_id)
        )
        return [replace(r, score=float(s)) for s, r in ranked[:k]]


def build_reranker(strategy: str, model: str) -> Reranker | None:
    if strategy == "none":
        return None
    if strategy == "lexical":
        return LexicalReranker()
    if not model:
        raise ValueError("rerank_strategy=cross_encoder needs RERANK_MODEL (no model is assumed)")
    return CrossEncoderReranker(model)
