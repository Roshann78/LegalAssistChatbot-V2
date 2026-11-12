"""Hybrid retrieval: dense + sparse fused with reciprocal rank fusion, plus diversity cap."""

from __future__ import annotations

from dataclasses import replace

from legal_rag.retrieval.base import Embedder, VectorStore
from legal_rag.retrieval.types import Filters, SearchResult


def rrf_fuse(
    ranked_lists: list[list[SearchResult]], weights: list[float] | None = None, k0: int = 60
) -> list[SearchResult]:
    """Reciprocal rank fusion. Keeps the max dense_score seen for each chunk."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for w, lst in zip(weights, ranked_lists, strict=True):
        for rank, r in enumerate(lst, start=1):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + w / (k0 + rank)
            prev = best.get(r.chunk_id)
            if prev is None or r.dense_score > prev.dense_score:
                best[r.chunk_id] = r
    fused = [replace(best[cid], score=s) for cid, s in scores.items()]
    fused.sort(key=lambda r: (-r.score, r.chunk_id))
    return fused


def cap_per_doc(results: list[SearchResult], cap: int) -> list[SearchResult]:
    seen: dict[str, int] = {}
    out = []
    for r in results:
        seen[r.doc_id] = seen.get(r.doc_id, 0) + 1
        if seen[r.doc_id] <= cap:
            out.append(r)
    return out


class Retriever:
    """Typed retrieval function: returns structured SearchResult lists."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        mode: str = "hybrid",
        candidates: int = 30,
        per_doc_cap: int = 3,
    ) -> None:
        self.embedder, self.store = embedder, store
        self.mode, self.candidates, self.per_doc_cap = mode, candidates, per_doc_cap

    def retrieve(
        self, query: str, k: int = 5, filters: Filters | None = None, extra_terms: str = ""
    ) -> list[SearchResult]:
        lists: list[list[SearchResult]] = []
        if self.mode in ("dense", "hybrid"):
            qv = self.embedder.embed([query])[0]
            lists.append(self.store.dense_search(qv, filters, self.candidates))
        if self.mode in ("sparse", "hybrid"):
            lists.append(
                self.store.sparse_search(f"{query} {extra_terms}", filters, self.candidates)
            )
        fused = rrf_fuse(lists) if len(lists) > 1 else lists[0]
        return cap_per_doc(fused, self.per_doc_cap)[:k]
