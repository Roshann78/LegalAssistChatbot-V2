"""Hybrid retrieval: dense + sparse fused with reciprocal rank fusion, plus diversity cap."""

from __future__ import annotations

import time
from dataclasses import replace

from legal_rag.retrieval.base import Embedder, VectorStore
from legal_rag.retrieval.rerank import Reranker
from legal_rag.retrieval.types import Chunk, Filters, SearchResult


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
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        reranker: Reranker | None = None,
        rerank_top_n: int = 30,
        parent_expand: bool = False,
        parent_max_chars: int = 2400,
    ) -> None:
        self.embedder, self.store = embedder, store
        self.mode, self.candidates, self.per_doc_cap = mode, candidates, per_doc_cap
        self.dense_weight, self.sparse_weight = dense_weight, sparse_weight
        self.reranker, self.rerank_top_n = reranker, rerank_top_n
        self.parent_expand, self.parent_max_chars = parent_expand, parent_max_chars
        self.last_rerank_ms = 0.0

    def retrieve(
        self, query: str, k: int = 5, filters: Filters | None = None, extra_terms: str = ""
    ) -> list[SearchResult]:
        lists: list[list[SearchResult]] = []
        weights: list[float] = []
        if self.mode in ("dense", "hybrid"):
            qv = self.embedder.embed([query])[0]
            lists.append(self.store.dense_search(qv, filters, self.candidates))
            weights.append(self.dense_weight)
        if self.mode in ("sparse", "hybrid"):
            lists.append(
                self.store.sparse_search(f"{query} {extra_terms}", filters, self.candidates)
            )
            weights.append(self.sparse_weight)
        fused = rrf_fuse(lists, weights) if len(lists) > 1 else lists[0]
        pool = cap_per_doc(fused, self.per_doc_cap)
        self.last_rerank_ms = 0.0
        if self.reranker is not None:
            t0 = time.perf_counter()
            pool = self.reranker.rerank(query, pool[: self.rerank_top_n], k)
            self.last_rerank_ms = (time.perf_counter() - t0) * 1e3
        out = pool[:k]
        return self._expand(out) if self.parent_expand else out

    def _expand(self, results: list[SearchResult]) -> list[SearchResult]:
        """Parent-child: return each hit with its neighbouring chunks as context."""
        cache: dict[str, list[Chunk]] = {}
        out = []
        for r in results:
            if r.doc_id not in cache:
                cache[r.doc_id] = sorted(self.store.get_chunks(r.doc_id), key=lambda c: c.chunk_id)
            chunks = cache[r.doc_id]
            i = next((n for n, c in enumerate(chunks) if c.chunk_id == r.chunk_id), None)
            if i is None:
                out.append(r)
                continue
            text = "\n\n".join(c.text for c in chunks[max(0, i - 1) : i + 2])
            if len(text) > self.parent_max_chars:
                text = text[: self.parent_max_chars]
            out.append(replace(r, text=text, metadata={**r.metadata, "child_chunk_id": r.chunk_id}))
        return out
