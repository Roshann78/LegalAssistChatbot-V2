"""In-memory vector store with dense + BM25 search and JSON persistence.

Enforces the config-fingerprint rule: an index never mixes fingerprints.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from legal_rag.retrieval.base import VectorStore
from legal_rag.retrieval.embedder import tokenize
from legal_rag.retrieval.types import Chunk, Filters, SearchResult


class FingerprintMismatchError(RuntimeError):
    """Raised when upserting under a different config fingerprint."""


class InMemoryVectorStore(VectorStore):
    def __init__(self, fingerprint: str | None = None) -> None:
        self.fingerprint = fingerprint
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}

    # ── readiness ──
    def is_ready(self) -> bool:
        return True

    def get_status(self) -> str:
        return f"ready (memory, {len(self._chunks)} chunks)"

    # ── writes ──
    def check_fingerprint(self, fingerprint: str) -> None:
        if self.fingerprint and self._chunks and self.fingerprint != fingerprint:
            raise FingerprintMismatchError(
                f"index fingerprint {self.fingerprint!r} != {fingerprint!r}; run reindex --all"
            )
        self.fingerprint = fingerprint

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for c, v in zip(chunks, vectors, strict=True):
            self._chunks[c.chunk_id] = c
            self._vectors[c.chunk_id] = v

    def delete_by_doc_id(self, doc_id: str) -> int:
        ids = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        for cid in ids:
            del self._chunks[cid]
            del self._vectors[cid]
        return len(ids)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

    # ── reads ──
    def count(self) -> int:
        return len(self._chunks)

    def list_docs(self) -> list[str]:
        return sorted({c.doc_id for c in self._chunks.values()})

    def doc_hashes(self) -> dict[str, str]:
        return {c.doc_id: c.metadata.get("file_hash", "") for c in self._chunks.values()}

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        return [c for c in self._chunks.values() if c.doc_id == doc_id]

    def _candidates(self, filters: Filters | None) -> list[Chunk]:
        if filters is None:
            return list(self._chunks.values())
        return [c for c in self._chunks.values() if filters.matches(c.metadata)]

    def _to_result(self, c: Chunk, score: float, dense: float) -> SearchResult:
        m = c.metadata
        return SearchResult(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=m.get("title", c.doc_id),
            section=m.get("section_or_article_no"),
            page=m.get("page_start"),
            text=c.text,
            score=score,
            dense_score=dense,
            status=m.get("status", "in_force"),
            metadata=m,
        )

    def dense_search(
        self, dense: list[float], filters: Filters | None, k: int
    ) -> list[SearchResult]:
        scored = []
        for c in self._candidates(filters):
            v = self._vectors[c.chunk_id]
            scored.append((sum(a * b for a, b in zip(dense, v, strict=False)), c))
        scored.sort(key=lambda t: (-t[0], t[1].chunk_id))
        return [self._to_result(c, s, s) for s, c in scored[:k]]

    def sparse_search(self, query: str, filters: Filters | None, k: int) -> list[SearchResult]:
        cands = self._candidates(filters)
        q_terms = set(tokenize(query))
        if not cands or not q_terms:
            return []
        docs = {c.chunk_id: Counter(tokenize(c.to_embed)) for c in self._chunks.values()}
        n = len(docs)
        avg = sum(sum(d.values()) for d in docs.values()) / max(n, 1)
        df = Counter(t for d in docs.values() for t in d)
        k1, b = 1.5, 0.75
        scored = []
        for c in cands:
            tf = docs[c.chunk_id]
            ln = sum(tf.values())
            s = 0.0
            for t in q_terms:
                if t in tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * ln / (avg or 1)))
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda t: (-t[0], t[1].chunk_id))
        return [self._to_result(c, s, 0.0) for s, c in scored[:k]]

    def search(self, dense, query_text, filters, k):  # type: ignore[override]
        if dense is not None:
            return self.dense_search(dense, filters, k)
        return self.sparse_search(query_text or "", filters, k)

    # ── persistence ──
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self.fingerprint,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "embed_text": c.embed_text,
                    "metadata": c.metadata,
                    "vector": self._vectors[c.chunk_id],
                }
                for c in self._chunks.values()
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> InMemoryVectorStore:
        store = cls()
        if not path.exists():
            return store
        data = json.loads(path.read_text(encoding="utf-8"))
        store.fingerprint = data.get("fingerprint")
        for d in data["chunks"]:
            c = Chunk(d["chunk_id"], d["doc_id"], d["text"], d["embed_text"], d["metadata"])
            store._chunks[c.chunk_id] = c
            store._vectors[c.chunk_id] = d["vector"]
        return store
