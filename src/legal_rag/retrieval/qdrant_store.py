"""Qdrant adapter behind the VectorStore interface. The client is injected, never created here.

Named vectors: `dense` (cosine) and `sparse` (hashed term counts, server-side IDF => BM25-style).
Point IDs are derived deterministically from chunk_id so re-upserts overwrite.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from typing import Any

from qdrant_client import QdrantClient, models

from legal_rag.retrieval.base import VectorStore
from legal_rag.retrieval.embedder import tokenize
from legal_rag.retrieval.memory_store import FingerprintMismatchError
from legal_rag.retrieval.types import Chunk, Filters, SearchResult

DENSE, SPARSE = "dense", "sparse"
SPARSE_METHOD = "hashed-tf-idf-1"
_NS = uuid.UUID("6f1c0e9a-0f64-4c5e-9d0e-0c1a2b3c4d5e")
_KEYWORD_FIELDS = ("domain", "doc_type", "status", "doc_id", "chunk_id", "section_or_article_no")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def sparse_vector(text: str) -> models.SparseVector:
    counts = Counter(int(hashlib.md5(t.encode()).hexdigest()[:7], 16) for t in tokenize(text))
    return models.SparseVector(indices=list(counts), values=[float(v) for v in counts.values()])


def _filter(filters: Filters | None, must_not: list | None = None) -> models.Filter | None:
    must: list[Any] = []
    if filters:
        for key, val in (("domain", filters.domain), ("doc_type", filters.doc_type)):
            if val:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=val)))
        if filters.status:
            must.append(
                models.FieldCondition(key="status", match=models.MatchAny(any=list(filters.status)))
            )
        if filters.doc_ids:
            must.append(
                models.FieldCondition(
                    key="doc_id", match=models.MatchAny(any=list(filters.doc_ids))
                )
            )
    if filters and filters.sections:
        must.append(
            models.FieldCondition(
                key="section_or_article_no", match=models.MatchAny(any=list(filters.sections))
            )
        )
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )


class QdrantVectorStore(VectorStore):
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client, self.collection, self.dim = client, collection, dim
        self._ensure_collection()

    # ── setup / readiness ──
    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata={"fingerprint": "", "sparse_method": SPARSE_METHOD},
        )
        for field in _KEYWORD_FIELDS:
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )

    def is_ready(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection))
        except Exception:
            return False

    def get_status(self) -> str:
        return (
            f"ready (qdrant:{self.collection}, {self.count()} chunks)"
            if self.is_ready()
            else ("not_ready (qdrant)")
        )

    # ── fingerprint ──
    @property
    def fingerprint(self) -> str | None:
        meta = self.client.get_collection(self.collection).config.metadata or {}
        return meta.get("fingerprint") or None

    @fingerprint.setter
    def fingerprint(self, value: str | None) -> None:
        self.client.update_collection(self.collection, metadata={"fingerprint": value or ""})

    def check_fingerprint(self, fingerprint: str) -> None:
        current = self.fingerprint
        if current and self.count() and current != fingerprint:
            raise FingerprintMismatchError(
                f"index fingerprint {current!r} != {fingerprint!r}; run reindex --all"
            )
        if current != fingerprint:
            self.fingerprint = fingerprint

    # ── writes ──
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector={DENSE: v, SPARSE: sparse_vector(c.to_embed)},
                payload={
                    **c.metadata,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "embed_text": c.embed_text,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points[i : i + 128], wait=True)

    def delete_by_doc_id(self, doc_id: str) -> int:
        n = self.client.count(self.collection, count_filter=_doc_filter(doc_id), exact=True).count
        if n:
            self.client.delete(self.collection, models.FilterSelector(filter=_doc_filter(doc_id)))
        return n

    def delete_except(self, doc_id: str, keep_chunk_ids: set[str]) -> int:
        """Remove this document's chunks that are not in keep_chunk_ids (run AFTER upsert)."""
        stale = [c.chunk_id for c in self.get_chunks(doc_id) if c.chunk_id not in keep_chunk_ids]
        if stale:
            self.client.delete(
                self.collection, models.PointIdsList(points=[point_id(c) for c in stale])
            )
        return len(stale)

    def clear(self) -> None:
        self.client.delete_collection(self.collection)
        self._ensure_collection()

    def persist(self) -> None:
        """Qdrant persists itself."""

    # ── reads ──
    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def _scroll(self, flt: models.Filter | None = None) -> list[Any]:
        out, offset = [], None
        while True:
            pts, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=256, offset=offset, with_payload=True
            )
            out.extend(pts)
            if offset is None:
                return out

    @staticmethod
    def _to_chunk(payload: dict) -> Chunk:
        meta = {k: v for k, v in payload.items() if k not in ("text", "embed_text")}
        return Chunk(
            payload["chunk_id"],
            payload["doc_id"],
            payload["text"],
            payload.get("embed_text", ""),
            meta,
        )

    def list_docs(self) -> list[str]:
        return sorted({p.payload["doc_id"] for p in self._scroll()})

    def doc_hashes(self) -> dict[str, str]:
        return {p.payload["doc_id"]: p.payload.get("file_hash", "") for p in self._scroll()}

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        return [self._to_chunk(p.payload) for p in self._scroll(_doc_filter(doc_id))]

    def chunk_type_counts(self) -> dict[str, int]:
        return dict(Counter(p.payload.get("chunk_type") for p in self._scroll()))

    def _query(self, using: str, query: Any, filters: Filters | None, k: int, dense: bool):
        res = self.client.query_points(
            self.collection,
            query=query,
            using=using,
            query_filter=_filter(filters),
            limit=k,
            with_payload=True,
        ).points
        return [self._hit(p, dense) for p in res]

    def _hit(self, p: Any, dense: bool) -> SearchResult:
        m = p.payload
        return SearchResult(
            chunk_id=m["chunk_id"],
            doc_id=m["doc_id"],
            title=m.get("title", m["doc_id"]),
            section=m.get("section_or_article_no"),
            page=m.get("page_start"),
            text=m["text"],
            score=float(p.score),
            dense_score=float(p.score) if dense else 0.0,
            status=m.get("status", "in_force"),
            metadata=m,
        )

    def dense_search(self, dense: list[float], filters: Filters | None, k: int):
        return self._query(DENSE, dense, filters, k, True)

    def sparse_search(self, query: str, filters: Filters | None, k: int):
        sv = sparse_vector(query)
        return self._query(SPARSE, sv, filters, k, False) if sv.indices else []
