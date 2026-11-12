"""Idempotent, hash-based indexing with config fingerprinting and safe-order replacement."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from legal_rag.config.settings import Settings
from legal_rag.ingestion.cache import EmbeddingCache, text_key
from legal_rag.ingestion.chunking import (
    STRATEGY_VERSIONS,
    approx_tokens,
    chunk_document,
    strategy_for,
)
from legal_rag.ingestion.manifest import sha256_file
from legal_rag.ingestion.normalize import ensure_normalized
from legal_rag.ingestion.parser import PARSER_VERSION
from legal_rag.ingestion.schema import DocumentRecord
from legal_rag.retrieval.base import Embedder, VectorStore

logger = logging.getLogger("legal_rag.ingestion")


def embedder_dim(embedder: Embedder) -> int:
    dim = getattr(embedder, "dim", None)
    return int(dim) if dim else len(embedder.embed(["dimension probe"])[0])


def config_fingerprint(settings: Settings, embedder: Embedder) -> str:
    """Hash of everything that changes index contents."""
    parts = {
        "embedder": embedder.name,
        "embedder_revision": getattr(embedder, "revision", ""),
        "dim": embedder_dim(embedder),
        "strategy": settings.chunk_strategy,
        "strategy_versions": STRATEGY_VERSIONS,
        "size": settings.chunk_size,
        "overlap": settings.chunk_overlap,
        "max_tokens": settings.chunk_max_tokens,
        "prefix": settings.contextual_prefix,
        "sparse": "hashed-tf-idf-1" if settings.vector_store_backend == "qdrant" else "bm25-mem-1",
        "parser": PARSER_VERSION,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class IndexSummary:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chunks: int = 0
    cache_hits: int = 0
    low_text_docs: dict[str, list[int]] = field(default_factory=dict)
    seconds: float = 0.0


class Indexer:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        base_dir: Path | None = None,
        cache: EmbeddingCache | None = None,
        count_tokens: Callable[[str], int] = approx_tokens,
        processed_dir: Path | None = None,
    ) -> None:
        self.s, self.embedder, self.store = settings, embedder, store
        self.base_dir = base_dir or settings.data_dir
        self.processed_dir = processed_dir or settings.processed_dir
        self.fingerprint = config_fingerprint(settings, embedder)
        self.cache = cache or EmbeddingCache(None, embedder.name)
        self.count_tokens = count_tokens
        self.cache_hits = 0
        self.run_id = uuid.uuid4().hex[:12]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Batched embedding through the persistent content-hash cache."""
        keys = [text_key(t) for t in texts]
        have = self.cache.get_many(list(set(keys)))
        self.cache_hits += sum(1 for k in keys if k in have)
        missing = list({k: t for k, t in zip(keys, texts, strict=True) if k not in have}.items())
        for i in range(0, len(missing), 64):
            batch = missing[i : i + 64]
            vecs = self.embedder.embed([t for _, t in batch])
            fresh = {k: v for (k, _), v in zip(batch, vecs, strict=True)}
            self.cache.put_many(fresh)
            have.update(fresh)
        return [have[k] for k in keys]

    def parse(self, rec: DocumentRecord, force: bool = False) -> str:
        """Parse to data/processed/<doc_id>.json. Returns 'parsed' or 'cached'."""
        return ensure_normalized(
            rec.doc_id, self.base_dir / rec.path, self.processed_dir, self.s.low_text_chars, force
        )[1]

    def index_document(
        self, rec: DocumentRecord, force: bool = False, summary: IndexSummary | None = None
    ) -> str:
        """Returns 'indexed' or 'skipped'.

        Safe order: write the new chunks first, then delete this document's stale chunks.
        """
        path = self.base_dir / rec.path
        file_hash = sha256_file(path)
        self.store.check_fingerprint(self.fingerprint)
        if not force and self.store.doc_hashes().get(rec.doc_id) == file_hash:
            return "skipped"
        norm, _ = ensure_normalized(rec.doc_id, path, self.processed_dir, self.s.low_text_chars)
        if summary is not None and norm.stats.low_text_pages:
            summary.low_text_docs[rec.doc_id] = norm.stats.low_text_pages
        meta = rec.model_dump(mode="json")
        meta.update(
            file_hash=file_hash, config_fingerprint=self.fingerprint, ingest_run=self.run_id
        )
        chunks = chunk_document(
            meta,
            norm.to_pages(),
            strategy_for(rec.doc_type.value, self.s.chunk_strategy),
            size=self.s.chunk_size,
            overlap=self.s.chunk_overlap,
            contextual_prefix=self.s.contextual_prefix,
            count_tokens=self.count_tokens,
            max_tokens=self.s.chunk_max_tokens,
        )
        vectors = self._embed([c.to_embed for c in chunks])
        self.store.upsert(chunks, vectors)
        self.store.delete_except(rec.doc_id, {c.chunk_id for c in chunks})
        return "indexed"

    def index_all(self, records: list[DocumentRecord], force: bool = False) -> IndexSummary:
        start = time.perf_counter()
        summary = IndexSummary()
        hits0 = self.cache_hits
        for rec in records:
            outcome = self.index_document(rec, force=force, summary=summary)
            (summary.indexed if outcome == "indexed" else summary.skipped).append(rec.doc_id)
        summary.chunks = self.store.count()
        summary.cache_hits = self.cache_hits - hits0
        summary.seconds = round(time.perf_counter() - start, 3)
        logger.info(
            "indexed=%d skipped=%d chunks=%d cache_hits=%d",
            len(summary.indexed),
            len(summary.skipped),
            summary.chunks,
            summary.cache_hits,
        )
        return summary

    def remove(self, doc_id: str) -> int:
        """Delete a document's chunks from the index. Source files are never touched."""
        return self.store.delete_by_doc_id(doc_id)

    def verify(self, records: list[DocumentRecord]) -> dict[str, list[str]]:
        """Manifest vs index consistency. Every non-empty list is a problem."""
        by_id = {r.doc_id: r for r in records}
        indexed = set(self.store.list_docs())
        hashes = self.store.doc_hashes()
        report: dict[str, list[str]] = {
            "missing_from_index": sorted(set(by_id) - indexed),
            "orphaned_in_index": sorted(indexed - set(by_id)),
            "hash_mismatch": [],
            "fingerprint_mismatch": [],
            "zero_chunks": [],
        }
        for doc_id in sorted(set(by_id) & indexed):
            path = self.base_dir / by_id[doc_id].path
            if path.exists() and sha256_file(path) != hashes.get(doc_id):
                report["hash_mismatch"].append(doc_id)
            chunks = self.store.get_chunks(doc_id)
            if not chunks:
                report["zero_chunks"].append(doc_id)
            elif any(c.metadata.get("config_fingerprint") != self.fingerprint for c in chunks):
                report["fingerprint_mismatch"].append(doc_id)
        return report

    def chunk_type_counts(self) -> Counter:
        return Counter(self.store.chunk_type_counts())
