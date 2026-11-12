"""Idempotent, hash-based indexing with config fingerprinting."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from legal_rag.config.settings import Settings
from legal_rag.ingestion.chunking import STRATEGY_VERSIONS, chunk_document, strategy_for
from legal_rag.ingestion.manifest import sha256_file
from legal_rag.ingestion.parser import PARSER_VERSION, clean_pages, parse_file
from legal_rag.ingestion.schema import DocumentRecord
from legal_rag.retrieval.base import Embedder
from legal_rag.retrieval.memory_store import InMemoryVectorStore

logger = logging.getLogger("legal_rag.ingestion")


def config_fingerprint(settings: Settings, embedder: Embedder) -> str:
    """Hash of everything that changes index contents (embedder, chunking, parser)."""
    parts = {
        "embedder": embedder.name,
        "strategy": settings.chunk_strategy,
        "strategy_versions": STRATEGY_VERSIONS,
        "size": settings.chunk_size,
        "overlap": settings.chunk_overlap,
        "parser": PARSER_VERSION,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class IndexSummary:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks: int = 0
    low_text_docs: list[str] = field(default_factory=list)
    seconds: float = 0.0


class Indexer:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: InMemoryVectorStore,
        base_dir: Path | None = None,
    ) -> None:
        self.s, self.embedder, self.store = settings, embedder, store
        self.base_dir = base_dir or settings.data_dir
        self.fingerprint = config_fingerprint(settings, embedder)
        self._cache: dict[str, list[float]] = {}

    def _doc_meta(self, rec: DocumentRecord, file_hash: str) -> dict:
        d = rec.model_dump(mode="json")
        d["file_hash"] = file_hash
        d["config_fingerprint"] = self.fingerprint
        return d

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Batched embedding with a content-hash cache."""
        keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in self._cache]
        uniq = list(dict(missing).items())
        for i in range(0, len(uniq), 64):
            batch = uniq[i : i + 64]
            for (k, _), v in zip(batch, self.embedder.embed([t for _, t in batch]), strict=True):
                self._cache[k] = v
        return [self._cache[k] for k in keys]

    def index_document(self, rec: DocumentRecord, force: bool = False) -> str:
        """Returns 'indexed' or 'skipped'. Re-indexes only if the file hash changed."""
        path = self.base_dir / rec.path
        file_hash = sha256_file(path)
        self.store.check_fingerprint(self.fingerprint)
        if not force and self.store.doc_hashes().get(rec.doc_id) == file_hash:
            return "skipped"
        pages = clean_pages(parse_file(path))
        self.store.delete_by_doc_id(rec.doc_id)
        strategy = strategy_for(rec.doc_type.value, self.s.chunk_strategy)
        chunks = chunk_document(
            self._doc_meta(rec, file_hash),
            pages,
            strategy,
            size=self.s.chunk_size,
            overlap=self.s.chunk_overlap,
        )
        vectors = self._embed([c.to_embed for c in chunks])
        self.store.upsert(chunks, vectors)
        return "indexed"

    def index_all(self, records: list[DocumentRecord], force: bool = False) -> IndexSummary:
        start = time.perf_counter()
        summary = IndexSummary()
        for rec in records:
            outcome = self.index_document(rec, force=force)
            (summary.indexed if outcome == "indexed" else summary.skipped).append(rec.doc_id)
        summary.chunks = self.store.count()
        summary.seconds = round(time.perf_counter() - start, 3)
        logger.info(
            "indexed=%d skipped=%d chunks=%d",
            len(summary.indexed),
            len(summary.skipped),
            summary.chunks,
        )
        return summary

    def remove(self, doc_id: str) -> int:
        return self.store.delete_by_doc_id(doc_id)

    def verify(self, records: list[DocumentRecord]) -> dict[str, list[str]]:
        manifest_ids = {r.doc_id for r in records}
        indexed = set(self.store.list_docs())
        return {
            "missing_from_index": sorted(manifest_ids - indexed),
            "orphaned_in_index": sorted(indexed - manifest_ids),
        }
