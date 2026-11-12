"""Ephemeral per-session document uploads: validated, parsed with the ingestion components,
indexed in an in-memory store with TTL, isolated by an unguessable upload ID."""

from __future__ import annotations

import re
import secrets
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from legal_rag.config.settings import Settings
from legal_rag.ingestion.chunking import chunk_document
from legal_rag.ingestion.parser import clean_pages, parse_file
from legal_rag.retrieval.base import Embedder
from legal_rag.retrieval.memory_store import InMemoryVectorStore

ALLOWED_SUFFIXES = {".pdf", ".txt"}


class UploadError(Exception):
    """Validation failure; `status` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def safe_filename(name: str) -> str:
    """Basename only, restricted charset, bounded length (never used as a path on disk)."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (base or "upload")[:80]


@dataclass
class UploadInfo:
    upload_id: str
    filename: str
    pages: int
    chunks: int
    expires_in_seconds: int


class UploadStore:
    """Bounded registry of per-upload indexes (LRU eviction at capacity, TTL expiry)."""

    def __init__(
        self, ttl_seconds: int, max_sessions: int, time_fn: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl, self.max_sessions, self.time_fn = ttl_seconds, max_sessions, time_fn
        self._items: OrderedDict[str, tuple[float, InMemoryVectorStore]] = OrderedDict()
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        for uid in [u for u, (t, _) in self._items.items() if now - t > self.ttl]:
            del self._items[uid]

    def add(self, store: InMemoryVectorStore) -> str:
        now = self.time_fn()
        uid = secrets.token_urlsafe(24)
        with self._lock:
            self._purge(now)
            while len(self._items) >= self.max_sessions:
                self._items.popitem(last=False)
            self._items[uid] = (now, store)
        return uid

    def get(self, uid: str) -> InMemoryVectorStore | None:
        now = self.time_fn()
        with self._lock:
            self._purge(now)
            item = self._items.get(uid)
            if item is None:
                return None
            self._items.move_to_end(uid)
            return item[1]

    def delete(self, uid: str) -> bool:
        with self._lock:
            return self._items.pop(uid, None) is not None

    def count(self) -> int:
        with self._lock:
            self._purge(self.time_fn())
            return len(self._items)


def build_upload_index(
    filename: str, data: bytes, settings: Settings, embedder: Embedder
) -> tuple[InMemoryVectorStore, str, int]:
    """Validate and index one upload. CPU/blocking: call via asyncio.to_thread."""
    name = safe_filename(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadError("Only .pdf and .txt files are accepted", 415)
    if not data:
        raise UploadError("Empty file")
    if len(data) > settings.upload_max_bytes:
        raise UploadError(f"File exceeds {settings.upload_max_bytes} bytes", 413)
    if suffix == ".pdf" and not data.startswith(b"%PDF-"):
        raise UploadError("File content is not a PDF", 415)
    with tempfile.TemporaryDirectory() as tmp:  # removed even on failure
        path = Path(tmp) / f"upload{suffix}"  # fixed name: user input never becomes a path
        path.write_bytes(data)
        try:
            pages = clean_pages(parse_file(path, settings.low_text_chars))
        except Exception as exc:
            raise UploadError("Could not read the document") from exc
    if len(pages) > settings.upload_max_pages:
        raise UploadError(f"Document exceeds {settings.upload_max_pages} pages", 413)
    meta = {
        "doc_id": "upload",
        "title": name,
        "domain": None,
        "doc_type": "upload",
        "status": "in_force",
        "file_hash": "",
        "config_fingerprint": "upload",
    }
    chunks = chunk_document(
        meta,
        pages,
        "s0",
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        contextual_prefix=settings.contextual_prefix,
        max_tokens=settings.chunk_max_tokens,
    )
    if not chunks:
        raise UploadError("No extractable text (scanned PDFs are not supported)")
    if len(chunks) > settings.upload_max_chunks:
        raise UploadError(f"Document too large (>{settings.upload_max_chunks} chunks)", 413)
    store = InMemoryVectorStore()
    store.upsert(chunks, embedder.embed([c.to_embed for c in chunks]))
    return store, name, len(pages)
