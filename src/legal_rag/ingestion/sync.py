"""One-command incremental sync: data/raw/ -> manifest -> index, touching only what changed."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from legal_rag.config.settings import Settings
from legal_rag.ingestion.cache import EmbeddingCache
from legal_rag.ingestion.classify import classify_file, discover
from legal_rag.ingestion.indexer import Indexer, IndexSummary
from legal_rag.ingestion.manifest import load_manifest, sha256_file, validate_manifest
from legal_rag.ingestion.manifest_edit import append_entries, remove_entry, replace_file_hash
from legal_rag.ingestion.schema import DocumentRecord
from legal_rag.retrieval.base import Embedder, VectorStore
from legal_rag.retrieval.memory_store import FingerprintMismatchError

logger = logging.getLogger("legal_rag.ingestion")

FORCE_HINT = (
    "The index was built with different settings (embedding model, chunking or parser). "
    "Run `legal-rag-ingest sync --force` to rebuild it."
)


@dataclass
class SyncSummary:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)  # file gone, kept (no --prune)
    orphans: list[str] = field(default_factory=list)  # in the index but not in the manifest
    low_text_warnings: dict[str, list[int]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    fingerprint_error: bool = False
    chunks: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems and not self.fingerprint_error

    def render(self) -> str:
        def row(label: str, ids: list[str]) -> str:
            return f"  {label:<18}{len(ids)}" + (f"  ({', '.join(ids)})" if ids else "")

        lines = ["sync summary"]
        lines += [row("added", self.added), row("updated", self.updated)]
        lines += [f"  {'skipped':<18}{len(self.skipped)}", row("removed", self.removed)]
        lines.append(f"  {'low_text_warnings':<18}{len(self.low_text_warnings)}")
        for doc_id, pages in self.low_text_warnings.items():
            lines.append(
                f"    ! {doc_id}: little/no text on page(s) {pages[:10]} (scanned? no OCR)"
            )
        for doc_id in self.deleted_files:
            lines.append(f"  WARNING: source file for {doc_id!r} is missing; rerun with --prune")
        for doc_id in self.orphans:
            lines.append(f"  WARNING: {doc_id!r} is indexed but not in the manifest; use --prune")
        for p in self.problems:
            lines.append(f"  ERROR: {p}")
        if self.fingerprint_error:
            lines.append(f"  ERROR: {FORCE_HINT}")
        lines.append(f"  total indexed chunks: {self.chunks}")
        return "\n".join(lines)


def run_sync(
    s: Settings,
    idx: Indexer,
    store: VectorStore,
    *,
    prune: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> SyncSummary:
    sm = SyncSummary()
    manifest = s.manifest_path
    raw_dir = s.data_dir / "raw"
    records = load_manifest(manifest) if manifest.exists() else []

    # 1. New files: classify and append to the manifest (existing entries are never touched).
    known_paths = {r.path for r in records}
    taken = {r.doc_id for r in records}
    new_entries: list[dict] = []
    for f in discover(raw_dir):
        rel = f.relative_to(s.data_dir).as_posix()
        if rel in known_paths:
            continue
        entry = classify_file(f, raw_dir, taken, rel)
        try:
            DocumentRecord(**entry)
        except Exception as exc:  # a bad guess must not corrupt the manifest
            sm.problems.append(f"could not classify {rel}: {exc}")
            continue
        taken.add(entry["doc_id"])
        new_entries.append(entry)
    if new_entries:
        if dry_run:
            records = records + [DocumentRecord(**e) for e in new_entries]
        else:
            append_entries(manifest, new_entries)
            records = load_manifest(manifest)

    present = [r for r in records if (s.data_dir / r.path).is_file()]
    missing = [r for r in records if r not in present]

    # 2. Changed files: refresh only the machine-owned file_hash so `validate` stays green.
    for r in present:
        actual = sha256_file(s.data_dir / r.path)
        if r.file_hash and r.file_hash != actual and not dry_run:
            replace_file_hash(manifest, r.file_hash, actual)
    if not dry_run and present:
        records = load_manifest(manifest)
        present = [r for r in records if (s.data_dir / r.path).is_file()]
        missing = [r for r in records if r not in present]

    # 3. Validate what is on disk (hash drift is expected in a dry run).
    errs = [e for e in validate_manifest(present, s.data_dir) if "hash mismatch" not in e]
    if errs:
        sm.problems.extend(errs)
        sm.chunks = store.count()
        return sm

    # 4. Config fingerprint: refuse to mix incompatible vectors unless --force.
    if force:
        store.fingerprint = None  # new chunks are written first; stale ones removed per document
    else:
        try:
            store.check_fingerprint(idx.fingerprint)
        except FingerprintMismatchError:
            sm.fingerprint_error = True
            sm.chunks = store.count()
            return sm

    # 5. Index only new/changed documents (hash + parse cache + embedding cache do the skipping).
    stored = store.doc_hashes()
    before = set(store.list_docs())
    summary = IndexSummary()
    for rec in present:
        status = idx.sync_status(rec, stored)
        if dry_run:
            if force or status != "unchanged":
                (sm.added if status == "new" else sm.updated).append(rec.doc_id)
            else:
                sm.skipped.append(rec.doc_id)
            continue
        outcome = idx.index_document(rec, force=force, summary=summary)
        if outcome == "skipped":
            sm.skipped.append(rec.doc_id)
        else:
            (sm.updated if rec.doc_id in before else sm.added).append(rec.doc_id)
    sm.low_text_warnings = dict(summary.low_text_docs)

    # 6. Deleted files / orphaned index entries.
    in_manifest = {r.doc_id for r in records}
    orphans = sorted(before - in_manifest)
    if prune:
        for rec in missing:
            if not dry_run:
                idx.prune_document(rec.doc_id)
                remove_entry(manifest, rec.doc_id)
            sm.removed.append(rec.doc_id)
        for doc_id in orphans:
            if not dry_run:
                idx.prune_document(doc_id)
            sm.removed.append(doc_id)
    else:
        sm.deleted_files = [r.doc_id for r in missing]
        sm.orphans = orphans

    if not dry_run:
        store.persist()
    sm.chunks = store.count()
    return sm


def startup_sync(s: Settings, embedder: Embedder, store: VectorStore) -> SyncSummary | None:
    """Blocking sync for API startup. Never raises: a failure is logged and serving continues."""
    cache = EmbeddingCache(s.embedding_cache_path, embedder.name)
    try:
        idx = Indexer(s, embedder, store, base_dir=s.data_dir, cache=cache)
        sm = run_sync(s, idx, store)
        logger.info("startup sync:\n%s", sm.render())
        return sm
    except Exception:
        logger.exception("startup sync failed; serving the existing index")
        return None
    finally:
        cache.close()
