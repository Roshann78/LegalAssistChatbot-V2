"""Manifest loading, validation, and draft-entry helper."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from pydantic import ValidationError

from legal_rag.ingestion.schema import DocumentRecord, check_consistency


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path) -> list[DocumentRecord]:
    """Load records; raises ValueError listing all invalid records."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("documents", []) if isinstance(raw, dict) else raw
    records, errors = [], []
    for i, item in enumerate(items):
        try:
            records.append(DocumentRecord(**item))
        except ValidationError as exc:
            errors.append(
                f"record #{i} ({item.get('doc_id', '?')}): {exc.error_count()} error(s): "
                f"{[e['loc'] for e in exc.errors()]}"
            )
    if errors:
        raise ValueError("invalid manifest records:\n" + "\n".join(errors))
    return records


def validate_manifest(records: list[DocumentRecord], base_dir: Path) -> list[str]:
    """Return a list of problems (empty == valid)."""
    errors: list[str] = []
    ids = [r.doc_id for r in records]
    for dup in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"duplicate doc_id: {dup}")
    id_set = set(ids)
    for r in records:
        f = base_dir / r.path
        if not f.exists():
            errors.append(f"{r.doc_id}: file missing: {r.path}")
        elif r.file_hash and sha256_file(f) != r.file_hash:
            errors.append(f"{r.doc_id}: hash mismatch (file changed; run update)")
        if r.superseded_by and r.superseded_by not in id_set:
            errors.append(f"{r.doc_id}: superseded_by '{r.superseded_by}' not in manifest")
        errors.extend(check_consistency(r))
    return errors


def draft_entries(raw_dir: Path, known: set[str] | None = None) -> list[dict]:
    """Scan raw_dir (<domain>/<doc_type>/<doc_id>.pdf) and draft manifest entries."""
    known = known or set()
    drafts = []
    for f in sorted(raw_dir.rglob("*")):
        if f.suffix.lower() not in {".pdf", ".txt"} or f.stem in known:
            continue
        parts = f.relative_to(raw_dir).parts
        drafts.append(
            {
                "doc_id": f.stem.lower().replace(" ", "_"),
                "title": f.stem.replace("_", " ").replace("-", " ").title(),
                "doc_type": parts[1] if len(parts) >= 3 else "",
                "domain": parts[0] if len(parts) >= 2 else "",
                "path": f.as_posix(),
                "file_hash": sha256_file(f),
            }
        )
    return drafts
