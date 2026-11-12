"""Ingestion acceptance helpers: report generation and metadata-only section lookup."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.ingestion.indexer import Indexer, IndexSummary
from legal_rag.ingestion.schema import DocumentRecord


def lookup_section(idx: Indexer, doc_id: str, section: str) -> list[str]:
    """Chunk IDs whose section label equals `section` (metadata only, no retrieval)."""
    want = section.strip().lower()
    return [
        c.chunk_id
        for c in idx.store.get_chunks(doc_id)
        if (c.metadata.get("section_or_article_no") or "").lower()
        in (want, f"section {want}", f"article {want}")
    ]


def snippet(idx: Indexer, chunk_id: str, doc_id: str, n: int = 150) -> str:
    for c in idx.store.get_chunks(doc_id):
        if c.chunk_id == chunk_id:
            return " ".join(c.text.split())[:n]
    return ""


def write_report(
    path: Path, idx: Indexer, records: list[DocumentRecord], summary: IndexSummary | None
) -> str:
    by_type: dict[str, list[int]] = defaultdict(list)
    per_doc: list[tuple[str, int, int, int]] = []
    tokens: list[int] = []
    for r in records:
        chunks = idx.store.get_chunks(r.doc_id)
        tok = [c.metadata.get("token_count", 0) for c in chunks]
        tokens += tok
        by_type[r.doc_type.value].append(len(chunks))
        per_doc.append((r.doc_id, len(chunks), min(tok, default=0), max(tok, default=0)))
    suspicious = [d for d, n, _, _ in per_doc if n == 0 or n > 3000]
    lines = [
        "# Ingestion report",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}  "
        f"fingerprint `{idx.fingerprint}`",
        "",
        f"- documents: {len(records)}  chunks: {idx.store.count()}",
    ]
    if summary:
        lines.append(
            f"- last run: indexed {len(summary.indexed)}, skipped {len(summary.skipped)}, "
            f"cache hits {summary.cache_hits}, {summary.seconds}s"
        )
    if tokens:
        q = statistics.quantiles(tokens, n=20) if len(tokens) > 1 else [tokens[0]] * 19
        lines.append(
            f"- tokens/chunk: min {min(tokens)}, median {int(statistics.median(tokens))}, "
            f"p95 {int(q[18])}, max {max(tokens)}"
        )
    lines += [
        "",
        "## Chunks per document type",
        "",
        "| doc_type | docs | chunks |",
        "|---|---|---|",
    ]
    lines += [f"| {t} | {len(v)} | {sum(v)} |" for t, v in sorted(by_type.items())]
    lines += [
        "",
        "## Per document",
        "",
        "| doc_id | chunks | min tok | max tok |",
        "|---|---|---|---|",
    ]
    lines += [f"| {d} | {n} | {lo} | {hi} |" for d, n, lo, hi in per_doc]
    lines += ["", "## Needs attention", ""]
    lines.append(f"- zero or >3000 chunks: {suspicious or 'none'}")
    low = summary.low_text_docs if summary else {}
    lines.append(f"- low-text pages (OCR candidates): {dict(low) or 'none'}")
    lines.append(f"- chunk types: {dict(Counter(idx.store.chunk_type_counts()))}")
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text
