"""Normalized per-document JSON (data/processed/<doc_id>.json): parse once, re-chunk freely."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from legal_rag.ingestion.manifest import sha256_file
from legal_rag.ingestion.parser import PARSER_VERSION, Page, clean_pages, parse_file

_HEADING = re.compile(
    r"^(?:(?:Chapter|Part|Schedule|Annex(?:ure)?)\b.{0,80}|[A-Z][A-Z0-9 ,&/()\-]{3,80}"
    r"|\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,60})$"
)
_LIST = re.compile(r"^(?:\(\w{1,4}\)|[-•*]|\d+\))\s+")


class Block(BaseModel):
    type: str  # heading | paragraph | list | other
    text: str


class NormalizedPage(BaseModel):
    page_no: int
    blocks: list[Block]


class Stats(BaseModel):
    page_count: int
    avg_chars_per_page: float
    low_text_pages: list[int]


class NormalizedDocument(BaseModel):
    doc_id: str
    source_file_hash: str
    parser_name: str
    parser_version: str
    parsed_at: str
    pages: list[NormalizedPage]
    stats: Stats

    def to_pages(self) -> list[Page]:
        low = set(self.stats.low_text_pages)
        return [
            Page(p.page_no, "\n\n".join(b.text for b in p.blocks), p.page_no in low)
            for p in self.pages
        ]


def classify_block(text: str) -> str:
    one_line = "\n" not in text.strip()
    if one_line and len(text) <= 90 and _HEADING.match(text.strip()):
        return "heading"
    if _LIST.match(text.strip()):
        return "list"
    return "paragraph"


def normalize(doc_id: str, path: Path, file_hash: str, low_chars: int = 50) -> NormalizedDocument:
    raw = parse_file(path, low_chars)
    pages = clean_pages(raw)
    npages = []
    for p in pages:
        blocks = [
            Block(type=classify_block(t), text=t.strip())
            for t in re.split(r"\n\s*\n", p.text)
            if t.strip()
        ]
        npages.append(NormalizedPage(page_no=p.page, blocks=blocks))
    chars = [len(p.text) for p in pages]
    return NormalizedDocument(
        doc_id=doc_id,
        source_file_hash=file_hash,
        parser_name="pymupdf" if path.suffix.lower() == ".pdf" else "text",
        parser_version=PARSER_VERSION,
        parsed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        pages=npages,
        stats=Stats(
            page_count=len(pages),
            avg_chars_per_page=round(sum(chars) / len(chars), 1) if chars else 0.0,
            low_text_pages=[p.page for p in raw if p.low_text],
        ),
    )


def ensure_normalized(
    doc_id: str, path: Path, processed_dir: Path, low_chars: int = 50, force: bool = False
) -> tuple[NormalizedDocument, str]:
    """Return (doc, 'parsed'|'cached'). Re-parses only if the hash or parser version changed."""
    out = processed_dir / f"{doc_id}.json"
    file_hash = sha256_file(path)
    if out.exists() and not force:
        try:
            cached = NormalizedDocument.model_validate_json(out.read_text(encoding="utf-8"))
            if cached.source_file_hash == file_hash and cached.parser_version == PARSER_VERSION:
                return cached, "cached"
        except ValueError:
            pass  # corrupt cache: fall through and re-parse
    doc = normalize(doc_id, path, file_hash, low_chars)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(doc.model_dump_json(), encoding="utf-8")
    tmp.replace(out)  # atomic: never leaves a half-written file
    return doc, "parsed"
