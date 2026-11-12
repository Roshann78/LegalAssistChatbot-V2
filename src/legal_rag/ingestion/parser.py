"""Parsing and cleaning: PDF/text -> normalized pages. Cleaning is layout-agnostic."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PARSER_VERSION = "pymupdf-1"
LOW_TEXT_CHARS = 50


@dataclass
class Page:
    page: int
    text: str
    low_text: bool = False


def parse_pdf(path: Path, low_chars: int = LOW_TEXT_CHARS) -> list[Page]:
    """Extract text per page with PyMuPDF; flag low-text pages (OCR candidates)."""
    import fitz  # PyMuPDF, imported lazily

    pages = []
    with fitz.open(path) as doc:
        for i, p in enumerate(doc, start=1):
            t = p.get_text("text") or ""
            pages.append(Page(i, t, low_text=len(t.strip()) < low_chars))
    return pages


def parse_text(path: Path, low_chars: int = LOW_TEXT_CHARS) -> list[Page]:
    """Plain text; form feeds (\\f) separate pages."""
    raw = path.read_text(encoding="utf-8")
    return [Page(i, t, len(t.strip()) < low_chars) for i, t in enumerate(raw.split("\f"), 1)]


def parse_file(path: Path, low_chars: int = LOW_TEXT_CHARS) -> list[Page]:
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path, low_chars)
    return parse_text(path, low_chars)


_PAGE_NUM = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)


_LEGAL_LINE = re.compile(
    r"^(?:Section|Sec\.|Article|Rule|Regulation)\s+\d|^\d+[A-Z]?\.\s|^\(\w{1,4}\)\s"
)


def clean_pages(pages: list[Page]) -> list[Page]:
    """Remove repeated headers/footers and bare page numbers; de-hyphenate; tidy spaces."""
    n = len(pages)
    counts: Counter[str] = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
        for ln in set(lines[:2] + lines[-2:]):
            counts[ln] += 1
    repeated = {
        ln
        for ln, c in counts.items()
        if n >= 4 and c >= max(3, int(0.5 * n)) and not _LEGAL_LINE.match(ln)
    }
    out = []
    for p in pages:
        kept = [
            ln.rstrip()
            for ln in p.text.splitlines()
            if ln.strip() not in repeated and not _PAGE_NUM.match(ln)
        ]
        text = "\n".join(kept)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        out.append(Page(p.page, text, p.low_text))
    return out
