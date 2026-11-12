"""Pluggable, versioned chunking strategies (S0 baseline, S1 statutes, S2 judgments, S3)."""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass

from legal_rag.ingestion.parser import Page
from legal_rag.retrieval.types import Chunk

STRATEGY_VERSIONS = {"s0": "s0-1", "s1": "s1-1", "s2": "s2-1", "s3": "s3-1"}

_DOCTYPE_STRATEGY = {
    "statute": "s1",
    "constitution": "s1",
    "rule": "s1",
    "scheme": "s1",
    "judgment": "s2",
    "circular": "s3",
    "master_direction": "s3",
    "faq": "s3",
    "mapping_table": "s3",
}


def strategy_for(doc_type: str, configured: str = "auto") -> str:
    return _DOCTYPE_STRATEGY.get(doc_type, "s0") if configured == "auto" else configured


def approx_tokens(text: str) -> int:
    """Default token counter (~4 chars/token). Inject a real tokenizer for production."""
    return -(-len(text) // 4)


_TOC_LINE = re.compile(r"(?:\.{2,}|\s{2,}|\s)\d{1,4}\s*$")


def is_toc_page(text: str) -> bool:
    """Table-of-contents / arrangement-of-sections page: mostly short lines ending in numbers."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    numbered = sum(1 for ln in lines if _TOC_LINE.search(ln) and len(ln) < 120)
    marker = bool(re.search(r"\b(contents|arrangement of sections)\b", text[:300], re.I))
    return numbered / len(lines) >= (0.5 if marker else 0.75)


@dataclass
class _Seg:
    start: int
    end: int
    section: str | None = None
    heading: list[str] | None = None
    para: str | None = None


class _Text:
    """Concatenated page text with offset -> page lookup."""

    def __init__(self, pages: list[Page]) -> None:
        self.parts, self.offsets, self.page_nos = [], [], []
        pos = 0
        for p in pages:
            self.offsets.append(pos)
            self.page_nos.append(p.page)
            self.parts.append(p.text)
            pos += len(p.text) + 2
        self.full = "\n\n".join(self.parts)

    def page_at(self, off: int) -> int:
        i = bisect.bisect_right(self.offsets, off) - 1
        return self.page_nos[max(i, 0)]


def fixed_spans(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Split [start,end) into <=size windows, preferring paragraph/sentence/space boundaries."""
    spans, pos = [], start
    while pos < end:
        stop = min(pos + size, end)
        if stop < end:
            window = text[pos:stop]
            floor = int(size * 0.5)
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep, floor)
                if cut != -1:
                    stop = pos + cut + len(sep)
                    break
        if text[pos:stop].strip():
            spans.append((pos, stop))
        if stop >= end:
            break
        pos = max(stop - overlap, pos + 1)
    return spans


_SECTION_RE = re.compile(
    r"^[ \t]*(?:(?:Section|Sec\.)[ \t]+(\d+[A-Z]{0,2})|(\d+[A-Z]{0,2}))\.[ \t]+\S"
    r"|^[ \t]*Article[ \t]+(\d+[A-Z]{0,2})[\.\-–—:]?[ \t]+\S",
    re.M,
)
_PARA_RE = re.compile(r"^[ \t]*(\d{1,3})\.[ \t]+\S", re.M)
_HEADING_RE = re.compile(
    r"^[ \t]*((?:Chapter|Part|Schedule|Annex(?:ure)?)\b.{0,80}|[A-Z][A-Z0-9 ,&/()\-]{3,80}"
    r"|\d+(?:\.\d+)*\.?[ \t]+[A-Z][^\n]{2,80})[ \t]*$",
    re.M,
)


def _segments_s0(t: _Text, size: int, overlap: int) -> list[_Seg]:
    return [_Seg(a, b) for a, b in fixed_spans(t.full, 0, len(t.full), size, overlap)]


def _split_long(t: _Text, seg: _Seg, size: int, overlap: int) -> list[_Seg]:
    if seg.end - seg.start <= size:
        return [seg]
    return [
        _Seg(a, b, seg.section, seg.heading, seg.para)
        for a, b in fixed_spans(t.full, seg.start, seg.end, size, overlap)
    ]


def _segments_s1(t: _Text, size: int, overlap: int) -> list[_Seg]:
    marks = []
    for m in _SECTION_RE.finditer(t.full):
        no = m.group(1) or m.group(2) or m.group(3)
        label = f"Article {no}" if m.group(3) else f"Section {no}"
        marks.append((m.start(), label))
    return _by_marks(t, marks, size, overlap)


def _segments_s3(t: _Text, size: int, overlap: int) -> list[_Seg]:
    marks = [(m.start(), m.group(1).strip()) for m in _HEADING_RE.finditer(t.full)]
    segs = _by_marks(t, marks, size, overlap)
    for s in segs:
        s.heading = [s.section] if s.section else []
        s.section = None
    return segs


def _by_marks(t: _Text, marks: list[tuple[int, str]], size: int, overlap: int) -> list[_Seg]:
    if not marks:
        return _segments_s0(t, size, overlap)
    segs: list[_Seg] = []
    if t.full[: marks[0][0]].strip():
        segs.extend(_split_long(t, _Seg(0, marks[0][0]), size, overlap))
    for i, (pos, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(t.full)
        if t.full[pos:end].strip():
            segs.extend(_split_long(t, _Seg(pos, end, section=label), size, overlap))
    return segs


def _segments_s2(t: _Text, size: int, overlap: int) -> list[_Seg]:
    marks = [(m.start(), m.group(1)) for m in _PARA_RE.finditer(t.full)]
    if not marks:
        return _segments_s0(t, size, overlap)
    paras: list[_Seg] = []
    if t.full[: marks[0][0]].strip():
        paras.append(_Seg(0, marks[0][0]))
    for i, (pos, no) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(t.full)
        paras.append(_Seg(pos, end, para=no))
    out: list[_Seg] = []
    cur: _Seg | None = None
    first_no = last_no = None
    for p in paras:
        for piece in _split_long(t, p, size, overlap):
            if cur is not None and piece.end - cur.start <= size and piece.para:
                cur.end, last_no = piece.end, piece.para
                cur.para = f"{first_no}-{last_no}" if first_no != last_no else first_no
            else:
                if cur is not None:
                    out.append(cur)
                cur = _Seg(piece.start, piece.end, para=piece.para)
                first_no = last_no = piece.para
    if cur is not None:
        out.append(cur)
    return out


def _enforce_max(
    t: _Text, segs: list[_Seg], count: Callable[[str], int], max_tokens: int, overlap: int
) -> list[_Seg]:
    out: list[_Seg] = []
    for s in segs:
        if count(t.full[s.start : s.end].strip()) <= max_tokens:
            out.append(s)
            continue
        size = s.end - s.start
        pieces = [s]
        while size > 20:
            size = int(size * 0.75)
            spans = fixed_spans(t.full, s.start, s.end, size, min(overlap, size // 5))
            pieces = [_Seg(a, b, s.section, s.heading, s.para) for a, b in spans]
            if all(count(t.full[p.start : p.end].strip()) <= max_tokens for p in pieces):
                break
        out.extend(pieces)
    return out


_STRATEGIES: dict[str, Callable[[_Text, int, int], list[_Seg]]] = {
    "s0": _segments_s0,
    "s1": _segments_s1,
    "s2": _segments_s2,
    "s3": _segments_s3,
}


def chunk_document(
    doc_meta: dict,
    pages: list[Page],
    strategy: str,
    *,
    size: int = 800,
    overlap: int = 150,
    contextual_prefix: bool = True,
    count_tokens: Callable[[str], int] = approx_tokens,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Chunk a document; every chunk carries the document metadata plus chunk-level fields."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown chunk strategy {strategy!r}")
    kept = [Page(p.page, "" if is_toc_page(p.text) else p.text, p.low_text) for p in pages]
    t = _Text(kept)
    segs = _STRATEGIES[strategy](t, size, overlap)
    if max_tokens:
        segs = _enforce_max(t, segs, count_tokens, max_tokens, overlap)
    doc_id, title = doc_meta["doc_id"], doc_meta.get("title", doc_meta["doc_id"])
    chunks = []
    for i, s in enumerate(segs):
        text = t.full[s.start : s.end].strip()
        if not text:
            continue
        heading_path = [title, *([s.section] if s.section else []), *(s.heading or [])]
        meta = {
            **doc_meta,
            "chunk_id": f"{doc_id}:{STRATEGY_VERSIONS[strategy]}:{i:05d}",
            "section_or_article_no": s.section,
            "heading_path": heading_path,
            "page_start": t.page_at(s.start),
            "page_end": t.page_at(max(s.end - 1, s.start)),
            "paragraph_no": s.para,
            "chunk_type": strategy,
            "token_count": count_tokens(text),
            "strategy_id": strategy,
            "strategy_version": STRATEGY_VERSIONS[strategy],
        }
        prefix = " > ".join(heading_path) + "\n" if contextual_prefix else ""
        chunks.append(Chunk(meta["chunk_id"], doc_id, text, prefix + text if prefix else "", meta))
    return chunks
