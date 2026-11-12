"""Citation parsing, validation, and mapping to structured objects."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from legal_rag.retrieval.types import SearchResult

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    index: int
    doc_id: str
    title: str
    section_or_article: str | None
    page: int | None
    status: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_citation_numbers(answer: str) -> list[int]:
    """All cited numbers in order of first appearance."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer):
        for n in re.split(r"\s*,\s*", m.group(1)):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def resolve_citations(
    answer: str, passages: list[SearchResult]
) -> tuple[list[Citation], list[int]]:
    """Return (valid citations, invalid numbers). Invalid ones never reach the client silently."""
    valid, invalid = [], []
    for n in parse_citation_numbers(answer):
        if 1 <= n <= len(passages):
            r = passages[n - 1]
            valid.append(Citation(n, r.doc_id, r.title, r.section, r.page, r.status, r.text[:240]))
        else:
            invalid.append(n)
    return valid, invalid


def strip_invalid_citations(answer: str, n_passages: int) -> str:
    """Remove citation markers that point at passages that do not exist."""

    def fix(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        ok = [n for n in nums if 1 <= n <= n_passages]
        return f"[{', '.join(map(str, ok))}]" if ok else ""

    return _CITE_RE.sub(fix, answer)
