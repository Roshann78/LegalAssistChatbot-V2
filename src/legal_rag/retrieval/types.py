"""Structured retrieval types shared across ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """An indexable chunk. `text` is shown/generated from; `embed_text` is embedded."""

    chunk_id: str
    doc_id: str
    text: str
    embed_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def to_embed(self) -> str:
        return self.embed_text or self.text


@dataclass(frozen=True)
class SearchResult:
    """Structured result; never a pre-joined string (keeps Part 2 agent tools easy)."""

    chunk_id: str
    doc_id: str
    title: str
    section: str | None
    page: int | None
    text: str
    score: float
    dense_score: float = 0.0
    status: str = "in_force"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Filters:
    """Metadata filters; None means 'no constraint'."""

    domain: str | None = None
    doc_type: str | None = None
    status: tuple[str, ...] | None = ("in_force", "amended")
    doc_ids: tuple[str, ...] | None = None

    def matches(self, meta: dict[str, Any]) -> bool:
        if self.domain and meta.get("domain") != self.domain:
            return False
        if self.doc_type and meta.get("doc_type") != self.doc_type:
            return False
        if self.status and meta.get("status", "in_force") not in self.status:
            return False
        return not (self.doc_ids and meta.get("doc_id") not in self.doc_ids)
