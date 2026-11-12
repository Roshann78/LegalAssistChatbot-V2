"""Typed metadata schema for documents (manifest records). See docs/metadata_schema.md."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class DocType(StrEnum):
    statute = "statute"
    constitution = "constitution"
    judgment = "judgment"
    rule = "rule"
    circular = "circular"
    master_direction = "master_direction"
    scheme = "scheme"
    faq = "faq"
    mapping_table = "mapping_table"


class Domain(StrEnum):
    legal = "legal"
    banking = "banking"


class Status(StrEnum):
    in_force = "in_force"
    amended = "amended"
    superseded = "superseded"
    repealed = "repealed"


class SourceType(StrEnum):
    pdf = "pdf"
    text = "text"
    api_text = "api_text"


class TextQuality(StrEnum):
    good = "good"
    fair = "fair"
    poor = "poor"
    needs_ocr = "needs_ocr"
    unknown = "unknown"


class DocumentRecord(BaseModel):
    """One manifest record. The manifest is the source of truth; the index is derived."""

    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$", description="Stable slug")
    title: str = Field(min_length=1)
    doc_type: DocType
    domain: Domain
    path: str = Field(min_length=1, description="File path relative to the data dir")
    jurisdiction: str = "India"
    authority_or_court: str | None = None
    year: int | None = Field(default=None, ge=1800, le=2100)
    date: dt.date | None = None
    case_citation: str | None = None
    act_name: str | None = None
    status: Status = Status.in_force
    effective_from: dt.date | None = None
    superseded_by: str | None = None
    version: str = "1"
    language: str = "en"
    source_url: str | None = None
    source_type: SourceType = SourceType.pdf
    file_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    page_count: int | None = Field(default=None, ge=0)
    text_quality: TextQuality = TextQuality.unknown

    @field_validator("superseded_by")
    @classmethod
    def _slug(cls, v: str | None) -> str | None:
        return v


class ChunkMetadata(BaseModel):
    """Chunk-level fields recorded in the index payload."""

    doc_id: str
    chunk_id: str
    section_or_article_no: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    paragraph_no: str | None = None
    chunk_type: str = "text"
    parent_id: str | None = None
    config_fingerprint: str = ""


def check_consistency(rec: DocumentRecord) -> list[str]:
    """Cross-field validation errors for a record."""
    errs = []
    if rec.status == Status.superseded and not rec.superseded_by:
        errs.append(f"{rec.doc_id}: status=superseded requires superseded_by")
    if rec.superseded_by and rec.status == Status.in_force:
        errs.append(f"{rec.doc_id}: superseded_by set but status is in_force")
    if rec.doc_type == DocType.judgment and not rec.authority_or_court:
        errs.append(f"{rec.doc_id}: judgments need authority_or_court")
    return errs
