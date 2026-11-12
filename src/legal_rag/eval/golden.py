"""Golden-set schema and loader (JSONL)."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Category(StrEnum):
    section_lookup = "section_lookup"
    conceptual = "conceptual"
    case_law = "case_law"
    old_vs_new = "old_vs_new"
    multi_hop = "multi_hop"
    banking_procedural = "banking_procedural"
    definitions = "definitions"
    out_of_scope = "out_of_scope"
    adversarial = "adversarial"


class GoldenQuestion(BaseModel):
    id: str
    question: str = Field(min_length=3)
    category: Category
    reference_answer: str = ""
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_locations: list[str] = Field(default_factory=list)
    answerable: bool = True
    difficulty: str = "medium"
    split: str = Field(default="dev", pattern="^(dev|test)$")
    domain: str | None = None

    @model_validator(mode="after")
    def _answerable_needs_gold(self) -> GoldenQuestion:
        if self.answerable and not self.gold_doc_ids:
            raise ValueError(f"{self.id}: answerable questions need gold_doc_ids")
        return self


def load_golden(path: Path, split: str | None = None) -> list[GoldenQuestion]:
    qs, ids = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        q = GoldenQuestion(**json.loads(line))
        if q.id in ids:
            raise ValueError(f"{path}:{n}: duplicate id {q.id}")
        ids.add(q.id)
        if split is None or q.split == split:
            qs.append(q)
    return qs
