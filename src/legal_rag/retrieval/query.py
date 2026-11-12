"""Query understanding: reference detection, abbreviation expansion, old->new law mapping."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ABBREVIATIONS = {
    "ipc": "Indian Penal Code",
    "bns": "Bharatiya Nyaya Sanhita",
    "crpc": "Code of Criminal Procedure",
    "bnss": "Bharatiya Nagarik Suraksha Sanhita",
    "iea": "Indian Evidence Act",
    "bsa": "Bharatiya Sakshya Adhiniyam",
    "rbi": "Reserve Bank of India",
    "kyc": "Know Your Customer",
    "rti": "Right to Information",
    "pmjdy": "Pradhan Mantri Jan Dhan Yojana",
}

_REF_RE = re.compile(r"\b(section|sec\.?|article)\s+(\d+[A-Za-z]{0,2})\b", re.I)
_ACT_RE = re.compile(r"\b(ipc|bns|crpc|bnss|iea|bsa)\b", re.I)


@dataclass
class UnderstoodQuery:
    original: str
    standalone: str
    references: list[tuple[str, str]] = field(default_factory=list)  # (kind, number)
    expansion: str = ""
    mapped: list[dict] = field(default_factory=list)

    @property
    def retrieval_text(self) -> str:
        return f"{self.standalone} {self.expansion}".strip()


class MappingTable:
    """Deterministic old->new section lookup. Only verified entries are used."""

    def __init__(self, entries: list[dict] | None = None) -> None:
        self.entries = entries or []

    @classmethod
    def load(cls, path: Path) -> MappingTable:
        if not path.exists():
            return cls()
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def lookup(self, old_act: str, old_section: str, verified_only: bool = True) -> list[dict]:
        return [
            e
            for e in self.entries
            if e["old_act"].lower() == old_act.lower()
            and str(e["old_section"]).lower() == old_section.lower()
            and (e.get("verified", False) or not verified_only)
        ]


def rewrite_followup(question: str, history: list[tuple[str, str]]) -> str:
    """Heuristic standalone rewrite: pronoun-led follow-ups inherit the last user subject."""
    if not history:
        return question
    pronoun = re.match(
        r"^\s*(what about|and|how about|what is|what are)?\s*.*\b(it|its|this|that|"
        r"they|their|the same)\b",
        question,
        re.I,
    )
    if pronoun and len(question.split()) <= 12:
        return f"{question.strip()} (regarding: {history[-1][0]})"
    return question


def understand(
    question: str,
    history: list[tuple[str, str]] | None = None,
    mapping: MappingTable | None = None,
) -> UnderstoodQuery:
    standalone = rewrite_followup(question, history or [])
    refs = [(m.group(1).lower().rstrip("."), m.group(2)) for m in _REF_RE.finditer(standalone)]
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", standalone)}
    expansion = " ".join(full for ab, full in ABBREVIATIONS.items() if ab in words)
    mapped: list[dict] = []
    acts = [a.lower() for a in _ACT_RE.findall(standalone)]
    if mapping and refs and acts:
        for _, num in refs:
            for act in acts:
                mapped.extend(mapping.lookup(act, num))
    if mapped:
        expansion += " " + " ".join(f"{m['new_act']} Section {m['new_section']}" for m in mapped)
    return UnderstoodQuery(question, standalone, refs, expansion.strip(), mapped)
