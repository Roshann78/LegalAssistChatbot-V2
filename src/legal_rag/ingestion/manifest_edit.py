"""Text-level manifest edits (append / remove / re-hash) that never rewrite user content.

The manifest is edited as plain text, not re-dumped from objects, so comments, ordering and any
manual edits (`status: superseded`, `source_url`, ...) on existing entries are always preserved.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_LIST_ITEM = re.compile(r"^(\s*)- ")
AUTO_NOTE = "auto-added by sync: doc_type, title and year are inferred; review and edit freely"


def append_entries(path: Path, entries: list[dict], note: str = AUTO_NOTE) -> None:
    """Append entries as raw YAML so existing content is untouched."""
    if not entries:
        return
    text = path.read_text(encoding="utf-8") if path.exists() else "documents:\n"
    text = re.sub(r"^(documents:)[ \t]*\[\s*\][ \t]*$", r"\1", text, flags=re.M)
    indent = ""
    for line in text.splitlines():
        m = _LIST_ITEM.match(line)
        if m:
            indent = m.group(1)
            break
    if not text.endswith("\n"):
        text += "\n"
    block = ""
    for e in entries:
        dumped = yaml.safe_dump([e], sort_keys=False, allow_unicode=True, width=1000)
        lines = [indent + ln for ln in dumped.splitlines()]
        if note:
            lines.insert(0, f"{indent}# {note}")
        block += "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + block, encoding="utf-8")


def _entry_start(lines: list[str], doc_id: str) -> int | None:
    pats = (
        re.compile(rf"^\s*- doc_id:\s*['\"]?{re.escape(doc_id)}['\"]?\s*$"),
        re.compile(rf"^\s*- \{{.*\bdoc_id:\s*['\"]?{re.escape(doc_id)}['\"]?\s*[,}}]"),
    )
    for i, ln in enumerate(lines):
        if any(p.match(ln) for p in pats):
            return i
    return None


def remove_entry(path: Path, doc_id: str) -> bool:
    """Delete one entry (block or flow style) and its preceding auto-note comment."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = _entry_start(lines, doc_id)
    if start is None:
        return False
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip():
            cur = len(ln) - len(ln.lstrip())
            if cur < indent or (cur == indent and ln.lstrip().startswith(("- ", "#"))):
                break
        end += 1
    if start > 0 and lines[start - 1].lstrip().startswith("# auto-added"):
        start -= 1
    path.write_text("".join(lines[:start] + lines[end:]), encoding="utf-8")
    return True


def replace_file_hash(path: Path, old: str, new: str) -> bool:
    """Refresh the machine-owned file_hash of one entry (a 64-hex value is unique in the file)."""
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        return False
    path.write_text(text.replace(old, new), encoding="utf-8")
    return True
