"""Versioned prompt loading and context assembly."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from legal_rag.retrieval.types import SearchResult

PROMPT_DIR = Path(__file__).resolve().parent.parent / "config" / "prompts"


@lru_cache(maxsize=8)
def load_prompt(version: str) -> str:
    path = PROMPT_DIR / f"answer_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt version {version!r} not found at {path}")
    return path.read_text(encoding="utf-8")


def _escape(text: str) -> str:
    """Neutralise attempts to close the data block from inside a passage."""
    return text.replace("</passages>", "&lt;/passages&gt;").replace(
        "<passages>", "&lt;passages&gt;"
    )


def format_passage(i: int, r: SearchResult) -> str:
    bits = [r.title]
    if r.section:
        bits.append(r.section)
    if r.page:
        bits.append(f"p.{r.page}")
    head = f"[{i}] {' | '.join(bits)}"
    if r.status in ("superseded", "repealed"):
        replaced = r.metadata.get("superseded_by") or "unknown"
        head += f" | SUPERSEDED (current: {replaced})"
    return f"{head}\n{_escape(r.text)}"


def build_messages(
    version: str,
    question: str,
    results: list[SearchResult],
    max_context_chars: int = 12000,
) -> tuple[list[dict[str, str]], list[SearchResult]]:
    """Numbered passages (deduplicated, budgeted) in the system prompt; question as user turn.

    Returns (messages, kept) where kept[i-1] is the passage numbered [i] in the prompt.
    """
    blocks, kept, used, seen = [], [], 0, set()
    for r in results:
        key = " ".join(r.text.split())[:200]
        if key in seen:
            continue
        block = format_passage(len(kept) + 1, r)
        if used + len(block) > max_context_chars and blocks:
            break
        seen.add(key)
        blocks.append(block)
        kept.append(r)
        used += len(block)
    system = load_prompt(version).format(context="\n\n".join(blocks))
    return [{"role": "system", "content": system}, {"role": "user", "content": question}], kept
