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


def _tokens(text: str) -> frozenset[str]:
    return frozenset(text.lower().split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def order_passages(results: list[SearchResult], order: str) -> list[SearchResult]:
    """'relevance' keeps rank order. 'edges' puts the best passages first AND last, weakest in the
    middle, to counter 'lost in the middle'."""
    if order != "edges" or len(results) < 3:
        return results
    head, tail = results[0::2], results[1::2]
    return head + tail[::-1]


def build_messages(
    version: str,
    question: str,
    results: list[SearchResult],
    max_context_chars: int = 12000,
    *,
    order: str = "relevance",
    dedupe_threshold: float = 0.9,
) -> tuple[list[dict[str, str]], list[SearchResult]]:
    """Numbered passages (near-duplicates removed, budgeted) in the system prompt.

    Deduplication runs on rank order (so the better copy wins), then `order` is applied.
    Returns (messages, kept) where kept[i-1] is the passage numbered [i] in the prompt.
    """
    unique: list[SearchResult] = []
    sigs: list[frozenset[str]] = []
    for r in results:
        sig = _tokens(r.text)
        if any(_jaccard(sig, s) >= dedupe_threshold for s in sigs):
            continue
        sigs.append(sig)
        unique.append(r)
    blocks, kept, used = [], [], 0
    for r in unique:
        block = format_passage(len(kept) + 1, r)
        if used + len(block) > max_context_chars and blocks:
            break
        blocks.append(block)
        kept.append(r)
        used += len(block)
    kept = order_passages(kept, order)
    blocks = [format_passage(i, r) for i, r in enumerate(kept, 1)]
    system = load_prompt(version).format(context="\n\n".join(blocks))
    return [{"role": "system", "content": system}, {"role": "user", "content": question}], kept
