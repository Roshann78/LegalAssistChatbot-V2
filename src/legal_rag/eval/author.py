"""Golden-set authoring: LLM drafts from sampled chunks -> human review -> explicit promotion.

Drafts never go straight into dev.jsonl/test.jsonl. A human verifies every question and gold
reference against the source PDF; `promote` then validates and appends accepted items.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.eval.golden import GoldenQuestion, lock_path
from legal_rag.generation.prompts import PROMPT_DIR
from legal_rag.llm import LLMClient
from legal_rag.retrieval.base import VectorStore
from legal_rag.retrieval.types import Chunk

PROTECTED = {"dev.jsonl", "test.jsonl"}
HINTS = {
    "section_lookup": "A question asking what a specific section/article says.",
    "conceptual": "A plain-language conceptual question.",
    "definitions": "A question about a definition, number or threshold.",
    "old_vs_new": "A question comparing old law and its replacement.",
    "case_law": "A question about the legal principle decided.",
    "banking_procedural": "A practical procedure question a bank customer would ask.",
    "multi_hop": "A question that needs both passages.",
    "out_of_scope": "A legal/banking-sounding question the corpus cannot answer.",
    "adversarial": "An ambiguous question or one containing an instruction to ignore its rules.",
}
ELIGIBLE_TYPES = {
    "section_lookup": {"statute", "constitution", "rule"},
    "conceptual": {"statute", "constitution", "rule", "scheme"},
    "definitions": {"statute", "constitution", "rule", "master_direction"},
    "old_vs_new": {"statute", "mapping_table"},
    "case_law": {"judgment"},
    "banking_procedural": {"master_direction", "circular", "scheme", "faq"},
}
_JSON = re.compile(r"\{.*\}", re.S)


def _doc_chunks(store: VectorStore) -> dict[str, list[Chunk]]:
    return {d: store.get_chunks(d) for d in store.list_docs()}


def sample_chunks(
    store: VectorStore, category: str, n: int, seed: int = 0, min_chars: int = 120
) -> list[Chunk]:
    """Stratified round-robin over documents so long judgments cannot dominate."""
    rng = random.Random(seed)
    by_doc: dict[str, list[Chunk]] = {}
    for doc, chunks in _doc_chunks(store).items():
        ok = [
            c
            for c in chunks
            if c.metadata.get("doc_type") in ELIGIBLE_TYPES.get(category, set())
            and len(c.text) >= min_chars
            and c.metadata.get("chunk_type") != "toc"
        ]
        if ok:
            rng.shuffle(ok)
            by_doc[doc] = ok
    picked: list[Chunk] = []
    docs = sorted(by_doc)
    rng.shuffle(docs)
    while len(picked) < n and any(by_doc.values()):
        for d in docs:
            if by_doc[d] and len(picked) < n:
                picked.append(by_doc[d].pop())
    return picked


def _json_of(raw: str) -> dict | None:
    m = _JSON.search(raw or "")
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def _location(c: Chunk) -> str:
    m = c.metadata
    return m.get("section_or_article_no") or (
        f"paragraphs {m['paragraph_no']}"
        if m.get("paragraph_no")
        else f"page {m.get('page_start')}"
    )


def draft_from_chunk(llm: LLMClient, chunk: Chunk, category: str, idx: int) -> dict | None:
    prompt = (
        (PROMPT_DIR / "author_v1.txt")
        .read_text(encoding="utf-8")
        .format(
            category=category,
            category_hint=HINTS[category],
            title=chunk.metadata.get("title", chunk.doc_id),
            location=_location(chunk),
            passage=chunk.text[:2500],
        )
    )
    data = _json_of(llm.generate([{"role": "user", "content": prompt}], temperature=0.3).content)
    if not data or not str(data.get("question", "")).strip():
        return None
    return {
        "id": f"draft-{category}-{idx:03d}",
        "question": str(data["question"]).strip(),
        "category": category,
        "reference_answer": str(data.get("reference_answer", "")).strip(),
        "gold_doc_ids": [chunk.doc_id],
        "gold_locations": [_location(chunk)],
        "answerable": True,
        "difficulty": "medium",
        "split": "dev",
        "domain": chunk.metadata.get("domain"),
        "source_chunk_id": chunk.chunk_id,
        "source_text": chunk.text,
    }


def draft_unanswerable(
    llm: LLMClient, titles: list[str], category: str, n: int, start: int = 0
) -> list[dict]:
    prompt = (
        (PROMPT_DIR / "author_unanswerable_v1.txt")
        .read_text(encoding="utf-8")
        .format(
            corpus_titles="; ".join(titles[:60]),
            category=category,
            category_hint=HINTS[category],
            n=n,
        )
    )
    data = _json_of(llm.generate([{"role": "user", "content": prompt}], temperature=0.7).content)
    qs = (data or {}).get("questions", [])
    return [
        {
            "id": f"draft-{category}-{start + i:03d}",
            "question": str(q).strip(),
            "category": category,
            "reference_answer": "",
            "gold_doc_ids": [],
            "gold_locations": [],
            "answerable": False,
            "difficulty": "medium",
            "split": "dev",
            "source_chunk_id": None,
            "source_text": "",
        }
        for i, q in enumerate(qs[:n])
        if str(q).strip()
    ]


def draft_all(
    llm: LLMClient, store: VectorStore, per_category: dict[str, int], seed: int = 0
) -> list[dict]:
    drafts: list[dict] = []
    titles = sorted(
        {c.metadata.get("title", "") for d in store.list_docs() for c in store.get_chunks(d)[:1]}
    )
    for cat, n in per_category.items():
        if cat in ("out_of_scope", "adversarial"):
            drafts += draft_unanswerable(llm, titles, cat, n)
        elif cat == "multi_hop":
            by_doc = _doc_chunks(store)
            rng = random.Random(seed)
            for i in range(n):
                docs = rng.sample(sorted(by_doc), k=min(2, len(by_doc)))
                chunks = [rng.choice(by_doc[d]) for d in docs if by_doc[d]]
                if len(chunks) == 2:
                    joined = Chunk(
                        chunks[0].chunk_id,
                        chunks[0].doc_id,
                        "\n\n---\n\n".join(c.text for c in chunks),
                        "",
                        chunks[0].metadata,
                    )
                    d = draft_from_chunk(llm, joined, cat, i)
                    if d:
                        d["gold_doc_ids"] = [c.doc_id for c in chunks]
                        d["gold_locations"] = [_location(c) for c in chunks]
                        drafts.append(d)
        else:
            for i, c in enumerate(sample_chunks(store, cat, n, seed)):
                d = draft_from_chunk(llm, c, cat, i)
                if d:
                    drafts.append(d)
    return drafts


def write_drafts(drafts: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"drafts_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.jsonl"
    if path.name in PROTECTED:
        raise ValueError("refusing to write drafts into a golden split file")
    path.write_text("\n".join(json.dumps(d) for d in drafts) + "\n", encoding="utf-8")
    return path


def review(
    drafts_path: Path,
    accepted_path: Path,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    """Show each draft beside its source passage: [a]ccept / [e]dit / [r]eject / [q]uit."""
    if accepted_path.name in PROTECTED:
        raise ValueError("accepted items go to a review file; use `promote` for dev/test")
    counts = defaultdict(int)
    for line in drafts_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out(f"\n=== {d['id']} [{d['category']}] answerable={d['answerable']}")
        out(
            f"Q: {d['question']}\nREF: {d['reference_answer']}\nGOLD: {d['gold_doc_ids']} "
            f"{d['gold_locations']}"
        )
        if d.get("source_text"):
            out(f"--- source ({d.get('source_chunk_id')}) ---\n{d['source_text'][:1200]}")
        choice = input_fn("[a]ccept / [e]dit / [r]eject / [q]uit > ").strip().lower()
        if choice == "q":
            break
        if choice == "e":
            d["question"] = input_fn("new question (blank = keep) > ").strip() or d["question"]
            d["reference_answer"] = (
                input_fn("new reference answer (blank = keep) > ").strip() or d["reference_answer"]
            )
            choice = "a"
        if choice == "a":
            d["verified_by_human"] = True
            with accepted_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(d) + "\n")
            counts["accepted"] += 1
        else:
            counts["rejected"] += 1
    return dict(counts)


def promote(accepted_path: Path, target: Path, split: str) -> int:
    """Validate human-accepted items and append them to dev.jsonl/test.jsonl."""
    if split == "test" and lock_path(target).exists():
        raise RuntimeError(f"{target} is frozen; new test questions need a new versioned file")
    existing = (
        {json.loads(ln)["id"] for ln in target.read_text(encoding="utf-8").splitlines() if ln}
        if target.exists()
        else set()
    )
    lines, n = [], len(existing)
    for line in accepted_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if not d.get("verified_by_human"):
            raise ValueError(f"{d['id']} was not verified by a human")
        n += 1
        d["id"] = f"{split}-{d['category']}-{n:03d}"
        d["split"] = split
        for k in ("source_chunk_id", "source_text", "verified_by_human"):
            d.pop(k, None)
        GoldenQuestion(**d)  # validates schema
        lines.append(json.dumps(d))
    with target.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main(argv: list[str] | None = None) -> int:
    from legal_rag.config.settings import Settings
    from legal_rag.llm import GroqClient
    from legal_rag.retrieval.embedder import build_embedder
    from legal_rag.retrieval.factory import build_store

    ap = argparse.ArgumentParser(prog="legal-rag-author")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--per-category", type=int, default=5)
    d.add_argument("--categories", nargs="*", default=list(HINTS))
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out-dir", type=Path, default=Path("eval/golden/drafts"))
    r = sub.add_parser("review")
    r.add_argument("drafts", type=Path)
    r.add_argument("--accepted", type=Path, default=Path("eval/golden/drafts/accepted.jsonl"))
    p = sub.add_parser("promote")
    p.add_argument("--accepted", type=Path, default=Path("eval/golden/drafts/accepted.jsonl"))
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--target", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "review":
        print(review(args.drafts, args.accepted))
        return 0
    if args.cmd == "promote":
        target = args.target or Path(f"eval/golden/{args.split}.jsonl")
        print(f"promoted {promote(args.accepted, target, args.split)} questions to {target}")
        return 0
    settings = Settings()
    store = build_store(settings, build_embedder(settings.embedding_model))
    if store.count() == 0:
        print("ERROR: index is empty; run ingestion first", file=sys.stderr)
        return 2
    drafts = draft_all(
        GroqClient(settings), store, {c: args.per_category for c in args.categories}, args.seed
    )
    print(f"{len(drafts)} drafts -> {write_drafts(drafts, args.out_dir)} (review before use)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
