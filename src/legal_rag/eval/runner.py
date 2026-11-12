"""Eval runner: run a named pipeline config over a golden split; report, compare, gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from legal_rag.config.settings import Settings
from legal_rag.eval import metrics as M
from legal_rag.eval.golden import GoldenQuestion, load_golden
from legal_rag.generation.pipeline import Answer, RAGPipeline
from legal_rag.ingestion.indexer import Indexer
from legal_rag.ingestion.manifest import load_manifest
from legal_rag.llm import LLMClient, LLMResponse
from legal_rag.retrieval.embedder import build_embedder
from legal_rag.retrieval.hybrid import Retriever
from legal_rag.retrieval.memory_store import InMemoryVectorStore
from legal_rag.retrieval.query import MappingTable

# name -> settings overrides + retrieval mode. B0 is the baseline; later rows change one thing.
CONFIGS: dict[str, dict] = {
    "B0_baseline": {"mode": "dense", "chunk_strategy": "s0", "prompt_version": "v1"},
    "B1_structure": {"mode": "dense", "chunk_strategy": "auto", "prompt_version": "v1"},
    "B3_hybrid": {"mode": "hybrid", "chunk_strategy": "auto", "prompt_version": "v1"},
    "G1_prompt_v2": {
        "mode": "hybrid",
        "chunk_strategy": "auto",
        "prompt_version": "v2",
        "abstain_threshold": 0.2,
    },
}

AnswerFn = Callable[[str], Awaitable[Answer]]


class ExtractiveFakeLLM(LLMClient):
    """Offline LLM: quotes the first passage and cites it. Used for CI/smoke evals."""

    def is_ready(self) -> bool:
        return True

    def generate(self, messages, **kw) -> LLMResponse:
        return LLMResponse(content=self._reply(messages), model="extractive-fake")

    def _reply(self, messages) -> str:
        sys_msg = messages[0]["content"]
        body = sys_msg.split("[1]", 1)[-1].split("\n", 1)[-1].split("\n\n")[0][:300]
        return f"{body.strip()} [1]"

    async def generate_stream(self, messages, **kw) -> AsyncIterator[str]:
        yield self._reply(messages)


def build_pipeline(
    config: str, settings: Settings, llm: LLMClient, data_dir: Path, manifest: Path
) -> RAGPipeline:
    cfg = CONFIGS[config]
    s = settings.model_copy(
        update={
            "chunk_strategy": cfg["chunk_strategy"],
            "prompt_version": cfg["prompt_version"],
            "abstain_threshold": cfg.get("abstain_threshold", 0.0),
        }
    )
    embedder = build_embedder(s.embedding_model)
    store = InMemoryVectorStore()
    Indexer(s, embedder, store, base_dir=data_dir).index_all(load_manifest(manifest))
    retriever = Retriever(
        embedder,
        store,
        mode=cfg["mode"],
        candidates=s.retrieval_candidates,
        per_doc_cap=s.retrieval_per_doc_cap,
    )
    return RAGPipeline(retriever, llm, s, MappingTable())


async def evaluate(
    pipeline: RAGPipeline, questions: list[GoldenQuestion], k: int = 5
) -> list[dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        ans = await pipeline.answer(q.question, domain=q.domain)
        latency = (time.perf_counter() - t0) * 1e3
        retrieved = [r.doc_id for r in ans.results]
        gold = set(q.gold_doc_ids)
        rows.append(
            {
                "id": q.id,
                "category": q.category.value,
                "answerable": q.answerable,
                "question": q.question,
                "answer": ans.text,
                "abstained": ans.abstained,
                "retrieved": retrieved,
                "sections": [r.section for r in ans.results],
                "citations": [c.to_dict() for c in ans.citations],
                "invalid_citations": ans.invalid_citations,
                "hit": M.hit_at_k(retrieved, gold, k),
                "recall": M.recall_at_k(retrieved, gold, 8),
                "mrr": M.mrr(retrieved, gold),
                "ndcg": M.ndcg_at_k(retrieved, gold, k),
                "location_hit": M.location_hit([r.section for r in ans.results], q.gold_locations),
                "abstention_correct": ans.abstained == (not q.answerable),
                "citation_valid": float(
                    not ans.abstained and bool(ans.citations) and not ans.invalid_citations
                ),
                "latency_ms": latency,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    def agg(rs: list[dict]) -> dict:
        ans = [r for r in rs if r["answerable"]]
        non_abstained = [r for r in ans if not r["abstained"]]
        return {
            "n": len(rs),
            "hit@5": M.mean([r["hit"] for r in ans]),
            "recall@8": M.mean([r["recall"] for r in ans]),
            "mrr": M.mean([r["mrr"] for r in ans]),
            "ndcg@5": M.mean([r["ndcg"] for r in ans]),
            "abstention_acc": M.mean([float(r["abstention_correct"]) for r in rs]),
            "citation_valid": M.mean([r["citation_valid"] for r in non_abstained]),
            "p50_ms": M.percentile([r["latency_ms"] for r in rs], 50),
            "p95_ms": M.percentile([r["latency_ms"] for r in rs], 95),
        }

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {"overall": agg(rows), "by_category": {c: agg(rs) for c, rs in sorted(by_cat.items())}}


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def render_report(meta: dict, summary: dict, rows: list[dict]) -> str:
    cols = [
        "n",
        "hit@5",
        "recall@8",
        "mrr",
        "ndcg@5",
        "abstention_acc",
        "citation_valid",
        "p50_ms",
        "p95_ms",
    ]

    def line(name: str, d: dict) -> str:
        cells = [f"{d[c]:.0f}" if c in ("n", "p50_ms", "p95_ms") else f"{d[c]:.3f}" for c in cols]
        return f"| {name} | " + " | ".join(cells) + " |"

    out = [
        f"# Eval run: {meta['config']} ({meta['split']})",
        "",
        f"- commit: `{meta['git_commit']}`  fingerprint: `{meta['fingerprint']}`",
        f"- prompt: {meta['prompt_version']}  llm: {meta['llm']}  embedder: {meta['embedder']}",
        "",
        "| category | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
        line("**overall**", summary["overall"]),
    ]
    out += [line(c, d) for c, d in summary["by_category"].items()]
    worst = sorted(
        rows, key=lambda r: (r["mrr"] if r["answerable"] else 1.0, float(r["abstention_correct"]))
    )[:10]
    out += ["", "## Worst 10", ""]
    out += [
        f"- `{r['id']}` ({r['category']}): mrr={r['mrr']:.2f} abstain_ok="
        f"{r['abstention_correct']} — {r['question'][:90]}"
        for r in worst
    ]
    return "\n".join(out) + "\n"


def compare(a: dict, b: dict) -> str:
    keys = ["hit@5", "recall@8", "mrr", "ndcg@5", "abstention_acc", "citation_valid"]
    lines = [
        f"| metric | {a['meta']['config']} | {b['meta']['config']} | delta |",
        "|---|---|---|---|",
    ]
    for k in keys:
        x, y = a["summary"]["overall"][k], b["summary"]["overall"][k]
        lines.append(f"| {k} | {x:.3f} | {y:.3f} | {y - x:+.3f} |")
    return "\n".join(lines)


def check_thresholds(summary: dict, thresholds: dict[str, float]) -> list[str]:
    return [
        f"{k}={summary['overall'][k]:.3f} < {v}"
        for k, v in thresholds.items()
        if summary["overall"][k] < v
    ]


def append_log(path: Path, meta: dict, overall: dict) -> None:
    row = (
        f"| {meta['config']} | {meta['date']} | {meta['config']} ({meta['split']}) | "
        f"{overall['hit@5']:.3f} | {overall['recall@8']:.3f} | {overall['mrr']:.3f} | | | "
        f"{overall['abstention_acc']:.3f} | {overall['citation_valid']:.3f} | "
        f"{overall['p50_ms']:.0f} / {overall['p95_ms']:.0f} ms | | | "
        f"commit {meta['git_commit']} |\n"
    )
    with path.open("a", encoding="utf-8") as f:
        f.write(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="legal-rag-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="B0_baseline", choices=sorted(CONFIGS))
    r.add_argument("--split", default="dev", choices=["dev", "test"])
    r.add_argument("--golden", type=Path, default=Path("eval/golden/dev.jsonl"))
    r.add_argument("--data-dir", type=Path, default=Path("data"))
    r.add_argument("--manifest", type=Path, default=Path("data/manifest.yaml"))
    r.add_argument("--embedder", default=None)
    r.add_argument("--out", type=Path, default=Path("eval/runs"))
    r.add_argument("--log", action="store_true", help="append row to experiment_log.md")
    r.add_argument(
        "--min",
        nargs="*",
        default=[],
        metavar="METRIC=VALUE",
        help="fail (exit 1) if overall metric is below value",
    )
    c = sub.add_parser("compare")
    c.add_argument("run_a", type=Path)
    c.add_argument("run_b", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        print(compare(json.loads(args.run_a.read_text()), json.loads(args.run_b.read_text())))
        return 0

    settings = Settings(embedding_model=args.embedder) if args.embedder else Settings()
    llm: LLMClient = ExtractiveFakeLLM()
    pipeline = build_pipeline(args.config, settings, llm, args.data_dir, args.manifest)
    questions = load_golden(args.golden, args.split)
    rows = asyncio.run(evaluate(pipeline, questions))
    summary = summarize(rows)
    fingerprint = pipeline.retriever.store.fingerprint
    meta = {
        "config": args.config,
        "split": args.split,
        "git_commit": git_commit(),
        "fingerprint": fingerprint,
        "prompt_version": pipeline.s.prompt_version,
        "llm": "extractive-fake",
        "embedder": pipeline.retriever.embedder.name,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    base = args.out / f"{stamp}_{args.config}"
    base.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    report = render_report(meta, summary, rows)
    base.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report)
    if args.log:
        append_log(Path("eval/experiment_log.md"), meta, summary["overall"])
    thresholds = {k: float(v) for k, v in (m.split("=") for m in args.min)}
    failures = check_thresholds(summary, thresholds)
    if failures:
        print("REGRESSION GATE FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
