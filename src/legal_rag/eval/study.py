"""Retrieval studies: embedding bake-off and the B0 -> final ablation table."""

from __future__ import annotations

import json
import tempfile
import time
import tracemalloc
from pathlib import Path

from legal_rag.config.settings import Settings
from legal_rag.eval import metrics as M
from legal_rag.eval.golden import GoldenQuestion
from legal_rag.ingestion.indexer import Indexer
from legal_rag.ingestion.manifest import load_manifest
from legal_rag.retrieval.embedder import build_embedder
from legal_rag.retrieval.factory import build_retriever
from legal_rag.retrieval.memory_store import InMemoryVectorStore


def run_bakeoff(
    models: list[str],
    questions: list[GoldenQuestion],
    data_dir: Path,
    manifest: Path,
    settings: Settings,
) -> list[dict]:
    """Retrieval-only comparison on identical chunking (generation is never touched)."""
    answerable = [q for q in questions if q.answerable]
    rows = []
    for model in models:
        s = settings.model_copy(update={"embedding_model": model, "retrieval_mode": "dense"})
        tracemalloc.start()
        t0 = time.perf_counter()
        embedder = build_embedder(model)
        load_s = time.perf_counter() - t0
        store = InMemoryVectorStore()
        t1 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            Indexer(s, embedder, store, base_dir=data_dir, processed_dir=Path(tmp)).index_all(
                load_manifest(manifest)
            )
        index_s = time.perf_counter() - t1
        retriever = build_retriever(s, embedder, store)
        hits, recs, mrrs, ndcgs, lat = [], [], [], [], []
        for q in answerable:
            t = time.perf_counter()
            got = [r.doc_id for r in retriever.retrieve(q.question, k=8)]
            lat.append((time.perf_counter() - t) * 1e3)
            gold = set(q.gold_doc_ids)
            hits.append(M.hit_at_k(got, gold, 5))
            recs.append(M.recall_at_k(got, gold, 8))
            mrrs.append(M.mrr(got, gold))
            ndcgs.append(M.ndcg_at_k(got, gold, 5))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "model": model,
                "hit@5": M.mean(hits),
                "recall@8": M.mean(recs),
                "mrr": M.mean(mrrs),
                "ndcg@5": M.mean(ndcgs),
                "load_s": load_s,
                "index_s": index_s,
                "query_p50_ms": M.percentile(lat, 50),
                "peak_py_mb": peak / 1e6,
                "dim": len(embedder.embed(["x"])[0]),
                "chunks": store.count(),
            }
        )
    return rows


def render_bakeoff(rows: list[dict]) -> str:
    head = [
        "model",
        "dim",
        "hit@5",
        "recall@8",
        "mrr",
        "ndcg@5",
        "index_s",
        "query_p50_ms",
        "peak_py_mb",
    ]
    out = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for r in rows:
        cells = [r["model"], str(r["dim"])]
        cells += [f"{r[k]:.3f}" for k in ("hit@5", "recall@8", "mrr", "ndcg@5")]
        cells += [f"{r['index_s']:.2f}", f"{r['query_p50_ms']:.1f}", f"{r['peak_py_mb']:.1f}"]
        out.append("| " + " | ".join(cells) + " |")
    out.append(
        "\nPeak memory counts Python allocations only (tracemalloc); native model weights are "
        "not included, so measure process RSS separately for real models."
    )
    return "\n".join(out) + "\n"


def latest_runs(runs_dir: Path, split: str = "dev") -> dict[str, dict]:
    """Newest run JSON per config name for the given split."""
    found: dict[str, tuple[str, dict]] = {}
    for f in sorted(runs_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        meta = data["meta"]
        if meta["split"] == split:
            found[meta["config"]] = (f.name, data)  # sorted by timestamped name: last wins
    return {k: v[1] for k, v in found.items()}


def render_ablation(runs: dict[str, dict], order: list[str]) -> str:
    cols = ["hit@5", "recall@8", "mrr", "ndcg@5", "abstention_acc", "citation_valid"]
    out = [
        "| config | commit | " + " | ".join(cols) + " | d_mrr vs prev |",
        "|---" * (len(cols) + 3) + "|",
    ]
    prev = None
    for name in order:
        run = runs.get(name)
        if run is None:
            out.append(f"| {name} | (no run) |" + " |" * (len(cols) + 1))
            continue
        o = run["summary"]["overall"]
        delta = "" if prev is None else f"{o['mrr'] - prev:+.3f}"
        out.append(
            f"| {name} | {run['meta']['git_commit']} | "
            + " | ".join(f"{o[c]:.3f}" for c in cols)
            + f" | {delta} |"
        )
        prev = o["mrr"]
    return "\n".join(out) + "\n"


def tune_abstain(
    rows: list[tuple[bool, float]], thresholds: list[float] | None = None
) -> list[dict]:
    """Threshold sweep on (answerable, best_dense_score) pairs from the DEV split.

    A question is abstained when its best score is below the threshold. Reports how many
    unanswerable questions are correctly refused (recall) and how many answerable ones are
    wrongly refused (false refusals).
    """
    thresholds = thresholds if thresholds is not None else [i / 20 for i in range(0, 21)]
    unans = [s for a, s in rows if not a]
    ans = [s for a, s in rows if a]
    out = []
    for t in thresholds:
        refused_unans = sum(1 for s in unans if s < t)
        refused_ans = sum(1 for s in ans if s < t)
        total_refused = refused_unans + refused_ans
        out.append(
            {
                "threshold": t,
                "unanswerable_refused": refused_unans / len(unans) if unans else 0.0,
                "answerable_wrongly_refused": refused_ans / len(ans) if ans else 0.0,
                "precision": refused_unans / total_refused if total_refused else 1.0,
                "accuracy": (refused_unans + len(ans) - refused_ans) / max(len(rows), 1),
            }
        )
    return out


def recommend_threshold(table: list[dict], max_false_refusal: float = 0.1) -> dict | None:
    """Highest unanswerable-refusal rate with false refusals within budget (ties: higher t)."""
    ok = [r for r in table if r["answerable_wrongly_refused"] <= max_false_refusal]
    return max(ok, key=lambda r: (r["unanswerable_refused"], r["threshold"])) if ok else None


def render_abstain_table(table: list[dict], pick: dict | None) -> str:
    out = [
        "| threshold | unanswerable refused | answerable wrongly refused | precision | accuracy |",
        "|---|---|---|---|---|",
    ]
    for r in table:
        mark = " **<- recommended**" if pick is r else ""
        out.append(
            f"| {r['threshold']:.2f} | {r['unanswerable_refused']:.2f} | "
            f"{r['answerable_wrongly_refused']:.2f} | {r['precision']:.2f} | "
            f"{r['accuracy']:.2f} |{mark}"
        )
    return "\n".join(out) + "\n"
