"""Phase 4: retrieval upgrades (weights, rerank, parent-child, router, QU flags) and studies."""

from __future__ import annotations

import asyncio
import warnings

import pytest
from qdrant_client import QdrantClient

from legal_rag.config.settings import Settings
from legal_rag.eval.golden import load_golden
from legal_rag.eval.runner import CONFIGS, ExtractiveFakeLLM, build_pipeline, evaluate, summarize
from legal_rag.eval.runner import main as eval_main
from legal_rag.eval.study import latest_runs, render_ablation, render_bakeoff, run_bakeoff
from legal_rag.generation.pipeline import RAGPipeline
from legal_rag.retrieval.factory import build_retriever
from legal_rag.retrieval.hybrid import Retriever
from legal_rag.retrieval.qdrant_store import QdrantVectorStore
from legal_rag.retrieval.query import route_domain, section_labels, understand
from legal_rag.retrieval.rerank import LexicalReranker, build_reranker
from legal_rag.retrieval.types import Filters, SearchResult
from tests.conftest import FIXTURES

SMOKE = FIXTURES.parent / "golden" / "smoke.jsonl"


def _res(cid, text, **kw) -> SearchResult:
    return SearchResult(cid, kw.get("doc", "d"), "T", kw.get("section"), 1, text, 0.0)


# ── retriever knobs ──
def test_weights_shift_ranking_between_dense_and_sparse(indexed):
    embedder, store, _, _ = indexed
    q = "theft punishment imprisonment"
    dense_only = Retriever(embedder, store, mode="dense").retrieve(q, k=4)
    sparse_only = Retriever(embedder, store, mode="sparse").retrieve(q, k=4)
    heavy_sparse = Retriever(embedder, store, dense_weight=0.0).retrieve(q, k=4)
    assert [r.chunk_id for r in heavy_sparse][:2] == [r.chunk_id for r in sparse_only][:2]
    assert dense_only and sparse_only


def test_lexical_reranker_reorders_and_respects_k():
    rr = LexicalReranker()
    pool = [_res("a", "unrelated words here"), _res("b", "theft punishment text", section="S4")]
    out = rr.rerank("punishment for theft", pool, k=1)
    assert [r.chunk_id for r in out] == ["b"] and out[0].score > 0
    assert rr.rerank("", pool, 5) == pool


def test_build_reranker_choices():
    assert build_reranker("none", "") is None
    assert isinstance(build_reranker("lexical", ""), LexicalReranker)
    with pytest.raises(ValueError, match="RERANK_MODEL"):
        build_reranker("cross_encoder", "")


def test_retriever_with_reranker_records_latency_and_keeps_dense_score(indexed):
    embedder, store, _, _ = indexed
    r = Retriever(embedder, store, reranker=LexicalReranker(), rerank_top_n=10)
    res = r.retrieve("punishment for theft", k=3)
    assert len(res) <= 3 and r.last_rerank_ms >= 0
    assert any(x.dense_score > 0 for x in res)  # abstention gate still sees dense scores


def test_parent_expand_adds_neighbouring_context(indexed):
    embedder, store, _, _ = indexed
    plain = Retriever(embedder, store).retrieve("Section 3 theft dishonestly", k=1)[0]
    parent = Retriever(embedder, store, parent_expand=True).retrieve(
        "Section 3 theft dishonestly", k=1
    )[0]
    assert len(parent.text) > len(plain.text) and plain.text in parent.text
    assert parent.metadata["child_chunk_id"] == plain.chunk_id
    capped = Retriever(embedder, store, parent_expand=True, parent_max_chars=200).retrieve(
        "Section 3 theft", k=1
    )[0]
    assert len(capped.text) <= 200


def test_section_filter_in_memory_and_qdrant(indexed):
    embedder, store, indexer, records = indexed
    vec = embedder.embed(["anything"])[0]
    hits = store.dense_search(vec, Filters(sections=("Section 4",)), 10)
    assert {h.section for h in hits} == {"Section 4"}
    q = QdrantVectorStore(QdrantClient(":memory:"), "t", 512)
    q.check_fingerprint("x")
    for rec in records:
        for c in store.get_chunks(rec.doc_id):
            q.upsert([c], [embedder.embed([c.to_embed])[0]])
    assert {h.section for h in q.dense_search(vec, Filters(sections=("Section 4",)), 10)} == {
        "Section 4"
    }


# ── query understanding components ──
def test_route_domain_only_when_clear():
    assert route_domain("Which bank handles KYC for my account?") == "banking"
    assert route_domain("What is the punishment under section 4 of the IPC?") == "legal"
    assert route_domain("hello there") is None
    assert route_domain("bank account bail section") is None  # mixed signal -> no routing


def test_section_labels_and_component_switches():
    assert section_labels([("section", "66a"), ("article", "21")]) == ("Article 21", "Section 66A")
    q = "What is Section 4 of the IPC?"
    on = understand(q)
    assert on.references and "Indian Penal Code" in on.expansion
    off = understand(q, refs=False, abbrev=False)
    assert off.references == [] and off.expansion == ""
    hist = [("What is theft?", "x")]
    assert understand("what about its punishment?", hist, rewrite=False).standalone == (
        "what about its punishment?"
    )


# ── pipeline wiring ──
def test_exact_section_reference_is_pinned_first(settings, indexed):
    embedder, store, _, _ = indexed
    s = settings.model_copy(update={"qu_refs": True, "retrieval_mode": "dense"})
    pipe = RAGPipeline(build_retriever(s, embedder, store), ExtractiveFakeLLM(), s)
    prepared = pipe.prepare("Please explain Section 5 of the code")
    assert prepared.results[0].section == "Section 5"


def test_domain_router_restricts_to_banking(settings, indexed):
    embedder, store, _, _ = indexed
    s = settings.model_copy(update={"domain_router": True})
    pipe = RAGPipeline(build_retriever(s, embedder, store), ExtractiveFakeLLM(), s)
    prepared = pipe.prepare("What does the bank require for KYC of a customer?")
    assert {r.doc_id for r in prepared.results} == {"sample_kyc_direction"}
    assert "rerank_ms" in prepared.timings


def test_wrong_route_falls_back_instead_of_missing(settings, indexed, monkeypatch):
    embedder, store, _, _ = indexed
    s = settings.model_copy(update={"domain_router": True})
    pipe = RAGPipeline(build_retriever(s, embedder, store), ExtractiveFakeLLM(), s)
    monkeypatch.setattr("legal_rag.generation.pipeline.route_domain", lambda q: "no_such_domain")
    prepared = pipe.prepare("What is the punishment for theft?")
    assert prepared.results and prepared.results[0].doc_id == "sample_penal_code"


def test_settings_validation_for_new_options():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            Settings(retrieval_mode="fuzzy")
        with pytest.raises(ValueError):
            Settings(rerank_strategy="magic")


# ── experiment configs and studies ──
def test_every_experiment_config_is_valid_and_runs(settings):
    qs = load_golden(SMOKE)
    for name, cfg in CONFIGS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Settings(**cfg)  # raises on a typo / invalid value
        pipe = build_pipeline(
            name, settings, ExtractiveFakeLLM(), FIXTURES, FIXTURES / "manifest.yaml"
        )
        rows = asyncio.run(evaluate(pipe, qs))
        overall = summarize(rows)["overall"]
        assert overall["hit@5"] >= 0.75, (name, overall["hit@5"])


def test_configs_differ_in_exactly_the_intended_knob():
    def diff(a: str, b: str) -> set[str]:
        return {
            k
            for k in CONFIGS[a].keys() | CONFIGS[b].keys()
            if CONFIGS[a].get(k) != CONFIGS[b].get(k)
        }

    assert diff("B0_baseline", "B1_structure") == {"chunk_strategy"}
    assert diff("B1_structure", "B1b_prefix") == {"contextual_prefix"}
    assert diff("B1b_prefix", "B1c_parent") == {"parent_expand"}
    assert diff("B1b_prefix", "B3_hybrid") == {"retrieval_mode"}
    assert diff("B3_hybrid", "B4_filters") == {"domain_router"}
    assert diff("B3_hybrid", "B5_rerank") == {"rerank_strategy"}
    assert diff("B3_hybrid", "B6a_rewrite") == {"qu_rewrite"}


def test_bakeoff_reports_all_models(settings):
    qs = load_golden(SMOKE)
    rows = run_bakeoff(
        ["hashing-128", "hashing-512"], qs, FIXTURES, FIXTURES / "manifest.yaml", settings
    )
    assert [r["model"] for r in rows] == ["hashing-128", "hashing-512"]
    assert rows[0]["dim"] == 128 and all(r["hit@5"] > 0 and r["peak_py_mb"] > 0 for r in rows)
    text = render_bakeoff(rows)
    assert "hashing-512" in text and "tracemalloc" in text


def test_ablation_table_from_saved_runs(tmp_path, capsys):
    base = [
        "--golden",
        str(SMOKE),
        "--data-dir",
        str(FIXTURES),
        "--manifest",
        str(FIXTURES / "manifest.yaml"),
        "--embedder",
        "hashing-512",
        "--out",
        str(tmp_path),
    ]
    for cfg in ("B0_baseline", "B3_hybrid"):
        assert eval_main(["run", "--config", cfg, *base]) == 0
    assert set(latest_runs(tmp_path)) == {"B0_baseline", "B3_hybrid"}
    out = tmp_path / "ablation.md"
    assert (
        eval_main(
            [
                "ablation",
                "--runs",
                str(tmp_path),
                "--order",
                "B0_baseline",
                "B3_hybrid",
                "R_final",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    text = out.read_text()
    assert "B0_baseline" in text and "(no run)" in text and "d_mrr" in text
    assert render_ablation({}, ["B0_baseline"]).count("(no run)") == 1
