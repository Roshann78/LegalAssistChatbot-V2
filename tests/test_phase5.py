"""Phase 5: context assembly, citations, currency notes, abstention, faithfulness guard."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from legal_rag.api.main import create_app
from legal_rag.eval.golden import load_golden
from legal_rag.eval.judge import Judge, faithfulness_guard, split_claims
from legal_rag.eval.runner import CONFIGS, ExtractiveFakeLLM, evaluate, summarize
from legal_rag.eval.runner import main as eval_main
from legal_rag.eval.study import recommend_threshold, render_abstain_table, tune_abstain
from legal_rag.generation.citations import strip_invalid_citations
from legal_rag.generation.pipeline import abstain_message, currency_notes
from legal_rag.generation.prompts import build_messages, order_passages
from legal_rag.retrieval.query import MappingTable
from legal_rag.retrieval.types import SearchResult
from tests.conftest import FIXTURES
from tests.test_api_v1 import parse_sse
from tests.test_phase3 import ScriptedLLM

SMOKE = FIXTURES.parent / "golden" / "smoke.jsonl"


def _r(cid, text, **kw) -> SearchResult:
    return SearchResult(
        cid,
        kw.get("doc", cid),
        kw.get("title", "T"),
        None,
        1,
        text,
        0.0,
        status=kw.get("status", "in_force"),
        metadata=kw.get("meta", {}),
    )


# ── 5.4 context assembly ──
def test_near_duplicates_are_removed_but_distinct_passages_kept():
    base = "whoever commits theft shall be punished with imprisonment up to three years or fine"
    a, b = _r("a", base), _r("b", base + " .")  # near-identical
    c = _r("c", "banks shall update kyc records periodically for every risk category")
    _, kept = build_messages("v2", "q", [a, b, c])
    assert [r.chunk_id for r in kept] == ["a", "c"]
    _, loose = build_messages("v2", "q", [a, b, c], dedupe_threshold=1.01)
    assert len(loose) == 3


def test_edge_ordering_puts_best_first_and_last():
    rs = [_r(str(i), f"unique passage number {i} " + "x" * i) for i in range(1, 6)]
    assert [r.chunk_id for r in order_passages(rs, "relevance")] == list("12345")
    assert [r.chunk_id for r in order_passages(rs, "edges")] == ["1", "3", "5", "4", "2"]
    msgs, kept = build_messages("v2", "q", rs, order="edges")
    assert kept[0].chunk_id == "1" and kept[-1].chunk_id == "2"
    assert "[5]" in msgs[0]["content"]  # numbering follows prompt position, matches `kept`


def test_context_budget_limits_passages():
    rs = [_r(str(i), f"passage {i} " + "word " * 200) for i in range(6)]
    _, kept = build_messages("v2", "q", rs, max_context_chars=2500)
    assert 1 <= len(kept) < 6


# ── 5.2 citations ──
def test_strip_invalid_citation_markers():
    assert strip_invalid_citations("Yes [1][7] and [2, 9] and [8].", 3) == "Yes [1] and [2] and ."
    assert strip_invalid_citations("no markers", 3) == "no markers"


def test_api_sends_sanitized_answer_and_notes(settings, indexed):
    embedder, store, _, _ = indexed

    class BadCiter(ExtractiveFakeLLM):
        def _reply(self, messages):
            return "Theft is punished [1] and also something unsupported [9]."

    from legal_rag.generation.pipeline import RAGPipeline
    from legal_rag.retrieval.factory import build_retriever

    pipe = RAGPipeline(build_retriever(settings, embedder, store), BadCiter(), settings)
    app = create_app(
        settings, llm_client=BadCiter(), embedder=embedder, vector_store=store, pipeline=pipe
    )
    with TestClient(app) as c:
        events = dict(parse_sse(c.post("/v1/ask", json={"question": "theft punishment"}).text))
    cit = events["citations"]
    assert cit["invalid"] == [9] and "[9]" not in cit["answer"] and "[1]" in cit["answer"]
    assert [x["index"] for x in cit["citations"]] == [1] and "notes" in cit


def test_pipeline_answer_text_never_contains_invalid_markers(settings, indexed):
    embedder, store, _, _ = indexed
    from legal_rag.generation.pipeline import RAGPipeline
    from legal_rag.retrieval.factory import build_retriever

    class BadCiter(ExtractiveFakeLLM):
        def _reply(self, messages):
            return "Claim [1][42]."

    pipe = RAGPipeline(build_retriever(settings, embedder, store), BadCiter(), settings)
    ans = asyncio.run(pipe.answer("What is the punishment for theft?"))
    assert ans.invalid_citations == [42] and "[42]" not in ans.text


# ── 5.3 abstention ──
def test_abstain_message_says_what_was_searched_and_closest_sources():
    msg = abstain_message("recipe for cake", [_r("a", "t", title="Sample Penal Code")])
    assert 'I searched for: "recipe for cake"' in msg and "Sample Penal Code" in msg
    assert "rephras" in msg.lower()
    assert "Closest" not in abstain_message("x", [])


def test_api_abstention_event_is_helpful(settings, indexed, pipeline):
    embedder, store, _, _ = indexed
    app = create_app(
        settings,
        llm_client=ExtractiveFakeLLM(),
        embedder=embedder,
        vector_store=store,
        pipeline=pipeline,
    )
    with TestClient(app) as c:
        events = dict(
            parse_sse(c.post("/v1/ask", json={"question": "recipe for chocolate cake"}).text)
        )
    assert "I searched for" in events["abstained"]["message"]


def test_threshold_sweep_and_recommendation():
    rows = [(True, 0.6), (True, 0.5), (True, 0.35), (False, 0.0), (False, 0.1), (False, 0.3)]
    table = tune_abstain(rows, [0.0, 0.2, 0.4, 0.7])
    by_t = {r["threshold"]: r for r in table}
    assert by_t[0.0]["unanswerable_refused"] == 0 and by_t[0.0]["answerable_wrongly_refused"] == 0
    assert by_t[0.4]["unanswerable_refused"] == 1.0 and by_t[0.4][
        "answerable_wrongly_refused"
    ] == pytest.approx(1 / 3)
    assert by_t[0.7]["answerable_wrongly_refused"] == 1.0
    pick = recommend_threshold(table, max_false_refusal=0.0)
    assert pick["threshold"] == 0.2
    assert recommend_threshold(table, max_false_refusal=-1) is None
    assert "recommended" in render_abstain_table(table, pick)


def test_tune_abstain_cli_on_fixture(capsys):
    rc = eval_main(
        [
            "tune-abstain",
            "--golden",
            str(SMOKE),
            "--data-dir",
            str(FIXTURES),
            "--manifest",
            str(FIXTURES / "manifest.yaml"),
            "--embedder",
            "hashing-512",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0 and "recommended" in out
    chosen = float(next(ln for ln in out.splitlines() if "recommended" in ln).split("|")[1])
    assert 0.0 < chosen <= 0.35  # fixture: unanswerable ~0.0, answerable >= ~0.35


# ── 5.5 legal currency ──
def test_currency_notes_for_superseded_and_mapped():
    old = _r(
        "o",
        "x",
        title="Old Sample Code",
        status="superseded",
        meta={"superseded_by": "sample_penal_code"},
    )
    mapped = [{"old_act": "ipc", "old_section": "378", "new_act": "BNS", "new_section": "303"}]
    notes = currency_notes([old, old, _r("n", "y")], mapped)
    assert notes == [
        "Old Sample Code is superseded; the current law is sample_penal_code.",
        "IPC Section 378 corresponds to BNS Section 303.",
    ]
    assert currency_notes([_r("n", "y")], []) == []


def test_pipeline_attaches_notes_for_old_act_questions(settings, indexed):
    embedder, store, _, _ = indexed
    from legal_rag.generation.pipeline import RAGPipeline
    from legal_rag.retrieval.factory import build_retriever

    mapping = MappingTable(
        [
            {
                "old_act": "ipc",
                "old_section": "378",
                "new_act": "Sample Penal Code",
                "new_section": "3",
                "verified": True,
            }
        ]
    )
    pipe = RAGPipeline(
        build_retriever(settings, embedder, store), ExtractiveFakeLLM(), settings, mapping
    )
    prepared = pipe.prepare("What is theft under Section 378 of the IPC?")
    assert any("corresponds to Sample Penal Code Section 3" in n for n in prepared.notes)
    assert any("superseded" in n for n in prepared.notes) or "sample_old_code" not in {
        r.doc_id for r in prepared.results
    }


# ── 5.6 faithfulness guard ──
def test_split_claims_skips_fragments():
    text = "Theft is punished by up to three years [1]. Yes. Trespass carries a fine [2]."
    assert split_claims(text) == [
        "Theft is punished by up to three years [1].",
        "Trespass carries a fine [2].",
    ]


def test_guard_flags_unsupported_sentences_and_counts_unjudged():
    scores = iter(['{"score": 1.0}', '{"score": 0.0}', "garbage"])
    judge = Judge(ScriptedLLM(lambda m: next(scores)), "j")
    answer = (
        "Theft is punished by up to three years [1]. Trespass always means death [2]. "
        "Third claim here is long enough."
    )
    res = asyncio.run(faithfulness_guard(judge, answer, [_r("a", "text")]))
    assert res.checked == 3 and res.unjudged == 1
    assert res.unsupported == ["Trespass always means death [2]."]
    assert res.judge_calls == judge.calls == 3


def test_evaluate_records_guard_latency_and_rate(pipeline):
    qs = [q for q in load_golden(SMOKE) if q.answerable][:3]
    judge = Judge(ScriptedLLM('{"score": 0.0}'), "j")
    rows = asyncio.run(evaluate(pipeline, qs, judge=judge, guard=True))
    assert all("guard_ms" in r and r["claims_checked"] >= 1 for r in rows)
    assert summarize(rows)["overall"]["unsupported_rate"] == 1.0
    rows2 = asyncio.run(evaluate(pipeline, qs, judge=judge, guard=False))
    assert "guard_ms" not in rows2[0] and summarize(rows2)["overall"]["unsupported_rate"] == 0.0


# ── 5.1 / 5.7 configs ──
def test_prompt_versions_are_files_and_v1_is_kept():
    from legal_rag.generation.prompts import load_prompt

    assert "{context}" in load_prompt("v1") and "<passages>" in load_prompt("v2")
    assert load_prompt("v1") != load_prompt("v2")


def test_context_configs_differ_by_one_knob_and_p_final_exists():
    def diff(a, b):
        return {
            k
            for k in CONFIGS[a].keys() | CONFIGS[b].keys()
            if CONFIGS[a].get(k) != CONFIGS[b].get(k)
        }

    assert diff("C3_ctx3", "C5_ctx5") == {"retrieval_top_k"}
    assert diff("C5_ctx5", "C5_edges") == {"context_order"}
    assert diff("G1_prompt_v2", "C5_ctx5") == {"retrieval_top_k"}
    assert (
        CONFIGS["P_final"]["prompt_version"] == "v2" and CONFIGS["P_final"]["abstain_threshold"] > 0
    )


def test_guard_flag_needs_judge_and_runs_end_to_end(tmp_path, capsys):
    rc = eval_main(
        [
            "run",
            "--guard",
            "--config",
            "G1_prompt_v2",
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
    )
    assert rc == 0  # no judge configured -> guard silently inactive, run still succeeds
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["summary"]["overall"]["unsupported_rate"] == 0.0
