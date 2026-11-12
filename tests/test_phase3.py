"""Phase 3: judge metrics, cost tracking, split lock, authoring workflow, runner wiring."""

from __future__ import annotations

import asyncio
import json
import shutil

import pytest

from legal_rag.eval.author import (
    draft_all,
    promote,
    review,
    sample_chunks,
    write_drafts,
)
from legal_rag.eval.golden import (
    freeze_test_split,
    load_golden,
    split_overlap,
    verify_test_lock,
)
from legal_rag.eval.judge import Judge, parse_judge_output
from legal_rag.eval.runner import evaluate, summarize
from legal_rag.eval.runner import main as eval_main
from legal_rag.llm import LLMClient, LLMResponse
from tests.conftest import FIXTURES

SMOKE = FIXTURES.parent / "golden" / "smoke.jsonl"


class ScriptedLLM(LLMClient):
    """Returns canned replies; records the calls it received."""

    def __init__(self, replies) -> None:
        self.replies, self.calls = replies, []

    def is_ready(self) -> bool:
        return True

    def generate(self, messages, **kw) -> LLMResponse:
        self.calls.append((messages, kw))
        r = self.replies(messages) if callable(self.replies) else self.replies
        return LLMResponse(content=r, model="scripted")

    async def generate_stream(self, messages, **kw):
        yield ""


# ── judge ──
def test_parse_judge_output_variants():
    assert parse_judge_output('{"score": 0.5, "reason": "ok"}').score == 0.5
    assert parse_judge_output('Sure!\n```json\n{"score": 1, "reason": "x"}\n```').score == 1.0
    for bad in ("no json", '{"score": "high"}', '{"score": 2}', "", '{"nope": 1}'):
        assert parse_judge_output(bad).score is None


def test_judge_runs_at_temperature_zero_and_survives_errors(pipeline):
    llm = ScriptedLLM('{"score": 0.8, "reason": "r"}')
    judge = Judge(llm, "judge-model-x")
    ans = asyncio.run(pipeline.answer("What is the punishment for theft?"))
    s = asyncio.run(judge.faithfulness(ans.text, ans.results))
    assert s.score == 0.8 and llm.calls[0][1]["temperature"] == 0.0

    class Boom(ScriptedLLM):
        def generate(self, *a, **k):
            raise RuntimeError("provider down")

    bad = asyncio.run(Judge(Boom(""), "m").relevance("q", "a"))
    assert bad.score is None and "judge error" in bad.reason


def test_evaluate_with_judge_cost_and_unusable_scores_excluded(pipeline):
    qs = load_golden(SMOKE)[:3]
    pipeline.s = pipeline.s.model_copy(update={"price_in_per_1k": 1.0, "price_out_per_1k": 2.0})
    flaky = iter(['{"score": 1.0}', "garbage", '{"score": 0.5}'] * 20)
    judge = Judge(ScriptedLLM(lambda m: next(flaky)), "j")
    rows = asyncio.run(evaluate(pipeline, qs, judge=judge))
    assert all(r["cost_usd"] > 0 and r["tokens_in"] > 0 for r in rows)
    overall = summarize(rows)["overall"]
    assert overall["cost_per_q"] > 0 and overall["tokens_per_q"] > 0
    judged = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    assert overall["faithfulness"] == pytest.approx(sum(judged) / len(judged))
    assert judge.calls >= 3


def test_abstained_rows_are_not_judged_and_cost_nothing(pipeline):
    qs = [q for q in load_golden(SMOKE) if not q.answerable]
    judge = Judge(ScriptedLLM('{"score": 1}'), "j")
    rows = asyncio.run(evaluate(pipeline, qs, judge=judge))
    assert judge.calls == 0 and all(r["cost_usd"] == 0 and r["abstained"] for r in rows)


def test_judge_flag_requires_pinned_model(capsys, tmp_path):
    rc = eval_main(
        [
            "run",
            "--judge",
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
    assert rc == 2 and "JUDGE_MODEL" in capsys.readouterr().err


def test_run_output_records_provenance(tmp_path):
    rc = eval_main(
        [
            "run",
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
    assert rc == 0
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    meta = data["meta"]
    for key in (
        "git_commit",
        "fingerprint",
        "prompt_version",
        "llm",
        "embedder",
        "golden_hash",
        "judge_model",
        "judge_prompt_version",
    ):
        assert key in meta
    assert "cost_per_q" in data["summary"]["overall"]
    assert "golden:" in next(tmp_path.glob("*.md")).read_text()


# ── split policy ──
def test_test_split_must_be_frozen_and_unmodified(tmp_path):
    test = tmp_path / "test.jsonl"
    shutil.copy(SMOKE, test)
    verify_test_lock(test, "dev")  # dev needs no lock
    with pytest.raises(RuntimeError, match="not frozen"):
        verify_test_lock(test, "test")
    freeze_test_split(test)
    verify_test_lock(test, "test")
    test.write_text(test.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed since"):
        verify_test_lock(test, "test")


def test_split_overlap_detects_leakage():
    qs = load_golden(SMOKE)
    assert split_overlap(qs[:3], qs[2:5]) == [" ".join(qs[2].question.lower().split())]
    assert split_overlap(qs[:2], qs[2:]) == []


def test_golden_rejects_bad_split_and_duplicate_ids(tmp_path):
    f = tmp_path / "g.jsonl"
    row = {"id": "a", "question": "why so?", "category": "out_of_scope", "answerable": False}
    f.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(f)
    f.write_text(json.dumps({**row, "split": "holdout"}) + "\n")
    with pytest.raises(ValueError):
        load_golden(f)


# ── authoring ──
def _author_llm():
    def reply(messages):
        text = messages[0]["content"]
        if "Write 3 distinct questions" in text or "Write 2 distinct questions" in text:
            return '{"questions": ["What is the weather in Delhi?", "Who is the best cricketer?"]}'
        return '{"question": "What does this provision say?", "reference_answer": "It says X."}'

    return ScriptedLLM(reply)


def test_sampling_is_stratified_across_documents_and_deterministic(indexed):
    _, store, _, _ = indexed
    a = sample_chunks(store, "conceptual", 3, seed=1, min_chars=20)
    b = sample_chunks(store, "conceptual", 3, seed=1, min_chars=20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.doc_id for c in a}) >= 2  # not all from one document
    assert all(c.metadata["doc_type"] in {"statute", "constitution", "rule", "scheme"} for c in a)
    assert {c.doc_id for c in sample_chunks(store, "case_law", 5, min_chars=20)} == {
        "sample_judgment"
    }


def test_draft_all_covers_categories_and_writes_only_to_drafts(indexed, tmp_path):
    _, store, _, _ = indexed
    drafts = draft_all(
        _author_llm(),
        store,
        {"conceptual": 2, "case_law": 1, "multi_hop": 1, "out_of_scope": 2, "adversarial": 2},
    )
    cats = {d["category"] for d in drafts}
    assert {"conceptual", "case_law", "multi_hop", "out_of_scope"} <= cats
    hop = next(d for d in drafts if d["category"] == "multi_hop")
    assert len(set(hop["gold_doc_ids"])) == 2
    unans = [d for d in drafts if not d["answerable"]]
    assert unans and all(d["gold_doc_ids"] == [] for d in unans)
    path = write_drafts(drafts, tmp_path / "drafts")
    assert path.parent.name == "drafts" and path.name not in {"dev.jsonl", "test.jsonl"}


def test_unparseable_llm_output_yields_no_draft(indexed):
    _, store, _, _ = indexed
    assert draft_all(ScriptedLLM("not json at all"), store, {"conceptual": 2}) == []


def test_review_accept_edit_reject_then_promote_only_verified(indexed, tmp_path):
    _, store, _, _ = indexed
    drafts = draft_all(_author_llm(), store, {"conceptual": 3})
    dpath = write_drafts(drafts, tmp_path / "drafts")
    answers = iter(["a", "e", "Edited question here?", "", "r"])
    shown: list[str] = []
    acc = tmp_path / "drafts" / "accepted.jsonl"
    counts = review(dpath, acc, input_fn=lambda _: next(answers), out=shown.append)
    assert counts == {"accepted": 2, "rejected": 1}
    assert any("--- source (" in s for s in shown)  # source passage shown beside the draft
    dev = tmp_path / "dev.jsonl"
    assert promote(acc, dev, "dev") == 2
    qs = load_golden(dev)
    assert any(q.question == "Edited question here?" for q in qs)
    assert all(q.split == "dev" and "source_text" not in q.model_dump() for q in qs)
    with pytest.raises(ValueError):
        review(dpath, tmp_path / "dev.jsonl", input_fn=lambda _: "a", out=lambda _: None)


def test_promote_refuses_unverified_and_frozen_test(indexed, tmp_path):
    _, store, _, _ = indexed
    drafts = draft_all(_author_llm(), store, {"conceptual": 1})
    raw = write_drafts(drafts, tmp_path)  # drafts are NOT human verified
    with pytest.raises(ValueError, match="not verified"):
        promote(raw, tmp_path / "dev.jsonl", "dev")
    test = tmp_path / "test.jsonl"
    test.write_text("")
    freeze_test_split(test)
    with pytest.raises(RuntimeError, match="frozen"):
        promote(raw, test, "test")
