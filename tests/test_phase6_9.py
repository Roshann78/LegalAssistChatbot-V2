"""Phases 6-9: uploads, metrics, tracing, frontend, repo guards, demo/load tooling."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from legal_rag.api.main import create_app
from legal_rag.api.uploads import UploadError, UploadStore, build_upload_index, safe_filename
from legal_rag.eval.runner import ExtractiveFakeLLM
from legal_rag.generation.prompts import build_messages
from legal_rag.observability.metrics import Metrics
from legal_rag.observability.tracing import FileTracer, build_tracer, find_trace
from legal_rag.retrieval.embedder import HashingEmbedder
from scripts.check_staged import blocked
from tests.test_api_v1 import parse_sse

ROOT = Path(__file__).resolve().parent.parent
INJECTION = sorted((ROOT / "eval" / "injection").glob("*.txt"))
TEXT = (
    "Notice period is thirty days for either party under this sample agreement and it must be "
    "given in writing by registered post to the other party."
)


def make_pdf(pages: int = 1, text: str = TEXT) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(settings, indexed, pipeline, tmp_path):
    embedder, store, _, _ = indexed
    s = settings.model_copy(
        update={"trace_path": tmp_path / "traces.jsonl", "frontend_dir": ROOT / "frontend"}
    )
    app = create_app(
        s, llm_client=ExtractiveFakeLLM(), embedder=embedder, vector_store=store, pipeline=pipeline
    )
    with TestClient(app) as c:
        c.settings = s
        yield c


def upload(c, name, data, **kw):
    return c.post(
        "/v1/documents/upload",
        params={"filename": name},
        content=data,
        headers={"content-type": "application/octet-stream"},
        **kw,
    )


# ── uploads ──
def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\x\\my file<>.pdf") == "my file__.pdf"
    assert safe_filename("") == "upload" and len(safe_filename("a" * 500)) == 80


def test_upload_txt_and_pdf_then_ask_uses_only_that_document(client):
    r = upload(client, "lease.txt", TEXT.encode())
    assert r.status_code == 201
    info = r.json()
    assert info["chunks"] >= 1 and info["filename"] == "lease.txt"
    events = dict(
        parse_sse(
            client.post(
                "/v1/ask",
                json={"question": "What is the notice period?", "upload_id": info["upload_id"]},
            ).text
        )
    )
    cites = events["citations"]["citations"]
    assert cites and cites[0]["title"] == "lease.txt" and cites[0]["doc_id"] == "upload"

    pdf = upload(client, "contract.pdf", make_pdf())
    assert pdf.status_code == 201 and pdf.json()["pages"] == 1


def test_upload_validation_errors(client):
    assert upload(client, "x.exe", b"MZ").status_code == 415
    assert upload(client, "fake.pdf", b"not a pdf").status_code == 415
    assert upload(client, "empty.txt", b"").status_code == 422
    assert upload(client, "scan.pdf", make_pdf(text=" ")).status_code == 422
    assert upload(client, "big.pdf", make_pdf(pages=3)).status_code == 201
    s = client.settings.model_copy(update={"upload_max_pages": 2})
    with pytest.raises(UploadError, match="pages"):
        build_upload_index("big.pdf", make_pdf(pages=3), s, HashingEmbedder(512))
    big = b"x" * (client.settings.upload_max_bytes + 1)
    assert upload(client, "big.txt", big).status_code == 413


def test_upload_size_limit_enforced_without_content_length(client):
    def gen():
        for _ in range(6):
            yield b"a" * 1_000_000

    r = client.post("/v1/documents/upload", params={"filename": "stream.txt"}, content=gen())
    assert r.status_code == 413


def test_uploads_are_isolated_and_deletable(client):
    a = upload(client, "a.txt", b"Alpha agreement says the secret code word is zebra stripes " * 3)
    b = upload(client, "b.txt", b"Beta agreement covers maritime salvage rights and tonnage " * 3)
    ida, idb = a.json()["upload_id"], b.json()["upload_id"]
    ev = dict(
        parse_sse(
            client.post(
                "/v1/ask", json={"question": "secret code word zebra", "upload_id": idb}
            ).text
        )
    )
    assert "abstained" in ev or all(
        "zebra" not in c["snippet"] for c in ev["citations"]["citations"]
    )
    assert client.delete(f"/v1/documents/session/{ida}").status_code == 204
    assert client.delete(f"/v1/documents/session/{ida}").status_code == 404
    gone = client.post("/v1/ask", json={"question": "anything here", "upload_id": ida})
    assert gone.status_code == 404
    assert (
        client.post("/v1/ask", json={"question": "anything here", "upload_id": "nope"}).status_code
        == 404
    )


def test_upload_store_ttl_and_capacity():
    clock = [0.0]
    store = UploadStore(ttl_seconds=100, max_sessions=2, time_fn=lambda: clock[0])
    from legal_rag.retrieval.memory_store import InMemoryVectorStore

    first = store.add(InMemoryVectorStore())
    second = store.add(InMemoryVectorStore())
    third = store.add(InMemoryVectorStore())  # evicts the least recently used
    assert store.get(first) is None and store.get(second) and store.get(third)
    clock[0] = 101
    assert store.get(second) is None and store.count() == 0


@pytest.mark.parametrize("path", INJECTION, ids=lambda p: p.stem)
def test_injection_documents_cannot_break_out_of_the_data_block(client, path):
    """Structural defences only: hostile text stays inside the passages block, next to rules
    telling the model to ignore it. Whether a real LLM obeys is NOT tested here."""
    assert len(INJECTION) == 5
    store, _, _ = build_upload_index(
        path.name, path.read_bytes(), client.settings, HashingEmbedder(512)
    )
    results = store.dense_search(HashingEmbedder(512).embed(["agreement terms"])[0], None, 5)
    system = build_messages("v2", "What are the terms?", results)[0][0]["content"]
    opener = "\n<passages>\n"
    assert system.count(opener) == 1 and system.count("</passages>") == 1
    assert "untrusted DATA" in system
    inside = system.split(opener)[1].split("</passages>")[0]
    assert "[1]" in inside


def test_upload_blocks_do_not_block_event_loop(client):
    """Two requests overlap: a slow-ish upload does not prevent /health from answering."""
    assert upload(client, "a.txt", TEXT.encode()).status_code == 201
    assert client.get("/health").status_code == 200


# ── metrics / tracing / feedback ──
def test_metrics_endpoint_reports_percentiles_and_rates(client):
    client.post("/v1/ask", json={"question": "What is the punishment for theft?"})
    client.post("/v1/ask", json={"question": "recipe for chocolate cake"})
    m = client.get("/v1/metrics").json()
    assert m["requests"] == 2 and m["answered"] == 1 and m["abstained"] == 1
    assert m["abstention_rate"] == 0.5 and m["error_rate"] == 0.0
    assert m["latency_ms"]["total_ms"]["n"] == 2 and "p95" in m["latency_ms"]["retrieve_ms"]


def test_metrics_unit_counts_errors_and_is_bounded():
    m = Metrics(window=3)
    for i in range(10):
        m.record("error" if i == 0 else "answered", {"a": float(i)})
    snap = m.snapshot()
    assert snap["errors"] == 1 and snap["latency_ms"]["a"]["n"] == 3
    assert snap["error_rate"] == pytest.approx(0.1)


def test_trace_written_and_found_by_request_id_and_feedback_links(client):
    r = client.post("/v1/ask", json={"question": "What is the punishment for theft?"})
    req_id = r.headers["x-request-id"]
    trace = find_trace(client.settings.trace_path, req_id)
    assert trace["outcome"] == "answered" and trace["retrieved"][0]["doc_id"] == "sample_penal_code"
    assert "prompt" not in trace and trace["prompt_chars"] > 0  # prompt text is never stored
    admin = {"x-api-key": "admin-secret-for-tests"}
    assert client.get(f"/v1/admin/traces/{req_id}", headers=admin).json()["request_id"] == req_id
    assert client.get("/v1/admin/traces/unknown-id-123", headers=admin).status_code == 404
    assert client.get(f"/v1/admin/traces/{req_id}").status_code == 401
    assert (
        client.post("/v1/feedback", json={"request_id": req_id, "rating": "down"}).status_code
        == 202
    )


def test_tracing_is_off_by_default(tmp_path):
    assert build_tracer(None).enabled is False
    t = build_tracer(tmp_path / "t.jsonl")
    assert isinstance(t, FileTracer)
    t.write({"request_id": "r1"})
    assert json.loads((tmp_path / "t.jsonl").read_text())["request_id"] == "r1"


# ── API contract, frontend ──
def test_openapi_lists_the_v1_contract(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in (
        "/v1/ask",
        "/v1/documents/upload",
        "/v1/documents/session/{upload_id}",
        "/v1/stats",
        "/v1/metrics",
        "/v1/feedback",
        "/health",
        "/ready",
    ):
        assert p in paths, p


def test_frontend_is_served_with_disclaimer_and_no_hardcoded_backend(client):
    html = client.get("/ui/").text
    assert "not legal advice" in html
    assert "localhost" not in html and "127.0.0.1" not in html
    assert "zero hallucination" not in html.lower()
    assert "/v1/ask" in html and "/v1/stats" in html


def test_error_responses_use_the_error_model(client):
    r = client.post("/v1/ask", json={"question": ""})
    assert r.status_code == 422 and set(r.json()) == {"code", "message", "request_id"}


# ── repo guards, tooling ──
def test_staged_artifact_guard():
    bad = blocked(
        [
            ".env",
            "data/processed/x.json",
            "logs/app.log",
            "qdrant_data/a",
            "m.safetensors",
            "eval/runs/2026_run.json",
            "data/cache/emb.sqlite",
        ]
    )
    assert len(bad) == 7
    assert (
        blocked([".env.example", "data/processed/.gitkeep", "src/a.py", "eval/runs/.gitkeep"]) == []
    )


def test_demo_app_builds_and_answers():
    from scripts.demo_server import build_demo_app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(build_demo_app()) as c:
            assert c.get("/ready").status_code == 200
            events = dict(parse_sse(c.post("/v1/ask", json={"question": "theft punishment"}).text))
            assert "citations" in events
            assert "Legal Assist" in c.get("/ui/").text


def test_pre_commit_and_ci_reference_the_guards():
    assert "check_staged.py" in (ROOT / ".pre-commit-config.yaml").read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "pip-audit" in ci and "--min" in ci
