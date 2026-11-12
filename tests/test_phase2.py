"""Phase 2: Qdrant, normalized JSON, ToC/token limits, cache, fingerprint, safe order, CLI."""

from __future__ import annotations

import shutil

import pytest
from qdrant_client import QdrantClient

from legal_rag.api.main import create_app
from legal_rag.eval.runner import ExtractiveFakeLLM
from legal_rag.ingestion.cache import EmbeddingCache
from legal_rag.ingestion.chunking import chunk_document, is_toc_page
from legal_rag.ingestion.cli import run as cli_run
from legal_rag.ingestion.indexer import Indexer, config_fingerprint
from legal_rag.ingestion.manifest import load_manifest
from legal_rag.ingestion.normalize import ensure_normalized
from legal_rag.ingestion.parser import Page, clean_pages
from legal_rag.retrieval.embedder import HashingEmbedder
from legal_rag.retrieval.hybrid import Retriever
from legal_rag.retrieval.memory_store import FingerprintMismatchError
from legal_rag.retrieval.qdrant_store import QdrantVectorStore, point_id
from legal_rag.retrieval.types import Chunk, Filters
from tests.conftest import FIXTURES


@pytest.fixture
def qstore():
    return QdrantVectorStore(QdrantClient(":memory:"), "t", 512)


@pytest.fixture
def qindexer(settings, qstore):
    s = settings.model_copy(update={"vector_store_backend": "qdrant"})
    return Indexer(s, HashingEmbedder(512), qstore, base_dir=FIXTURES), qstore


def _chunk(cid, doc, text, **meta) -> Chunk:
    return Chunk(cid, doc, text, "", {"doc_id": doc, "title": doc, "status": "in_force", **meta})


# ── Qdrant adapter ──
def test_qdrant_upsert_filtered_search_and_structured_hits(qstore):
    emb = HashingEmbedder(512)
    cs = [
        _chunk(
            "a:1", "a", "theft punishment imprisonment", domain="legal", section_or_article_no="S4"
        ),
        _chunk("b:1", "b", "kyc customer identification", domain="banking"),
    ]
    qstore.upsert(cs, emb.embed([c.text for c in cs]))
    hits = qstore.dense_search(emb.embed(["theft punishment"])[0], Filters(domain="legal"), 5)
    assert [h.doc_id for h in hits] == ["a"] and hits[0].section == "S4" and hits[0].dense_score > 0
    assert qstore.sparse_search("kyc", None, 5)[0].doc_id == "b"
    assert qstore.sparse_search("zzzz", None, 5) == []
    assert {
        h.doc_id for h in qstore.dense_search(emb.embed(["x"])[0], Filters(status=None), 5)
    } == {
        "a",
        "b",
    }


def test_qdrant_reupsert_is_deterministic_and_does_not_duplicate(qstore):
    emb = HashingEmbedder(512)
    c = _chunk("a:1", "a", "same text")
    for _ in range(3):
        qstore.upsert([c], emb.embed([c.text]))
    assert qstore.count() == 1 and point_id("a:1") == point_id("a:1")
    assert qstore.list_docs() == ["a"]


def test_qdrant_delete_by_doc_and_delete_except(qstore):
    emb = HashingEmbedder(512)
    cs = [_chunk(f"a:{i}", "a", f"text {i}") for i in range(3)] + [_chunk("b:0", "b", "other")]
    qstore.upsert(cs, emb.embed([c.text for c in cs]))
    assert qstore.delete_except("a", {"a:0"}) == 2 and qstore.count() == 2
    assert qstore.delete_by_doc_id("a") == 1 and qstore.list_docs() == ["b"]
    assert qstore.delete_by_doc_id("ghost") == 0


def test_qdrant_fingerprint_is_enforced_and_persisted_in_collection(qstore):
    emb = HashingEmbedder(512)
    qstore.check_fingerprint("fp1")
    assert qstore.fingerprint == "fp1"
    qstore.upsert([_chunk("a:1", "a", "t")], emb.embed(["t"]))
    with pytest.raises(FingerprintMismatchError, match="reindex"):
        qstore.check_fingerprint("fp2")
    qstore.clear()
    qstore.check_fingerprint("fp2")  # empty collection may adopt a new fingerprint


def test_qdrant_ready_and_hybrid_retriever(qindexer):
    indexer, store = qindexer
    assert store.is_ready() and "ready" in store.get_status()
    indexer.index_all(load_manifest(FIXTURES / "manifest.yaml"))
    res = Retriever(indexer.embedder, store).retrieve("Section 4 punishment for theft", k=3)
    assert res[0].doc_id == "sample_penal_code" and res[0].section == "Section 4"


def test_app_ready_reflects_qdrant_store(settings, qstore):
    app = create_app(
        settings, llm_client=ExtractiveFakeLLM(), embedder=HashingEmbedder(512), vector_store=qstore
    )
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        body = c.get("/ready").json()
        assert c.get("/ready").status_code == 200 and "qdrant" in body["components"]["vector_store"]


# ── normalized JSON ──
def test_normalized_json_written_cached_and_reparsed_on_change(tmp_path):
    src = tmp_path / "d.txt"
    body = "INTRODUCTION\n\n" + "first paragraph text. " * 5
    src.write_text(body + "\f\n\n\n", encoding="utf-8")
    out = tmp_path / "processed"
    doc, status = ensure_normalized("d", src, out)
    assert status == "parsed" and (out / "d.json").exists()
    assert doc.pages[0].blocks[0].type == "heading" and doc.stats.low_text_pages == [2]
    assert doc.source_file_hash and doc.parser_name == "text"
    assert ensure_normalized("d", src, out)[1] == "cached"
    src.write_text("changed content that is long enough to avoid the low text flag at all", "utf-8")
    assert ensure_normalized("d", src, out)[1] == "parsed"
    (out / "d.json").write_text("{not json")
    assert ensure_normalized("d", src, out)[1] == "parsed"  # corrupt cache recovers


def test_legal_looking_repeated_line_is_not_removed_by_cleaner():
    pages = [Page(i, f"Section 5. Repeated clause\nGAZETTE HEADER\nbody {i}") for i in range(1, 7)]
    text = clean_pages(pages)[0].text
    assert "Section 5. Repeated clause" in text and "GAZETTE HEADER" not in text


# ── chunking limits ──
def test_toc_page_excluded():
    toc = "ARRANGEMENT OF SECTIONS\n" + "\n".join(
        f"{i}. Heading number {i} ........ {i}" for i in range(1, 9)
    )
    body = "1. Short title. This Act is the Sample Act and applies everywhere in the country."
    assert is_toc_page(toc) and not is_toc_page(body)
    chunks = chunk_document({"doc_id": "d"}, [Page(1, toc), Page(2, body)], "s1")
    assert len(chunks) == 1 and chunks[0].metadata["page_start"] == 2


def test_max_tokens_enforced_with_injected_counter_and_ids_are_stable():
    text = "5. Big. " + "alpha beta gamma delta. " * 100
    count = lambda t: len(t.split())  # noqa: E731
    kw = dict(size=5000, overlap=10, count_tokens=count, max_tokens=30)
    a = chunk_document({"doc_id": "d"}, [Page(1, text)], "s1", **kw)
    b = chunk_document({"doc_id": "d"}, [Page(1, text)], "s1", **kw)
    assert len(a) > 3 and all(c.metadata["token_count"] <= 30 for c in a)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.text.strip() for c in a)
    for key in ("strategy_id", "strategy_version", "page_start", "heading_path", "chunk_type"):
        assert all(key in c.metadata for c in a)


# ── cache, fingerprint, safe order ──
def test_embedding_cache_persists_across_indexers(settings):
    path = settings.embedding_cache_path
    records = load_manifest(FIXTURES / "manifest.yaml")
    from legal_rag.retrieval.memory_store import InMemoryVectorStore

    emb = HashingEmbedder(512)
    first = Indexer(
        settings, emb, InMemoryVectorStore(), FIXTURES, cache=EmbeddingCache(path, emb.name)
    ).index_all(records)
    assert first.cache_hits == 0
    second = Indexer(
        settings, emb, InMemoryVectorStore(), FIXTURES, cache=EmbeddingCache(path, emb.name)
    ).index_all(records)
    assert second.cache_hits == second.chunks > 0


def test_fingerprint_changes_when_any_ingredient_changes(settings):
    emb = HashingEmbedder(512)
    base = config_fingerprint(settings, emb)
    assert base == config_fingerprint(settings, emb)
    for update in (
        {"chunk_size": 700},
        {"chunk_overlap": 100},
        {"chunk_strategy": "s0"},
        {"contextual_prefix": False},
        {"chunk_max_tokens": 128},
        {"vector_store_backend": "qdrant"},
    ):
        assert config_fingerprint(settings.model_copy(update=update), emb) != base, update
    assert config_fingerprint(settings, HashingEmbedder(256)) != base


def test_safe_order_failure_after_upsert_never_loses_the_document(indexed, monkeypatch):
    _, store, indexer, records = indexed
    rec = next(r for r in records if r.doc_id == "sample_penal_code")
    before = {c.chunk_id for c in store.get_chunks(rec.doc_id)}
    monkeypatch.setattr(store, "delete_except", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        indexer.index_document(rec, force=True)
    after = {c.chunk_id for c in store.get_chunks(rec.doc_id)}
    assert before <= after  # old data still present; new data already written


def test_changed_file_replaces_chunks_and_removes_stale_ones(settings, tmp_path):
    shutil.copytree(FIXTURES, tmp_path / "fx")
    from legal_rag.retrieval.memory_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    records = load_manifest(tmp_path / "fx" / "manifest.yaml")
    idx = Indexer(settings, HashingEmbedder(512), store, tmp_path / "fx")
    idx.index_all(records)
    n_before = len(store.get_chunks("sample_penal_code"))
    f = tmp_path / "fx" / "corpus" / "sample_penal_code.txt"
    f.write_text("1. Short title. Only one section now exists in this amended sample act.", "utf-8")
    assert (
        idx.index_document(next(r for r in records if r.doc_id == "sample_penal_code")) == "indexed"
    )
    after = store.get_chunks("sample_penal_code")
    assert len(after) == 1 < n_before and "amended" in after[0].text


# ── CLI ──
def _cli_settings(settings, **extra):
    return settings.model_copy(
        update={"manifest_path": FIXTURES / "manifest.yaml", "data_dir": FIXTURES, **extra}
    )


@pytest.mark.parametrize("backend", ["memory", "qdrant"])
def test_cli_full_cycle_on_both_backends(settings, capsys, backend):
    s = _cli_settings(settings, vector_store_backend=backend)
    assert cli_run(["parse", "--all"], s) == 0
    assert (s.processed_dir / "sample_penal_code.json").exists()
    assert cli_run(["verify"], s) == 1
    assert cli_run(["--dry-run", "reindex", "--all"], s) == 0
    assert cli_run(["verify"], s) == 1  # dry-run changed nothing
    assert cli_run(["--yes", "reindex", "--all"], s) == 0
    assert cli_run(["verify"], s) == 0
    assert cli_run(["add", "sample_judgment"], s) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert cli_run(["status"], s) == 0
    assert cli_run(["remove", "sample_judgment"], s) == 1  # refuses without --yes
    assert cli_run(["verify"], s) == 0
    assert cli_run(["--yes", "remove", "sample_judgment"], s) == 0
    assert cli_run(["verify"], s) == 1
    assert (FIXTURES / "corpus" / "sample_judgment.txt").exists()  # source never deleted
    assert cli_run(["update", "sample_judgment"], s) == 0
    assert cli_run(["verify"], s) == 0


def test_cli_verify_detects_orphan_and_hash_mismatch(settings, tmp_path, capsys):
    shutil.copytree(FIXTURES, tmp_path / "fx")
    s = settings.model_copy(
        update={"manifest_path": tmp_path / "fx" / "manifest.yaml", "data_dir": tmp_path / "fx"}
    )
    assert cli_run(["--yes", "reindex", "--all"], s) == 0
    (tmp_path / "fx" / "corpus" / "sample_old_code.txt").write_text("changed on disk", "utf-8")
    manifest = (tmp_path / "fx" / "manifest.yaml").read_text().splitlines()
    (tmp_path / "fx" / "manifest.yaml").write_text(
        "\n".join(ln for ln in manifest if "sample_judgment" not in ln) + "\n"
    )
    assert cli_run(["verify"], s) == 1
    out = capsys.readouterr().out
    assert "orphaned_in_index" in out and "sample_judgment" in out and "sample_old_code" in out


def test_cli_fingerprint_mismatch_is_rejected_until_reindex(settings):
    s = _cli_settings(settings)
    assert cli_run(["--yes", "reindex", "--all"], s) == 0
    changed = s.model_copy(update={"chunk_size": 500})
    assert cli_run(["update", "sample_penal_code"], changed) == 3
    assert cli_run(["--yes", "reindex", "--all"], changed) == 0
    assert cli_run(["verify"], changed) == 0


def test_cli_missing_source_file_is_a_clear_error(settings, tmp_path, capsys):
    shutil.copytree(FIXTURES, tmp_path / "fx")
    s = settings.model_copy(
        update={"manifest_path": tmp_path / "fx" / "manifest.yaml", "data_dir": tmp_path / "fx"}
    )
    (tmp_path / "fx" / "corpus" / "sample_judgment.txt").unlink()
    assert cli_run(["add", "sample_judgment"], s) == 2
    assert "source file missing" in capsys.readouterr().err


def test_report_and_metadata_lookup(settings, tmp_path, capsys):
    s = _cli_settings(settings)
    out = tmp_path / "report.md"
    assert cli_run(["report", "--out", str(out)], s) == 0
    text = out.read_text(encoding="utf-8")
    assert "Chunks per document type" in text and "sample_penal_code" in text
    assert cli_run(["lookup", "sample_penal_code", "Section 4"], s) == 0
    assert "Punishment for theft" in capsys.readouterr().out
    assert cli_run(["lookup", "sample_penal_code", "99"], s) == 1
