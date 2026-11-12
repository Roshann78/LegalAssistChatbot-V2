"""Chunking, manifest validation, parsing/cleaning and idempotent indexing."""

from __future__ import annotations

import shutil

import pytest

from legal_rag.ingestion.chunking import chunk_document, fixed_spans, strategy_for
from legal_rag.ingestion.cli import run as cli_run
from legal_rag.ingestion.manifest import (
    draft_entries,
    load_manifest,
    sha256_file,
    validate_manifest,
)
from legal_rag.ingestion.parser import Page, clean_pages
from legal_rag.retrieval.memory_store import FingerprintMismatchError
from tests.conftest import FIXTURES

STATUTE = (
    "1. Short title. This Act is the Sample Act.\n\n"
    "2. Theft. Whoever takes property commits theft. Provided that a bona fide claim of right "
    "is not theft. Explanation: the proviso stays with its section.\n\n"
    "3. Punishment. Imprisonment up to three years.\n"
)
JUDGMENT = (
    "1. First paragraph about facts.\n\n2. Second paragraph about the issue.\n\n"
    "3. Third paragraph holding.\n"
)


def test_s1_keeps_section_with_proviso_and_records_metadata():
    chunks = chunk_document({"doc_id": "d", "title": "Sample Act"}, [Page(1, STATUTE)], "s1")
    theft = [c for c in chunks if c.metadata["section_or_article_no"] == "Section 2"]
    assert len(theft) == 1 and "Provided that" in theft[0].text
    md = theft[0].metadata
    assert md["heading_path"] == ["Sample Act", "Section 2"]
    assert md["page_start"] == 1 and md["chunk_id"].startswith("d:s1-1:")
    assert theft[0].embed_text.startswith("Sample Act > Section 2")


def test_s1_long_section_is_split_with_overlap_and_same_label():
    long_text = "5. Big section. " + ("Sentence about duties. " * 200)
    chunks = chunk_document({"doc_id": "d"}, [Page(1, long_text)], "s1", size=500, overlap=50)
    assert len(chunks) > 3
    assert {c.metadata["section_or_article_no"] for c in chunks} == {"Section 5"}
    assert all(len(c.text) <= 500 for c in chunks)


def test_s2_groups_numbered_paragraphs_and_never_cuts_mid_paragraph():
    chunks = chunk_document({"doc_id": "j"}, [Page(1, JUDGMENT)], "s2", size=70, overlap=0)
    assert [c.metadata["paragraph_no"] for c in chunks][0] in {"1", "1-2"}
    assert all(c.text[0].isdigit() for c in chunks)


def test_s3_heading_based_and_page_tracking():
    pages = [Page(1, "PART ONE\nalpha text here"), Page(2, "PART TWO\nbeta text here")]
    chunks = chunk_document({"doc_id": "c", "title": "Circ"}, pages, "s3")
    assert [c.metadata["page_start"] for c in chunks] == [1, 2]
    assert chunks[1].metadata["heading_path"][-1] == "PART TWO"


def test_s0_baseline_and_unknown_strategy():
    chunks = chunk_document({"doc_id": "d"}, [Page(1, "word " * 500)], "s0", size=200, overlap=20)
    assert len(chunks) >= 10
    with pytest.raises(ValueError):
        chunk_document({"doc_id": "d"}, [Page(1, "x")], "nope")
    assert strategy_for("judgment") == "s2" and strategy_for("statute", "s0") == "s0"


def test_fixed_spans_always_progress():
    spans = fixed_spans("a" * 1000, 0, 1000, 100, 99)
    assert spans[-1][1] == 1000 and len(spans) < 1000


def test_clean_pages_removes_headers_footers_page_numbers_and_hyphens():
    pages = [
        Page(i, f"GAZETTE OF INDIA\nbody line {i} exam-\nple text\n{i}\nFooter note")
        for i in range(1, 6)
    ]
    cleaned = clean_pages(pages)
    assert "GAZETTE" not in cleaned[0].text and "Footer note" not in cleaned[0].text
    assert "example text" in cleaned[0].text
    assert cleaned[2].page == 3 and "\n3" not in cleaned[2].text


def test_manifest_validates_fixture_and_catches_problems(tmp_path):
    records = load_manifest(FIXTURES / "manifest.yaml")
    assert validate_manifest(records, FIXTURES) == []
    bad = [
        r.model_copy(update={"superseded_by": "ghost", "path": "corpus/missing.txt"})
        for r in records[:1]
    ]
    bad.append(records[0])
    errs = " | ".join(validate_manifest(bad, FIXTURES))
    assert "duplicate doc_id" in errs and "file missing" in errs and "'ghost'" in errs


def test_manifest_rejects_bad_enum(tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text("documents:\n  - {doc_id: a, title: A, doc_type: bogus, domain: legal, path: x}\n")
    with pytest.raises(ValueError, match="invalid manifest"):
        load_manifest(f)


def test_hash_mismatch_detected(tmp_path):
    shutil.copytree(FIXTURES, tmp_path / "fx")
    base = tmp_path / "fx"
    records = load_manifest(base / "manifest.yaml")
    rec = records[0].model_copy(update={"file_hash": sha256_file(base / records[0].path)})
    assert validate_manifest([rec], base) == []
    (base / rec.path).write_text("changed")
    assert any("hash mismatch" in e for e in validate_manifest([rec], base))


def test_draft_entries_scan(tmp_path):
    f = tmp_path / "legal" / "statute" / "my_act.pdf"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"%PDF-1.4")
    d = draft_entries(tmp_path)
    assert d[0]["doc_id"] == "my_act" and d[0]["domain"] == "legal" and len(d[0]["file_hash"]) == 64


def test_indexing_is_idempotent_and_metadata_complete(indexed):
    _, store, indexer, records = indexed
    first = store.count()
    assert first > 0
    summary = indexer.index_all(records)
    assert summary.indexed == [] and len(summary.skipped) == len(records)
    assert store.count() == first
    for chunks in (store.get_chunks(r.doc_id) for r in records):
        for c in chunks:
            for key in (
                "doc_id",
                "chunk_id",
                "title",
                "domain",
                "status",
                "page_start",
                "heading_path",
                "config_fingerprint",
                "file_hash",
            ):
                assert key in c.metadata and c.metadata[key] not in (None, "")


def test_update_replaces_and_remove_deletes(indexed, tmp_path):
    _, store, indexer, records = indexed
    rec = next(r for r in records if r.doc_id == "sample_judgment")
    before = {c.chunk_id for c in store.get_chunks(rec.doc_id)}
    assert indexer.index_document(rec, force=True) == "indexed"
    assert {c.chunk_id for c in store.get_chunks(rec.doc_id)} == before
    assert indexer.remove(rec.doc_id) == len(before)
    assert rec.doc_id not in store.list_docs()
    assert indexer.verify(records)["missing_from_index"] == [rec.doc_id]


def test_fingerprint_mismatch_is_rejected(indexed, settings):
    _, store, _, records = indexed
    store.fingerprint = "different"
    from legal_rag.ingestion.indexer import Indexer
    from legal_rag.retrieval.embedder import HashingEmbedder

    with pytest.raises(FingerprintMismatchError):
        Indexer(settings, HashingEmbedder(512), store, base_dir=FIXTURES).index_document(records[0])


def test_cli_validate_reindex_verify_list(settings, capsys):
    s = settings.model_copy(
        update={"manifest_path": FIXTURES / "manifest.yaml", "data_dir": FIXTURES}
    )
    assert cli_run(["validate"], s) == 0
    assert cli_run(["verify"], s) == 1  # nothing indexed yet
    assert cli_run(["reindex", "--all"], s) == 0
    assert cli_run(["verify"], s) == 0
    assert cli_run(["add", "sample_judgment"], s) == 0
    out = capsys.readouterr().out
    assert "sample_judgment: skipped" in out
    assert cli_run(["remove", "sample_judgment"], s) == 0
    assert cli_run(["verify"], s) == 1
    assert cli_run(["add", "nope"], s) == 2
    assert cli_run(["--dry-run", "list"], s) == 0
    assert cli_run(["stats"], s) == 0
