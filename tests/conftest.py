"""Shared fixtures: tiny fixture corpus indexed with the offline hashing embedder."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from legal_rag.config.settings import Settings
from legal_rag.eval.runner import ExtractiveFakeLLM
from legal_rag.generation.pipeline import RAGPipeline
from legal_rag.ingestion.indexer import Indexer
from legal_rag.ingestion.manifest import load_manifest
from legal_rag.retrieval.embedder import HashingEmbedder
from legal_rag.retrieval.hybrid import Retriever
from legal_rag.retrieval.memory_store import InMemoryVectorStore

FIXTURES = Path(__file__).resolve().parent.parent / "eval" / "fixtures"


@pytest.fixture
def settings(tmp_path) -> Settings:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Settings(
            llm_api_key="dummy-test-key",  # pragma: allowlist secret
            embedding_model="hashing-512",
            abstain_threshold=0.2,
            index_path=tmp_path / "index.json",
            feedback_path=tmp_path / "feedback.jsonl",
            admin_api_key="admin-secret-for-tests",  # pragma: allowlist secret
        )


@pytest.fixture
def indexed(settings):
    embedder = HashingEmbedder(512)
    store = InMemoryVectorStore()
    records = load_manifest(FIXTURES / "manifest.yaml")
    indexer = Indexer(settings, embedder, store, base_dir=FIXTURES)
    indexer.index_all(records)
    return embedder, store, indexer, records


@pytest.fixture
def pipeline(settings, indexed) -> RAGPipeline:
    embedder, store, _, _ = indexed
    return RAGPipeline(Retriever(embedder, store), ExtractiveFakeLLM(), settings)
