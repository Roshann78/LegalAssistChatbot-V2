"""Local rehearsal server: fixture corpus + offline fake LLM (no keys, no model downloads).

    python scripts/demo_server.py [--port 8000] [--delay-ms 0]

--delay-ms adds a per-token delay to mimic a slow LLM. Do not use this for real answers.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import warnings
from pathlib import Path

import uvicorn

from legal_rag.api.main import create_app
from legal_rag.config.settings import Settings
from legal_rag.eval.runner import ExtractiveFakeLLM
from legal_rag.ingestion.indexer import Indexer
from legal_rag.ingestion.manifest import load_manifest
from legal_rag.retrieval.embedder import HashingEmbedder
from legal_rag.retrieval.memory_store import InMemoryVectorStore

FIXTURES = Path(__file__).resolve().parent.parent / "eval" / "fixtures"


class SlowFakeLLM(ExtractiveFakeLLM):
    def __init__(self, delay_ms: int) -> None:
        self.delay = delay_ms / 1000

    async def generate_stream(self, messages, **kw):
        text = self._reply(messages)
        for word in text.split(" "):
            await asyncio.sleep(self.delay)
            yield word + " "


def build_demo_app(delay_ms: int = 0, rate_limit_rpm: int = 100000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        settings = Settings(
            llm_api_key="demo-not-a-real-key",  # pragma: allowlist secret
            embedding_model="hashing-512",
            abstain_threshold=0.2,
            rate_limit_rpm=rate_limit_rpm,
            frontend_dir=Path(__file__).resolve().parent.parent / "frontend",
        )
    embedder, store = HashingEmbedder(512), InMemoryVectorStore()
    with tempfile.TemporaryDirectory() as tmp:
        Indexer(settings, embedder, store, FIXTURES, processed_dir=Path(tmp)).index_all(
            load_manifest(FIXTURES / "manifest.yaml")
        )
    return create_app(
        settings, llm_client=SlowFakeLLM(delay_ms), embedder=embedder, vector_store=store
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--delay-ms", type=int, default=0)
    args = ap.parse_args()
    uvicorn.run(
        build_demo_app(args.delay_ms), host="127.0.0.1", port=args.port, log_level="warning"
    )
