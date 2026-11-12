"""Stateless RAG pipeline with separate callable stages (understand, retrieve, gate, generate)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from legal_rag.config.settings import Settings
from legal_rag.generation.citations import Citation, resolve_citations
from legal_rag.generation.prompts import build_messages
from legal_rag.llm import LLMClient
from legal_rag.retrieval.hybrid import Retriever
from legal_rag.retrieval.query import MappingTable, UnderstoodQuery, understand
from legal_rag.retrieval.types import Filters, SearchResult

_OLD_ACTS = {"ipc", "crpc", "iea"}
ABSTAIN_MESSAGE = (
    "I could not find enough support in the indexed documents to answer this reliably. "
    "Try rephrasing, or name the specific Act, section, article or circular you mean."
)


@dataclass
class Prepared:
    understood: UnderstoodQuery
    results: list[SearchResult]
    messages: list[dict[str, str]] = field(default_factory=list)
    abstain: bool = False
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    invalid_citations: list[int]
    abstained: bool
    results: list[SearchResult]
    timings: dict[str, float]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        settings: Settings,
        mapping: MappingTable | None = None,
    ) -> None:
        self.retriever, self.llm, self.s = retriever, llm, settings
        self.mapping = mapping or MappingTable()

    def prepare(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        domain: str | None = None,
    ) -> Prepared:
        """Everything before generation. CPU-bound: call via a thread from async code."""
        t0 = time.perf_counter()
        uq = understand(question, (history or [])[-self.s.history_turns :], self.mapping)
        t1 = time.perf_counter()
        wants_old = any(w in uq.standalone.lower().split() for w in _OLD_ACTS)
        filters = Filters(domain=domain, status=None if wants_old else ("in_force", "amended"))
        results = self.retriever.retrieve(
            uq.standalone, k=self.s.retrieval_top_k, filters=filters, extra_terms=uq.expansion
        )
        t2 = time.perf_counter()
        timings = {"understand_ms": (t1 - t0) * 1e3, "retrieve_ms": (t2 - t1) * 1e3}
        best = max((r.dense_score for r in results), default=0.0)
        if not results or best < self.s.abstain_threshold:
            return Prepared(uq, results, abstain=True, timings=timings)
        messages, kept = build_messages(self.s.prompt_version, question, results)
        return Prepared(uq, kept, messages, timings=timings)

    async def stream(self, prepared: Prepared) -> AsyncIterator[str]:
        async for tok in self.llm.generate_stream(prepared.messages):
            yield tok

    async def answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        domain: str | None = None,
    ) -> Answer:
        """Non-streaming path used by the eval runner."""
        prepared = await asyncio.to_thread(self.prepare, question, history, domain)
        if prepared.abstain:
            return Answer(ABSTAIN_MESSAGE, [], [], True, prepared.results, prepared.timings)
        t = time.perf_counter()
        text = "".join([tok async for tok in self.stream(prepared)])
        timings = {**prepared.timings, "generate_ms": (time.perf_counter() - t) * 1e3}
        cites, invalid = resolve_citations(text, prepared.results)
        return Answer(text, cites, invalid, False, prepared.results, timings)
