"""Stateless RAG pipeline with separate callable stages (understand, retrieve, gate, generate)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace

from legal_rag.config.settings import Settings
from legal_rag.generation.citations import Citation, resolve_citations, strip_invalid_citations
from legal_rag.generation.prompts import build_messages
from legal_rag.llm import LLMClient
from legal_rag.retrieval.hybrid import Retriever
from legal_rag.retrieval.query import (
    MappingTable,
    UnderstoodQuery,
    route_domain,
    section_labels,
    understand,
)
from legal_rag.retrieval.types import Filters, SearchResult

_OLD_ACTS = {"ipc", "crpc", "iea"}
ABSTAIN_MESSAGE = (
    "I could not find enough support in the indexed documents to answer this reliably. "
    "Try rephrasing, or name the specific Act, section, article or circular you mean."
)


def abstain_message(searched: str, results: list[SearchResult]) -> str:
    """Helpful abstention: what was searched, the closest sources found, how to rephrase."""
    titles = list(dict.fromkeys(r.title for r in results))[:3]
    closest = f" Closest documents found: {'; '.join(titles)}." if titles else ""
    return f'{ABSTAIN_MESSAGE} I searched for: "{searched}".{closest}'


def currency_notes(results: list[SearchResult], mapped: list[dict]) -> list[str]:
    """Legal-currency notes: superseded sources and verified old->new section mappings."""
    notes = []
    for r in results:
        if r.status in ("superseded", "repealed"):
            new = r.metadata.get("superseded_by")
            tail = f"; the current law is {new}" if new else ""
            notes.append(f"{r.title} is {r.status}{tail}.")
    for m in mapped:
        notes.append(
            f"{m['old_act'].upper()} Section {m['old_section']} corresponds to "
            f"{m['new_act']} Section {m['new_section']}."
        )
    return list(dict.fromkeys(notes))


@dataclass
class Prepared:
    understood: UnderstoodQuery
    results: list[SearchResult]
    messages: list[dict[str, str]] = field(default_factory=list)
    abstain: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    message: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    invalid_citations: list[int]
    abstained: bool
    results: list[SearchResult]
    timings: dict[str, float]
    notes: list[str] = field(default_factory=list)


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
        uq = understand(
            question,
            (history or [])[-self.s.history_turns :],
            self.mapping,
            rewrite=self.s.qu_rewrite,
            refs=self.s.qu_refs,
            abbrev=self.s.qu_abbrev,
            use_mapping=self.s.qu_mapping,
        )
        t1 = time.perf_counter()
        wants_old = any(w in uq.standalone.lower().split() for w in _OLD_ACTS)
        status = None if wants_old else ("in_force", "amended")
        routed = domain is None and self.s.domain_router
        if routed:
            domain = route_domain(uq.standalone)
        filters = Filters(domain=domain, status=status)
        k = self.s.retrieval_top_k
        results = self.retriever.retrieve(
            uq.standalone, k=k, filters=filters, extra_terms=uq.expansion
        )
        if not results and routed and domain:  # a wrong route must never cause a miss
            filters = Filters(status=status)
            results = self.retriever.retrieve(
                uq.standalone, k=k, filters=filters, extra_terms=uq.expansion
            )
        rerank_ms = self.retriever.last_rerank_ms
        if uq.references:  # exact section/article references: pin matching chunks first
            exact = self.retriever.retrieve(
                uq.standalone,
                k=k,
                filters=replace(filters, sections=section_labels(uq.references)),
                extra_terms=uq.expansion,
            )[:2]
            exact_ids = {r.chunk_id for r in exact}
            results = (exact + [r for r in results if r.chunk_id not in exact_ids])[:k]
        t2 = time.perf_counter()
        timings = {
            "understand_ms": (t1 - t0) * 1e3,
            "retrieve_ms": (t2 - t1) * 1e3,
            "rerank_ms": rerank_ms,
        }
        best = max((r.dense_score for r in results), default=0.0)
        if not results or best < self.s.abstain_threshold:
            msg = abstain_message(uq.standalone, results)
            return Prepared(uq, results, abstain=True, timings=timings, message=msg)
        messages, kept = build_messages(
            self.s.prompt_version,
            question,
            results,
            self.s.context_max_chars,
            order=self.s.context_order,
            dedupe_threshold=self.s.context_dedupe_threshold,
        )
        notes = currency_notes(kept, uq.mapped)
        return Prepared(uq, kept, messages, timings=timings, notes=notes)

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
            return Answer(prepared.message, [], [], True, prepared.results, prepared.timings)
        t = time.perf_counter()
        text = "".join([tok async for tok in self.stream(prepared)])
        timings = {**prepared.timings, "generate_ms": (time.perf_counter() - t) * 1e3}
        cites, invalid = resolve_citations(text, prepared.results)
        text = strip_invalid_citations(text, len(prepared.results))
        return Answer(text, cites, invalid, False, prepared.results, timings, prepared.notes)
