"""Versioned /v1 routes: ask (SSE), stats, feedback, admin document removal."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from legal_rag.api.middleware import get_request_id
from legal_rag.api.security import require_admin
from legal_rag.api.streaming import create_sse_response, format_sse
from legal_rag.generation.citations import resolve_citations, strip_invalid_citations
from legal_rag.llm import LLMError

logger = logging.getLogger("legal_rag.api.routes")
DISCLAIMER = "Informational only; not legal advice."


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    domain: Literal["legal", "banking"] | None = None
    conversation_id: str | None = Field(default=None, max_length=128)


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=8, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


def build_v1_router() -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/ask")
    async def ask(body: AskRequest, request: Request):
        st = request.app.state
        if len(body.question) > st.settings.max_question_length:
            raise HTTPException(422, f"question exceeds {st.settings.max_question_length} chars")
        pipeline = getattr(st, "pipeline", None)
        if pipeline is None:
            raise HTTPException(503, "Retrieval pipeline is not ready (no index loaded)")
        req_id = get_request_id()

        session = await st.session_store.get(body.conversation_id) if body.conversation_id else None
        if session is None:
            session = await st.session_store.create({"history": []})
        history: list[tuple[str, str]] = session.data.setdefault("history", [])
        window = st.settings.history_turns

        async def events() -> AsyncIterator[str]:
            t0 = time.perf_counter()
            meta = {"conversation_id": session.session_id, "request_id": req_id}
            try:
                prepared = await asyncio.to_thread(
                    pipeline.prepare, body.question, history[-window:], body.domain
                )
                if prepared.abstain:
                    yield format_sse(
                        "abstained", {**meta, "message": prepared.message, "disclaimer": DISCLAIMER}
                    )
                    yield format_sse("done", {**meta, "finish_reason": "abstained"})
                    return
                parts: list[str] = []
                async for tok in pipeline.stream(prepared):
                    if tok:
                        parts.append(tok)
                        yield format_sse("token", {"content": tok})
                answer = "".join(parts)
                cites, invalid = resolve_citations(answer, prepared.results)
                if invalid:
                    logger.warning("[%s] dropped invalid citations: %s", req_id, invalid)
                clean = strip_invalid_citations(answer, len(prepared.results))
                yield format_sse(
                    "citations",
                    {
                        "citations": [c.to_dict() for c in cites],
                        "invalid": invalid,
                        # Tokens already streamed may contain invalid markers; clients should
                        # replace the displayed text with this sanitized version.
                        "answer": clean,
                        "notes": prepared.notes,
                    },
                )
                history.append((body.question, clean))
                del history[: -window or None]
                await st.session_store.update(session)
                yield format_sse(
                    "done",
                    {
                        **meta,
                        "finish_reason": "stop",
                        "disclaimer": DISCLAIMER,
                        "latency_ms": round((time.perf_counter() - t0) * 1e3),
                        **{k: round(v) for k, v in prepared.timings.items()},
                    },
                )
            except asyncio.CancelledError:
                raise
            except LLMError as exc:
                logger.error("[%s] LLM failure: %s", req_id, type(exc).__name__)
                yield format_sse(
                    "error",
                    {
                        "code": "LLM_ERROR",
                        "request_id": req_id,
                        "message": "The language model is unavailable.",
                    },
                )
            except Exception:
                logger.exception("[%s] ask failed", req_id)
                yield format_sse(
                    "error",
                    {
                        "code": "STREAM_ERROR",
                        "request_id": req_id,
                        "message": "An error occurred while answering.",
                    },
                )

        return create_sse_response(events(), request_id=req_id)

    @router.get("/stats")
    async def stats(request: Request):
        vs = request.app.state.vector_store
        docs = vs.list_docs() if vs else []
        return {"documents": len(docs), "chunks": vs.count() if vs else 0}

    @router.post("/feedback", status_code=202)
    async def feedback(body: FeedbackRequest, request: Request):
        path = request.app.state.settings.feedback_path
        record = {"ts": time.time(), **body.model_dump()}

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        await asyncio.to_thread(write)
        return {"status": "recorded"}

    @router.delete("/admin/documents/{doc_id}", dependencies=[Depends(require_admin)])
    async def remove_document(doc_id: str, request: Request):
        vs = request.app.state.vector_store
        removed = vs.delete_by_doc_id(doc_id)
        if not removed:
            raise HTTPException(404, f"No chunks for document {doc_id!r}")
        return {"doc_id": doc_id, "chunks_removed": removed}

    return router
