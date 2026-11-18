"""Versioned /v1 routes: ask (SSE), uploads, stats, metrics, feedback, admin."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from legal_rag.api.middleware import get_request_id
from legal_rag.api.security import require_admin
from legal_rag.api.streaming import create_sse_response, format_sse
from legal_rag.api.uploads import UploadError, build_upload_index
from legal_rag.generation.citations import resolve_citations, strip_invalid_citations
from legal_rag.generation.pipeline import RAGPipeline
from legal_rag.llm import LLMError
from legal_rag.observability.tracing import find_trace
from legal_rag.retrieval.factory import build_retriever

logger = logging.getLogger("legal_rag.api.routes")
DISCLAIMER = "Informational only; not legal advice."


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    domain: Literal["legal", "banking"] | None = None
    conversation_id: str | None = Field(default=None, max_length=128)
    upload_id: str | None = Field(default=None, max_length=128)


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=8, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


def _trace_record(req_id, body, prepared, outcome, answer, cites, timings, upload) -> dict:
    rec: dict = {
        "request_id": req_id,
        "outcome": outcome,
        "question": body.question,
        "upload": bool(upload),
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
    }
    if prepared is not None:
        rec["standalone"] = prepared.understood.standalone
        rec["references"] = prepared.understood.references
        rec["retrieved"] = [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "section": r.section,
                "score": round(r.score, 4),
                "dense": round(r.dense_score, 4),
            }
            for r in prepared.results
        ]
        rec["prompt_chars"] = sum(len(m["content"]) for m in prepared.messages)
        rec["notes"] = prepared.notes
    rec["answer"] = answer
    rec["citations"] = [c.index for c in cites]
    return rec


def build_v1_router() -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/ask")
    async def ask(body: AskRequest, request: Request):
        st = request.app.state
        if len(body.question) > st.settings.max_question_length:
            raise HTTPException(422, f"question exceeds {st.settings.max_question_length} chars")
        pipeline = getattr(st, "pipeline", None)
        if body.upload_id:
            upload_store = st.uploads.get(body.upload_id)
            if upload_store is None:
                raise HTTPException(404, "Upload not found or expired")
            pipeline = RAGPipeline(
                build_retriever(st.settings, st.embedder, upload_store),
                st.llm_client,
                st.settings,
            )
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
            outcome, prepared, answer, cites, timings = "error", None, "", [], {}
            try:
                prepared = await asyncio.to_thread(
                    pipeline.prepare,
                    body.question,
                    history[-window:],
                    None if body.upload_id else body.domain,
                )
                timings = dict(prepared.timings)
                if prepared.abstain:
                    outcome = "abstained"
                    yield format_sse(
                        "abstained", {**meta, "message": prepared.message, "disclaimer": DISCLAIMER}
                    )
                    yield format_sse("done", {**meta, "finish_reason": "abstained"})
                    return
                parts: list[str] = []
                tg = time.perf_counter()
                async for tok in pipeline.stream(prepared):
                    if tok:
                        parts.append(tok)
                        yield format_sse("token", {"content": tok})
                timings["generate_ms"] = (time.perf_counter() - tg) * 1e3
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
                timings["total_ms"] = (time.perf_counter() - t0) * 1e3
                outcome = "answered"
                yield format_sse(
                    "done",
                    {
                        **meta,
                        "finish_reason": "stop",
                        "disclaimer": DISCLAIMER,
                        "latency_ms": round(timings["total_ms"]),
                        **{k: round(v) for k, v in timings.items() if k != "total_ms"},
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
            finally:
                timings.setdefault("total_ms", (time.perf_counter() - t0) * 1e3)
                st.metrics.record(outcome, timings, len(answer) // 4)
                logger.info(
                    "ask outcome=%s stages_ms=%s",
                    outcome,
                    json.dumps({k: round(v, 1) for k, v in timings.items()}),
                )
                if st.tracer.enabled:
                    st.tracer.write(
                        _trace_record(
                            req_id, body, prepared, outcome, answer, cites, timings, body.upload_id
                        )
                    )

        return create_sse_response(events(), request_id=req_id)

    @router.post("/documents/upload", status_code=201)
    async def upload(request: Request, filename: str = Query(min_length=1, max_length=200)):
        """Raw-body upload (`?filename=x.pdf`): an ephemeral index for chatting with one file."""
        st = request.app.state
        limit = st.settings.upload_max_bytes
        data = bytearray()
        async for chunk in request.stream():  # bounded even without a Content-Length header
            data += chunk
            if len(data) > limit:
                raise HTTPException(413, f"File exceeds {limit} bytes")
        try:
            store, name, pages = await asyncio.to_thread(
                build_upload_index, filename, bytes(data), st.settings, st.embedder
            )
        except UploadError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(503, "Embedder is not ready for uploads") from exc
        uid = st.uploads.add(store)
        return {
            "upload_id": uid,
            "filename": name,
            "pages": pages,
            "chunks": store.count(),
            "expires_in_seconds": st.settings.upload_ttl_seconds,
        }

    @router.delete("/documents/session/{upload_id}", status_code=204)
    async def delete_upload(upload_id: str, request: Request):
        if not request.app.state.uploads.delete(upload_id):
            raise HTTPException(404, "Upload not found or expired")

    @router.get("/stats")
    async def stats(request: Request):
        vs = request.app.state.vector_store
        docs = vs.list_docs() if vs else []
        return {"documents": len(docs), "chunks": vs.count() if vs else 0}

    @router.get("/metrics")
    async def metrics(request: Request):
        return request.app.state.metrics.snapshot()

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

    @router.get("/admin/traces/{request_id}", dependencies=[Depends(require_admin)])
    async def get_trace(request_id: str, request: Request):
        path = request.app.state.settings.trace_path
        rec = await asyncio.to_thread(find_trace, path, request_id) if path else None
        if rec is None:
            raise HTTPException(404, "No trace for that request ID (is TRACE_PATH set?)")
        return rec

    @router.delete("/admin/documents/{doc_id}", dependencies=[Depends(require_admin)])
    async def remove_document(doc_id: str, request: Request):
        vs = request.app.state.vector_store
        removed = vs.delete_by_doc_id(doc_id)
        if not removed:
            raise HTTPException(404, f"No chunks for document {doc_id!r}")
        return {"doc_id": doc_id, "chunks_removed": removed}

    return router
