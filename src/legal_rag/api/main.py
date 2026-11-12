"""FastAPI application factory, lifespan, routing, and exception handlers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from legal_rag.api.middleware import (
    RequestIDMiddleware,
    get_request_id,
    is_valid_request_id,
)
from legal_rag.api.routes import build_v1_router
from legal_rag.api.schemas import ErrorResponse, HealthResponse, ReadyResponse
from legal_rag.api.security import AbuseControlMiddleware
from legal_rag.api.sessions import (
    InMemorySessionStore,
    SessionStore,
    run_periodic_session_cleanup,
)
from legal_rag.config.settings import Settings, get_settings
from legal_rag.generation.pipeline import RAGPipeline
from legal_rag.llm import GroqClient, LLMAuthError, LLMClient, ResilientLLMClient
from legal_rag.observability.logging import setup_logging
from legal_rag.retrieval.base import (
    Embedder,
    PlaceholderEmbedder,
    PlaceholderVectorStore,
    VectorStore,
)
from legal_rag.retrieval.embedder import build_embedder
from legal_rag.retrieval.hybrid import Retriever
from legal_rag.retrieval.memory_store import InMemoryVectorStore
from legal_rag.retrieval.query import MappingTable

logger = logging.getLogger("legal_rag.api")


class UnreadyLLMClient(LLMClient):
    """Fallback LLM client when no API key or provider is configured."""

    def __init__(self, reason: str = "LLM_API_KEY is not configured") -> None:
        self.reason = reason

    def is_ready(self) -> bool:
        return False

    def generate(self, *args, **kwargs):
        raise LLMAuthError(self.reason)

    async def generate_stream(self, *args, **kwargs):
        raise LLMAuthError(self.reason)
        yield ""  # pragma: no cover


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
    session_store: SessionStore | None = None,
    pipeline: RAGPipeline | None = None,
) -> FastAPI:
    """Application factory for Legal Assist RAG API.

    Allows injecting custom settings or pre-built mock components for testing.
    """
    app_settings = settings or get_settings()
    setup_logging(app_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Initialize heavy shared components once at startup; close on shutdown."""
        # 1. Create shared HTTP client for outbound provider calls
        http_client = httpx.AsyncClient(timeout=app_settings.llm_timeout_seconds)
        app.state.http_client = http_client

        # 2. Initialize LLM client and wrap in resilience layer
        raw_llm: LLMClient
        if llm_client is not None:
            raw_llm = llm_client
        else:
            try:
                raw_llm = GroqClient(app_settings, async_client=http_client)
            except LLMAuthError as exc:
                logger.warning("LLM client not ready at startup: %s", exc)
                raw_llm = UnreadyLLMClient(reason=str(exc))

        # Always wrap in ResilientLLMClient for retry & backoff protection
        if isinstance(raw_llm, ResilientLLMClient):
            app.state.llm_client = raw_llm
        else:
            app.state.llm_client = ResilientLLMClient(raw_llm, settings=app_settings)

        # 3. Initialize embedder, vector store and pipeline (real ones only if an index exists)
        active_embedder, active_store = embedder, vector_store
        load_failed = False
        if active_store is None and app_settings.index_path.exists():
            try:
                active_store = InMemoryVectorStore.load(app_settings.index_path)
                if active_embedder is None:
                    active_embedder = build_embedder(app_settings.embedding_model)
            except Exception as exc:  # model/index failure must not crash startup
                logger.error("Failed to load index/embedder: %s", exc)
                active_store, active_embedder = None, None
                load_failed = True
        app.state.embedder = active_embedder or PlaceholderEmbedder(ready=not load_failed)
        app.state.vector_store = active_store or PlaceholderVectorStore(ready=not load_failed)
        app.state.pipeline = pipeline
        if (
            pipeline is None
            and isinstance(active_store, InMemoryVectorStore)
            and active_embedder is not None
        ):
            app.state.pipeline = RAGPipeline(
                Retriever(
                    active_embedder,
                    active_store,
                    candidates=app_settings.retrieval_candidates,
                    per_doc_cap=app_settings.retrieval_per_doc_cap,
                ),
                app.state.llm_client,
                app_settings,
                MappingTable.load(app_settings.mappings_path),
            )

        # 4. Initialize session store and background cleanup
        active_session_store = session_store or InMemorySessionStore(
            ttl_seconds=app_settings.session_ttl_seconds,
            max_sessions=app_settings.session_max_count,
        )
        app.state.session_store = active_session_store
        cleanup_task = asyncio.create_task(
            run_periodic_session_cleanup(active_session_store, interval_seconds=60.0)
        )

        app.state.settings = app_settings

        logger.info("Application components initialized (lifespan startup complete)")

        yield

        # 5. Clean shutdown
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        await http_client.aclose()
        logger.info("Application components cleanly shut down (lifespan shutdown complete)")

    app = FastAPI(
        title="Legal Assist RAG API",
        version="0.1.0",
        description="Production-style Legal & Banking RAG System",
        lifespan=lifespan,
    )

    # ── Middleware (Order: outer first) ──────────────────────────────────
    # RequestIDMiddleware wraps ASGI scope to propagate X-Request-ID
    app.add_middleware(
        AbuseControlMiddleware,
        rate_limit_rpm=app_settings.rate_limit_rpm,
        max_body_bytes=app_settings.max_body_bytes,
    )
    app.add_middleware(RequestIDMiddleware)

    # CORS configuration: Never allow credentials when wildcard origin is set
    is_wildcard = "*" in app_settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=not is_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ───────────────────────────────────────────────

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or get_request_id()
        logger.warning("[%s] Request validation error: %s", req_id, exc.errors())
        error_body = ErrorResponse(
            code="VALIDATION_ERROR",
            message="Invalid request parameters",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=422,
            content=error_body.model_dump(),
            headers={"x-request-id": req_id},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None) or get_request_id()
        code_map = {
            404: "NOT_FOUND",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE",
            422: "VALIDATION_ERROR",
            429: "RATE_LIMITED",
            503: "SERVICE_UNAVAILABLE",
        }
        code = code_map.get(exc.status_code, "HTTP_ERROR")
        error_body = ErrorResponse(
            code=code,
            message=str(exc.detail),
            request_id=req_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body.model_dump(),
            headers={"x-request-id": req_id},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Extract request ID from request state, header, or contextvar
        req_id = getattr(request.state, "request_id", None)
        if not req_id:
            incoming = request.headers.get("x-request-id")
            if incoming and is_valid_request_id(incoming):
                req_id = incoming
            else:
                ctx_id = get_request_id()
                req_id = ctx_id if ctx_id != "unknown-request-id" else str(uuid.uuid4())

        # Log the full exception server-side with traceback
        logger.exception("[%s] Unhandled server exception: %s", req_id, exc)
        # Return generic error body to client with zero internal stack details
        error_body = ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred.",
            request_id=req_id,
        )
        return JSONResponse(
            status_code=500,
            content=error_body.model_dump(),
            headers={"x-request-id": req_id},
        )

    # ── Routes ───────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe: always returns 200 OK."""
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready(request: Request) -> JSONResponse:
        """Readiness probe: checks local state of dependencies.

        Does NOT make external network calls. Returns 200 if all components are
        ready, or 503 with component status details if any component is not ready.
        """
        llm = getattr(request.app.state, "llm_client", None)
        emb = getattr(request.app.state, "embedder", None)
        vs = getattr(request.app.state, "vector_store", None)

        llm_ready = llm.is_ready() if llm else False
        emb_ready = emb.is_ready() if emb else False
        vs_ready = vs.is_ready() if vs else False

        components = {
            "llm": "ready" if llm_ready else "not_ready",
            "embedder": emb.get_status() if emb else "not_ready",
            "vector_store": vs.get_status() if vs else "not_ready",
        }

        all_ready = llm_ready and emb_ready and vs_ready
        status_code = 200 if all_ready else 503
        status_str = "ready" if all_ready else "not_ready"

        response = ReadyResponse(status=status_str, components=components)
        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(),
            headers={"x-request-id": get_request_id()},
        )

    app.include_router(build_v1_router())
    return app
