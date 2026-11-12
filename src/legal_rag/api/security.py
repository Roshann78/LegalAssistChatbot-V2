"""Abuse controls: per-IP rate limiting and request-size limits (pure ASGI)."""

from __future__ import annotations

import json
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from fastapi import Header, HTTPException, Request

from legal_rag.api.middleware import get_request_id

PROTECTED_PREFIXES = ("/v1/",)


class AbuseControlMiddleware:
    """Rejects oversized bodies (413) and over-limit clients (429) on /v1 routes.

    Note: size is enforced from Content-Length; chunked uploads are bounded by the server.
    """

    def __init__(
        self,
        app: Callable,
        *,
        rate_limit_rpm: int,
        max_body_bytes: int,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.rpm, self.max_body = rate_limit_rpm, max_body_bytes
        self.time_fn = time_fn
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def _reject(self, send: Callable, status: int, code: str, msg: str, extra=None) -> None:
        body = json.dumps({"code": code, "message": msg, "request_id": get_request_id()}).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        headers += extra or []
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(PROTECTED_PREFIXES):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        length = headers.get(b"content-length")
        if length and length.isdigit() and int(length) > self.max_body:
            await self._reject(send, 413, "PAYLOAD_TOO_LARGE", "Request body too large")
            return
        ip = (scope.get("client") or ("unknown", 0))[0]
        now = self.time_fn()
        window = self._hits[ip]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.rpm:
            retry = max(1, int(60 - (now - window[0])))
            await self._reject(
                send,
                429,
                "RATE_LIMITED",
                "Too many requests",
                [(b"retry-after", str(retry).encode())],
            )
            return
        window.append(now)
        await self.app(scope, receive, send)


async def require_admin(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Dependency: admin endpoints need the configured key (disabled if none is set)."""
    expected = request.app.state.settings.admin_api_key.get_secret_value()
    if not expected:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
