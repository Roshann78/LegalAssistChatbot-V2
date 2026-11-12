"""Opt-in request tracing to a local JSONL file (Langfuse is not integrated).

Traces contain the user's question and the answer, so they are OFF unless TRACE_PATH is set.
The prompt text is deliberately not stored (it can contain uploaded-document text); retrieved
passages are recorded by id/section/score only.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class Tracer:
    """No-op tracer (default)."""

    enabled = False

    def write(self, record: dict[str, Any]) -> None:  # pragma: no cover - trivial
        return None


class FileTracer(Tracer):
    enabled = True

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps({"ts": time.time(), **record}, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_tracer(path: Path | None) -> Tracer:
    return FileTracer(path) if path else Tracer()


def find_trace(path: Path, request_id: str) -> dict | None:
    """Look up one request's trace by its request ID (also the ID feedback is tied to)."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("request_id") == request_id:
            return rec
    return None
