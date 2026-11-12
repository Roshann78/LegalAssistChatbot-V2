"""In-process operational metrics: stage latency percentiles, error and abstention rates."""

from __future__ import annotations

import threading
from collections import defaultdict, deque

from legal_rag.eval.metrics import percentile


class Metrics:
    """Bounded, thread-safe. Percentiles are over the most recent `window` samples per stage."""

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._stages: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._tokens = 0

    def record(self, outcome: str, timings: dict[str, float], tokens_out: int = 0) -> None:
        """outcome: answered | abstained | error."""
        with self._lock:
            self._counts["requests"] += 1
            self._counts[outcome] += 1
            self._tokens += tokens_out
            for stage, ms in timings.items():
                self._stages[stage].append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            n = self._counts["requests"]
            return {
                "requests": n,
                "answered": self._counts["answered"],
                "abstained": self._counts["abstained"],
                "errors": self._counts["error"],
                "error_rate": self._counts["error"] / n if n else 0.0,
                "abstention_rate": self._counts["abstained"] / n if n else 0.0,
                "approx_output_tokens_per_request": self._tokens / n if n else 0.0,
                "latency_ms": {
                    stage: {
                        "p50": round(percentile(list(vals), 50), 2),
                        "p95": round(percentile(list(vals), 95), 2),
                        "n": len(vals),
                    }
                    for stage, vals in sorted(self._stages.items())
                },
            }
