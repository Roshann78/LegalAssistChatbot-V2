"""Small concurrent load test against a running server.

    python scripts/load_test.py --url http://127.0.0.1:8000 --concurrency 10 --seconds 60

Reports request count, error rate (non-200 or SSE `error` events), and p50/p95 latency of
full streamed responses. It measures whatever backend the server uses; with demo_server.py that
is a fake LLM and a 4-document fixture, so the numbers describe the API layer only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

QUESTIONS = [
    "What is the punishment for theft?",
    "What does Section 3 say about theft?",
    "How often must KYC be updated for high risk customers?",
    "Must the grounds of arrest be communicated to the arrested person?",
    "Give me a recipe for chocolate cake",
]


async def worker(client: httpx.AsyncClient, url: str, deadline: float, wid: int, out: list):
    i = wid
    while time.monotonic() < deadline:
        q = QUESTIONS[i % len(QUESTIONS)]
        i += 1
        t0 = time.perf_counter()
        ok = False
        try:
            async with client.stream("POST", f"{url}/v1/ask", json={"question": q}) as r:
                body = (await r.aread()).decode()
                ok = r.status_code == 200 and "event: error" not in body
                status = r.status_code
        except httpx.HTTPError as exc:
            status = type(exc).__name__
        out.append((time.perf_counter() - t0, ok, status))


async def run(url: str, concurrency: int, seconds: int) -> dict:
    results: list = []
    deadline = time.monotonic() + seconds
    async with httpx.AsyncClient(timeout=60) as client:
        await asyncio.gather(
            *(worker(client, url, deadline, n, results) for n in range(concurrency))
        )
        metrics = (await client.get(f"{url}/v1/metrics")).json()
    lat = sorted(r[0] * 1e3 for r in results)
    errors = [r for r in results if not r[1]]
    return {
        "requests": len(results),
        "errors": len(errors),
        "error_rate": len(errors) / len(results) if results else 0.0,
        "p50_ms": statistics.median(lat) if lat else 0.0,
        "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
        "rps": len(results) / seconds,
        "server_metrics": metrics,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=60)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(run(a.url, a.concurrency, a.seconds)), indent=2))
