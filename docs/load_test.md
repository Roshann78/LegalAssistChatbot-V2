# Load test (API layer only)

**What this measures:** the FastAPI layer, retrieval over a 4-document fixture, and SSE streaming. The LLM is
a fake that streams words with a 20 ms per-token delay, so latency is dominated by that delay and says
nothing about a real model or a real corpus.

Command: `python scripts/demo_server.py --port 8765 --delay-ms 20` then
`python scripts/load_test.py --url http://127.0.0.1:8765 --concurrency 10 --seconds 45`
(Windows 11, single process, hashing embedder, one run).

| Metric | Result |
|---|---|
| Requests completed | 398 (320 answered, 78 abstained by design: 1 of 5 questions is off-topic) |
| Errors (non-200 or SSE `error`) | 0 (0.0%) |
| Throughput | ~8.8 requests/s at 10 concurrent clients |
| End-to-end p50 / p95 | ~1400 ms / ~1520 ms (fake-LLM streaming delay) |
| Server-side retrieve p50 / p95 | 2.5 ms / 4.5 ms |
| Understand p50 | 0.1 ms |

Not done: the roadmap's 5-minute run, resource (CPU/RAM) measurement, a real LLM, and the failure drills
against a live composed stack (vector DB down, invalid key, slow LLM). The failure paths are covered by unit
tests instead: LLM failure becomes a safe `error` SSE event (`LLM_ERROR`, no provider details); a missing
index returns 503 on `/v1/ask`; `/ready` goes 503 when a component is not ready.
