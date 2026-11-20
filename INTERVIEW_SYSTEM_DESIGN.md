# System Design and Scaling

Numbers marked **(measured)** come from this repo or from my real run. Numbers marked **(estimate)** are back-of-envelope and should be presented as such.

---

## 1. Current architecture (topology)

```
 Browser (Next.js :3010 / static /ui)
        │  HTTPS, SSE
        ▼
 ┌──────────────────────────────── 1 process (uvicorn, 1 worker) ─────────────────────────────┐
 │  Middleware: RequestID → AbuseControl (per-IP 60/min, body ≤ 1 MB, upload ≤ 5 MB) → CORS   │
 │  FastAPI routes                                                                            │
 │     │ asyncio.to_thread                  │ async                                           │
 │     ▼                                    ▼                                                 │
 │  Pipeline.prepare()                  LLM stream (ResilientLLMClient → GroqClient)          │
 │   ├ understand (rules)                   │ httpx → Groq API (external, internet)           │
 │   ├ embed query (MiniLM, in-process)     │                                                 │
 │   ├ dense scan + BM25 (in-memory store, ~27k chunks)                                       │
 │   └ RRF → cap → pin sections → gate                                                        │
 │                                                                                            │
 │  In-process state: sessions (TTL 3600s, ≤1000) · uploads (TTL 1800s, ≤50) · metrics        │
 │  (last 1000 samples/stage) · rate-limit windows                                            │
 └───────────────┬────────────────────────────────────────────────────────────────────────────┘
                 │ files
   data/processed/index.json (chunks + vectors) · data/cache/embeddings.sqlite
   data/manifest.yaml · data/raw/{legal,banking}/*.pdf · data/processed/feedback.jsonl
```

Offline: `python -m legal_rag.ingestion sync` writes the same files; `AUTO_SYNC_ON_STARTUP=true` runs it inside start-up.

---

## 2. Bottleneck analysis

| # | Component | Limit / single point of failure | Evidence |
|---|---|---|---|
| 1 | Single uvicorn process | One process, one machine: no redundancy; a crash drops every session and upload | `api/main.py` builds all state in `app.state` |
| 2 | LLM provider (Groq) | External dependency; dominates latency (generation p50 about 1.1 s of 1.14 s total) | Metrics screenshot, 3 requests **(measured)** |
| 3 | Dense search | Full scan, O(N·d) per query in pure Python (`memory_store.py:111-119`) | code |
| 4 | BM25 | Term counters and document frequencies rebuilt per query over all chunks (`memory_store.py:126-129`) | code |
| 5 | Index load | Whole `index.json` read and held in RAM at start (`InMemoryVectorStore.load`) | code |
| 6 | Embedding model | Loaded in every process; CPU inference for each query (and each upload chunk) | `embedder.py` |
| 7 | Sessions / uploads | Process-local; lost on restart; not shared across replicas | `sessions.py`, `uploads.py` |
| 8 | Rate limiter and metrics | Per-process dicts; limits multiply with replicas | `security.py`, `metrics.py` |
| 9 | File writes | Feedback appended to a local JSONL; trace lookup scans a file linearly | `routes.py:236`, `tracing.py:44` |
| 10 | Ingestion | Single-threaded, embeds sequentially in batches of 64 | `indexer.py:_embed` |
| 11 | Observed retrieval spread | Median 38 ms but p95 2.5 s in 3 samples, unexplained **(measured)** | Metrics screenshot |

---

## 3. Scaling strategies per bottleneck

| Bottleneck | Strategy | Notes |
|---|---|---|
| 1 Single process | Several uvicorn workers or replicas behind a load balancer; health from `/health`, traffic gating from `/ready` | Needs 7 and 8 solved first |
| 2 LLM latency | Stream (done), cap `max_tokens`, shorter context, response cache for repeated questions, provider fallback behind `LLMClient` | A cache needs a key of (question, corpus fingerprint, prompt version) |
| 3 Dense scan | Qdrant (already an adapter) with HNSW ANN; or NumPy matrix multiply as a cheap first step | NumPy over 27k×384 float32 is a single matrix-vector product |
| 4 BM25 | Build and cache statistics at index load; update incrementally on upsert; or use Qdrant sparse | Removes per-query O(corpus) |
| 5 Index load | Use Qdrant server mode; or memory-map vectors | Memory store is for local use |
| 6 Embedding | Batch queries, ONNX or quantised model, or a dedicated embedding service; GPU if QPS justifies | Keep the `Embedder` interface |
| 7 Sessions | Redis with TTL (interface `SessionStore` already abstract) | `InMemorySessionStore` is one implementation |
| 8 Rate limit/metrics | Redis counters at the gateway; Prometheus for metrics | Don't rely on per-process counts |
| 9 Files | Append to a queue or database; index traces by request ID | |
| 10 Ingestion | Worker pool for parse and embed; keep the embedding cache; batch up to model limits | Already idempotent, so safe to parallelise by document |
| Corpus growth | Shard by domain (legal/banking) or by document family; filters already exist | Payload indexes are created for 6 fields |

---

## 4. Target architectures

### 4.1 Today → 10x (about 300k chunks, tens of concurrent users)

```
 Users ─► CDN/Vercel (Next.js) ─► API (3 replicas) ─► Qdrant (1 node, HNSW)
                                     │                    ▲
                                     ├──► Redis (sessions, uploads index refs, rate limit)
                                     ├──► Groq API (retry/backoff, fallback provider)
                                     └──► Embedding model in-process (CPU)
 Ingestion job (worker) ──► embeddings cache (shared volume/SQLite→Postgres) ──► Qdrant
```
Changes: Qdrant instead of the memory store; Redis for session and rate-limit state; build the BM25 index once or use Qdrant sparse; add a cross-encoder reranker over the top 30 only if the measured precision demands it.

### 4.2 100x (about 3M chunks, hundreds of concurrent users)

```
 Users ─► CDN ─► API Gateway (auth, global rate limit, request IDs)
                      │
              ┌───────┴────────┐
              ▼                ▼
        API pods (N, autoscaled on RPS)     Embedding service (GPU/ONNX, batched)
              │  │  │                              ▲
              │  │  └─► Redis cluster (sessions, cache) │
              │  └────► Qdrant cluster (sharded by domain, replicated)
              └──────► LLM gateway (queue, quotas, provider fallback, response cache)
 Ingestion pipeline: object storage (PDFs) → queue → parse/embed workers → Qdrant
 Observability: Prometheus + Grafana, trace store keyed by request ID, eval-in-CI + nightly golden run
```

### 4.3 1000x (about 30M chunks, thousands of concurrent users)

```
 Global CDN ─► Regional gateways ─► Stateless API fleet (multi-region)
                                       │
        ┌──────────────┬───────────────┼───────────────┬────────────────┐
        ▼              ▼               ▼               ▼                ▼
   Query cache     Embedding      Qdrant shards    LLM routing       Feedback/trace
   (semantic +     fleet (GPU)    (per domain +    (multi-provider,  pipeline (Kafka →
   exact)                         per doc family)  self-hosted tier) warehouse)
 Corpus: versioned snapshots, blue/green index swaps keyed by config fingerprint
 Eval: continuous regression runs on a frozen golden set; answer sampling by judge model
```
Fingerprint-based swaps matter here: build the new index offline under a new fingerprint, validate, then switch traffic, instead of mutating the live index.

---

## 5. Failure scenarios

| What fails | What happens today | What I'd do |
|---|---|---|
| Groq returns 429 or 5xx before the first token | `ResilientLLMClient` retries up to 3 times with backoff and `Retry-After`; then SSE `error` `LLM_ERROR` | Fallback provider; surface "try again" in the UI |
| Groq fails mid-stream | No retry; error propagates; client shows partial text plus an error event | Mark the partial answer unverified in the UI |
| Invalid or missing API key | `LLMAuthError`; `/ready` reports `llm: not_ready` | Alert on readiness |
| Embedding model fails to load | Startup catches it, components become placeholders, `/ready` 503, `/v1/ask` 503 | Alert; fall back to sparse-only mode (not implemented) |
| Index file missing | Same as above, or `AUTO_SYNC_ON_STARTUP` builds it | Bake the index into the image |
| Index built with a different embedder | Sync refuses ("run sync --force"), but the API can still load and serve the stale file | Make the mismatch fail readiness |
| Process crash | All sessions, uploads and metrics lost | Redis for sessions; restart policy |
| Disk full | Feedback write raises inside a thread; ask path unaffected | Alert on disk; send feedback to a queue |
| Qdrant down (if used) | Store calls raise, ask returns `STREAM_ERROR`; `/ready` only checks local flags | Make `/ready` ping the store; circuit breaker |
| Malformed or huge upload | 413/415/422 before indexing; temp dir removed even on failure | Virus scanning if public |
| Abusive client | 429 with `Retry-After`; per-IP window | Gateway-level limits, authenticated quotas |
| Slow client on SSE | Server writes back-pressured by the transport | Timeout idle streams |

---

## 6. Capacity planning (back-of-envelope)

**Index size**
- Chunks **26,974 (measured)**, average text 75 to 169 tokens, about 640 characters average on one document **(measured)**.
- Vectors: 26,974 × 384 floats = 10.36M floats. As float32 that's about 41 MB **(estimate)**. Held as Python lists of floats, roughly 8 B pointer + 24 B float object per value, about 330 MB **(estimate)**.
- `index.json` stores vectors as JSON text, on the order of 150 to 250 MB **(estimate)**; load time scales with it.
- Scale to 10x (270k chunks): about 410 MB as float32, which is why Qdrant (compact float32 + HNSW) is the next step.

**Latency budget** (3-request sample, **measured**): total p50 1,144 ms = generation 1,105 + retrieval 38 + understanding 0.1. The LLM is about 97% of the budget at p50.

**Throughput**
- Load test with a fake 20 ms/token LLM and a 4-document corpus: **8.8 req/s at 10 clients, 0 errors (measured)**. That tests the API layer, not the real model.
- For the real system, concurrency is bounded by (a) provider rate limits, (b) CPU for query embedding and dense search in threads, (c) the default rate limit of 60 requests per minute per IP.
- Rule of thumb: concurrent in-flight requests ≈ QPS × latency. At 5 QPS and about 1.5 s per request, about 8 requests are in flight; async handles that easily, but each needs a retrieval thread slice.

**Memory per replica (estimate):** index (about 0.3 to 0.5 GB at 27k chunks) + MiniLM and its runtime (hundreds of MB) + Python overhead; plan 1 to 2 GB per replica for the current corpus.

**Ingestion time**
- My first real sync of 10 PDFs on the repo's hashing embedder produced 12,828 chunks in about 19 s **(measured, hashing embedder)**; a second run that skipped everything took about 4 s **(measured)**. With MiniLM on CPU, embedding dominates and the model download adds a one-time cost.

**Cost model.** Cost per question ≈ input tokens (≤ 12,000 characters of context, about 3,000 tokens **(estimate)** plus the question) + output up to 2,048 tokens. I haven't measured the cost per query (`price_in_per_1k` and `price_out_per_1k` exist in settings but are 0 by default).
