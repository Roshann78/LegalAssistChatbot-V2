# Technology Deep Dive

For each technology: what it is, why it is in this project, how it works inside, and why not the alternative. Project references are to this repository.

---

## 1. FastAPI, Starlette and Uvicorn (ASGI)

**What it is.** FastAPI is a Python web framework built on Starlette (the ASGI toolkit) and Pydantic. Uvicorn is the ASGI server that runs it.

**Why I chose it.** I need async streaming responses (SSE), request validation, and generated OpenAPI docs (`/docs`, which I screenshotted for the report). `create_app()` in `api/main.py:82` is an application factory, so tests inject fakes.

**How it works internally.** ASGI is a calling convention: an app is `async def app(scope, receive, send)`. Uvicorn parses the HTTP request, builds `scope`, and calls the app; the app sends `http.response.start` and `http.response.body` messages. FastAPI routes the scope to a path operation, validates the body with a Pydantic model, and for a `StreamingResponse` iterates an async generator, sending each yielded string as a body chunk with `more_body=True`. My `AbuseControlMiddleware` (`api/security.py`) is a *pure ASGI* middleware: it inspects `scope["headers"]` and either calls the next app or sends its own 413/429 messages, which avoids the overhead and quirks of `BaseHTTPMiddleware`.

**Why not the alternative.** *Flask* is WSGI: no native async streaming, so SSE needs workarounds. *Django* is heavier than I need. *Raw Starlette* would work, but I'd lose automatic validation and OpenAPI.

---

## 2. Server-Sent Events (SSE)

**What it is.** A one-way text stream over HTTP with `Content-Type: text/event-stream`, framed as `event: name\ndata: {...}\n\n`.

**Why.** The answer flows server to client only. SSE gives incremental tokens with a plain HTTP request and works through proxies that allow streaming. My events: `token`, `citations`, `abstained`, `done`, `error` (`api/streaming.py:format_sse`).

**How it works.** The server keeps the response open and writes frames as tokens arrive; a blank line ends a frame. The browser `EventSource` API can only issue GET, so `web/lib/api.ts` uses `fetch` with a `ReadableStream` reader and parses frames itself.

**Why not WebSockets.** Bidirectional framing and connection state I don't need. **Why not long-polling.** Higher latency and more requests.

---

## 3. Pydantic v2 and pydantic-settings

**What it is.** Runtime data validation based on type hints (v2 core is written in Rust). `pydantic-settings` loads settings from environment variables and `.env`.

**Why.** One typed definition for the manifest (`DocumentRecord`), request bodies (`AskRequest`, `FeedbackRequest`), responses (`ErrorResponse`) and configuration (`Settings`, about 70 fields). `SecretStr` keeps API keys out of logs and dumps. `field_validator` handles `llm_provider`, retrieval modes and CORS parsing.

**How it works.** Models compile to a validator schema at class creation; validation coerces and checks types, constraints (`ge`, `le`, `pattern`, `max_length`) and runs custom validators, raising `ValidationError` with per-field locations (my manifest loader reports `[('field',)]` for each bad record). In settings, `NoDecode` stops pydantic-settings from forcing JSON decoding on a list field, so my own validator can accept comma-separated values.

**Why not the alternative.** *dataclasses* give no validation. *Marshmallow* is a separate schema layer and wouldn't integrate with FastAPI's OpenAPI generation.

---

## 4. httpx (and the Groq API)

**What it is.** An async-capable HTTP client. Groq exposes an OpenAI-compatible chat-completions endpoint at `https://api.groq.com/openai/v1`.

**Why.** `GroqClient` (`llm/groq_client.py`) posts `{model, messages, temperature, max_tokens, stream}` with a bearer token, so no vendor SDK is needed. Tests use `httpx.MockTransport` to return canned status codes and streaming bodies with no network.

**How it works.** With `stream: true` the server returns SSE lines `data: {"choices":[{"delta":{"content":"..."}}]}` ending with `data: [DONE]`; the client reads `aiter_lines()`, parses each JSON line and yields the content delta. HTTP errors map to typed exceptions: 401/403 → `LLMAuthError`, 429 → `LLMRateLimitError(retry_after)`, 5xx → `LLMProviderError(status)`, timeouts → `LLMTimeoutError`.

**Why not the alternative.** The official `groq`/`openai` SDKs add a dependency and hide behaviour I want to test. *requests* is synchronous and can't stream asynchronously.

---

## 5. Retry with exponential backoff and jitter (`llm/resilience.py`)

**What it is.** A wrapper class `ResilientLLMClient` around any `LLMClient`.

**Why.** Providers return 429 and transient 5xx. `is_transient_error` retries only timeouts, rate limits and 5xx, never 401/403/4xx (`resilience.py:25`).

**How it works.** Wait = `min(min_wait · 2^attempt + U(0, 0.5·min_wait), max_wait)`, or the provider's `Retry-After` when present (`calculate_backoff`, line 43). Defaults: 3 retries, 0.5 s to 8 s. For streams, the key rule is lines 123-136: once any token has been yielded, an exception is re-raised without retry, because restarting would duplicate or corrupt text the user already saw. Before the first token it's safe to retry.

**Why not the alternative.** Libraries like `tenacity` are fine, but the stream rule is domain-specific, and the sleep functions need to be injectable for instant tests.

---

## 6. PyMuPDF (`fitz`)

**What it is.** Python bindings for MuPDF, a C PDF renderer and text extractor.

**Why.** Fast text extraction with page numbers and no external services. `parse_pdf` (`ingestion/parser.py`) reads `page.get_text("text")` per page and marks pages with under 50 characters as `low_text`. `classify.py` also reads PDF metadata and the first three pages.

**How it works.** MuPDF parses the PDF's page content streams and font encodings, then reconstructs text in reading order. It extracts embedded text only; scanned pages without a text layer return nothing, which is why I flag low-text pages.

**Why not the alternative.** *pdfminer.six* is slower and more verbose; *PyPDF* is weaker at layout. *OCR stacks* (Tesseract) solve a different problem and aren't in this codebase. Note the licence (AGPL or commercial): worth checking before commercial deployment.

---

## 7. Sentence-Transformers and `all-MiniLM-L6-v2`

**What it is.** A library for sentence embeddings; `all-MiniLM-L6-v2` is a 6-layer, 384-dimensional transformer distilled from MiniLM and trained with contrastive learning on over a billion sentence pairs. It truncates input at 256 word pieces by default.

**Why.** Small and CPU-friendly (about 90 MB), a strong baseline, and it's the default `embedding_model`. My `chunk_max_tokens` of 256 lines up with its input limit. It sits behind the `Embedder` interface (`retrieval/embedder.py`), and I call `encode(..., normalize_embeddings=True, batch_size=32)`.

**How it works.** The text is tokenised into word pieces, passed through the transformer, and the token outputs are mean-pooled into one vector; normalisation makes it unit length so a dot product equals cosine similarity. Similar meanings land close together in the 384-dimensional space.

**Why not the alternative.** *OpenAI embeddings* add cost and send text off-box. *Larger models* (bge, e5-large) are better on benchmarks but slower; I haven't run a bake-off on my real corpus (the `bakeoff` command exists), so I can't claim MiniLM is optimal. **Known limitation:** an embedding model doesn't know an acronym the text never contains. That caused the "DPDP" failure.

---

## 8. The hashing embedder

**What it is.** A deterministic bag-of-words embedder: each token is MD5-hashed to one of 512 buckets with a ±1 sign, then the vector is L2-normalised (`embedder.py:25-50`).

**Why.** CI and tests must not download a model. It's explicitly not semantic: it only captures word overlap.

**How it works.** The "hashing trick" avoids a vocabulary; the sign bit reduces collision bias. Two texts sharing words have a positive dot product.

**Why not the alternative.** Random vectors don't preserve overlap; a real model needs a download and nondeterministic hardware paths.

---

## 9. BM25 (sparse retrieval)

**What it is.** A ranking function: `score = Σ idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·len/avglen))`.

**Why.** Exact identifiers ("Section 66A", "Data Fiduciary") matter in law. The in-memory store implements it in pure Python with `k1 = 1.5`, `b = 0.75` (`memory_store.py:121-143`); Qdrant uses hashed term counts plus an IDF modifier (`qdrant_store.py:31`).

**How it works.** Term frequency saturates (diminishing returns for repeats), IDF down-weights common terms, and length normalisation stops long chunks winning by volume. **Caveat in my code:** I rebuild the term counters and document frequencies on every query, an O(corpus) cost I'd cache.

**Why not the alternative.** *Elasticsearch/OpenSearch* is more infrastructure than a local project needs. *TF-IDF* lacks saturation and length normalisation.

---

## 10. Reciprocal Rank Fusion

**What it is.** A rank-aggregation method: `RRF(d) = Σ w_i / (k0 + rank_i(d))` with `k0 = 60` (`hybrid.py:13-28`).

**Why.** Combines the dense and sparse lists without calibrating scores. It also keeps the maximum dense score per chunk so the abstention gate still sees a cosine value.

**How it works.** Each list contributes a smaller amount the lower a chunk ranks; chunks that appear in both lists get two contributions. Then `cap_per_doc` limits each document to 3 results for diversity.

**Why not the alternative.** *Weighted score fusion* needs normalisation of incomparable scores; *learned rankers* need training data I don't have.

---

## 11. Qdrant

**What it is.** An open-source vector database written in Rust.

**Why.** An optional backend (`vector_store_backend=qdrant`) for scale: named vectors (`dense` cosine, `sparse` with IDF), payload filters, and local or server mode. My adapter (`retrieval/qdrant_store.py`) derives deterministic UUID5 point IDs from chunk IDs so upserts overwrite, and stores the config fingerprint in collection metadata.

**How it works.** Dense vectors are indexed with HNSW (a layered proximity graph) for approximate nearest-neighbour search; payload fields marked as keyword indexes make filtered queries (`domain`, `status`, `section_or_article_no`) fast. Sparse vectors are stored as index/value lists and scored with a dot product, with `Modifier.IDF` applying inverse document frequency server-side.

**Why not the alternative.** *Chroma/FAISS* are fine for prototypes but lack Qdrant's filtering and hybrid support; *pgvector* ties me to Postgres; *Pinecone* is hosted-only. I haven't benchmarked them against each other.

---

## 12. SQLite (embedding cache)

**What it is.** An embedded, serverless SQL database in a single file (stdlib `sqlite3`).

**Why.** `ingestion/cache.py` stores `emb(model, key, vec)` with primary key `(model, key)`. Re-indexing a changed document only embeds new text; switching models doesn't collide.

**How it works.** WAL-less default journaling, a B-tree on the primary key; reads batch 500 keys per `IN (...)` query; vectors are stored as JSON text.

**Why not the alternative.** *Redis* adds a server; *a pickle file* isn't concurrent-safe or queryable; *Postgres* is overkill.

---

## 13. YAML manifest and text-level editing

**What it is.** A human-editable document registry parsed with PyYAML.

**Why.** Metadata needs review in git and manual overrides (marking an Act superseded). `manifest_edit.py` appends, removes and re-hashes entries as *text*, so comments and ordering survive.

**How it works.** `append_entries` detects the list indentation of the existing file and prefixes the dumped entry with a comment; `remove_entry` finds the entry's block by indentation; `replace_file_hash` replaces a unique 64-hex string.

**Why not the alternative.** Re-dumping with `yaml.safe_dump` would drop comments and reorder keys; `ruamel.yaml` preserves them but adds a dependency.

---

## 14. asyncio and threading

**What it is.** Cooperative concurrency (`asyncio`) plus OS threads for CPU work.

**Why.** The event loop must never block. `prepare()` (embedding plus search) runs via `asyncio.to_thread`; the LLM stream is natively async; background cleanup is an `asyncio.create_task`. The session store uses `asyncio.Lock`; the upload store and metrics use `threading.Lock` because threads touch them.

**How it works.** `to_thread` submits to the default thread-pool executor and awaits the future, freeing the loop. A cancelled request raises `CancelledError`, which I re-raise so cleanup in `finally` runs and metrics still record.

**Why not the alternative.** Running CPU work directly in the handler would freeze every other request; multiprocessing would complicate in-memory state.

---

## 15. Next.js, React and TypeScript

**What it is.** React is a UI library; Next.js a framework on top of it (routing, bundling); TypeScript adds static types.

**Why.** `web/` (Next ^16.4.6, React ^19.3.1, TS ^5.9.3) provides Ask, Metrics and Documents pages. Types in `lib/types.ts` mirror the API's event shapes so changes break the build, not production. CI runs `npm run typecheck` and `npm run build`.

**How it works.** The App Router maps `app/page.tsx`, `app/metrics/page.tsx` and `app/documents/page.tsx` to routes; client components keep shared state in a context (`AppContext`); `lib/api.ts` consumes the SSE stream with a `ReadableStream` reader.

**Why not the alternative.** The older static page (`frontend/index.html`, 134 lines, still served at `/ui`) has no types or components. *Streamlit/Gradio* would be faster to build but harder to customise (clickable citation markers, source panel).

---

## 16. pytest, pytest-asyncio, ruff, pre-commit, detect-secrets

**What it is.** Test runner, async test support, a very fast linter and formatter, git hook runner, and a secret scanner.

**Why.** 218 tests in 16 files; `asyncio_mode = "auto"`; `-m "not slow"` by default so real-model tests are opt-in. `ruff check .` gates CI. `scripts/check_staged.py` blocks staging `.env`.

**How it works.** pytest collects `test_*.py`, injects fixtures (`conftest.py` builds a tmp-path `Settings` and a fixture corpus indexed with the hashing embedder). Time and sleep are injected, so no test waits.

**Why not the alternative.** *unittest* is more verbose; *flake8 + black + isort* is three tools versus ruff.

---

## 17. uv, hatchling, GitHub Actions

**What it is.** `uv` is a fast Python package manager; `hatchling` is the build backend; GitHub Actions runs CI.

**Why.** `uv sync` gives reproducible installs (lock file stale at present, a known item). CI has three jobs: test (ruff, pytest, staged guard, advisory `pip-audit`, eval smoke gate), web (typecheck, build) and docker (image build).

**How it works.** The eval gate runs `python -m legal_rag.eval run --config G1_prompt_v2 ... --min hit@5=0.9 mrr=0.8 abstention_acc=0.9 citation_valid=0.9` on the fixture corpus; a regression fails the job and uploads the report as an artifact.

**Why not the alternative.** *pip + requirements.txt* has no lock or resolver speed; *Poetry* works but is slower.

---

## 18. Docker (multi-stage)

**What it is.** Container packaging. The `Dockerfile` has a build stage that installs the package into `/install`, and a runtime stage on `python:3.11-slim` that copies it, creates a non-root user (uid 1000), copies `data/`, and defines a `/health` healthcheck.

**Why.** Reproducible deployment; the compose file adds an `index-data` volume and an `ingest` profile for one-shot ingestion.

**How it works.** Layers are cached; the slim base and multi-stage copy keep the image small; running as non-root limits blast radius.

**Honest status.** I haven't run the build here. The Dockerfile also installs without the `embeddings` extra, so the default MiniLM model would fail inside the container until the extra is added.

---

## 19. LLM: llama-3.3-70b-versatile on Groq

**What it is.** A 70-billion-parameter open-weights chat model served on Groq's inference hardware through an OpenAI-compatible API.

**Why.** Fast token streaming and a free tier suit a demo. Defaults: temperature 0.1 (low variance for factual answers), `max_tokens` 2,048, 30 s timeout.

**How it works.** I send a system prompt containing the numbered passages inside `<passages>...</passages>` and the user's question; the model generates tokens autoregressively, which I forward as SSE `token` events.

**Why not the alternative.** *GPT/Claude* are strong but paid; a local model would need a GPU. The provider is behind `LLMClient`, so switching is a small adapter.

---

## Cheat sheet

| Technology | Category | One-liner | "Why not X?" |
|---|---|---|---|
| FastAPI / Uvicorn | Web framework / server | Async ASGI with validation and OpenAPI | Flask is WSGI: no native async streaming |
| SSE | Streaming protocol | One-way text frames over HTTP | WebSockets add bidirectional state I don't need |
| Pydantic v2 / pydantic-settings | Validation / config | Typed models and env-driven settings | dataclasses don't validate |
| httpx | HTTP client | Async client; easy to mock with `MockTransport` | requests is sync; SDKs hide behaviour |
| Retry with jitter | Resilience | Retry only transient errors, never mid-stream | Generic retry would duplicate streamed text |
| PyMuPDF | PDF parsing | Fast, page-aware text extraction | pdfminer slower; OCR is a different problem |
| all-MiniLM-L6-v2 | Dense embeddings | 384-d, 256-token, CPU-friendly | Larger models are slower; no bake-off run yet |
| Hashing embedder | Test embeddings | Deterministic bag-of-words, 512-d | A real model needs a download in CI |
| BM25 | Sparse retrieval | tf saturation, idf, length norm | TF-IDF lacks saturation and normalisation |
| RRF (k0=60) | Rank fusion | Sum of `1/(60+rank)` | Score mixing needs calibration |
| Qdrant | Vector DB | HNSW dense + IDF sparse + payload filters | pgvector ties to Postgres; Pinecone is hosted only |
| SQLite | Cache | Embedding cache keyed by model + hash | Redis needs a server |
| YAML manifest | Registry | Source of truth, edited as text | Re-dumping loses comments |
| asyncio + threads | Concurrency | Event loop for I/O, threads for CPU | CPU in handlers would block all requests |
| Next.js / React / TS | UI | Typed streaming interface | Streamlit is harder to customise |
| pytest / ruff / pre-commit | Quality | 218 tests, lint, secret guard | flake8+black+isort = three tools |
| uv / GitHub Actions | Tooling / CI | Fast installs, 3-job pipeline with eval gate | pip lacks a lock and resolver speed |
| Docker multi-stage | Packaging | Slim non-root image with healthcheck | Never built here; missing embeddings extra |
| llama-3.3-70b on Groq | LLM | Fast streaming, temp 0.1 | Paid APIs cost; local needs a GPU |
