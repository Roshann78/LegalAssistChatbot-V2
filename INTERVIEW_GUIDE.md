# Legal Assist RAG: Interview Guide (Master Document)

Everything here is taken from this repository. File and line references are to the code as committed (`src/legal_rag/...`). Where a number comes from a screenshot of a real run rather than from code, it says so.

---

## 1. The 30-second elevator pitch (say this out loud)

> "I built a retrieval-augmented generation system for Indian legal and banking questions. You ask a question; it searches a corpus of statutes, judgments and RBI documents with hybrid dense-plus-keyword search, sends only the best passages to an LLM, and streams back an answer where every claim carries a numbered citation I can validate. If the evidence is missing it says so instead of guessing. It's a FastAPI service with server-sent-event streaming, a Next.js interface, a drop-in ingestion pipeline that classifies new PDFs and indexes only what changed, and an evaluation harness with a CI regression gate. About 6,500 lines of Python, 218 tests, 21 documents and roughly 27,000 indexed chunks on my real run."

---

## 2. System architecture

```
                         ┌─────────────────────────────┐
                         │  Next.js UI (web/, :3010)   │   static UI /ui (frontend/index.html)
                         │  fetch + ReadableStream SSE │
                         └───────────────┬─────────────┘
                                         │ POST /v1/ask   (SSE)    POST /v1/documents/upload
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ FastAPI app  (api/main.py: create_app + lifespan)                                         │
│  RequestIDMiddleware ► AbuseControlMiddleware (rate limit, body size) ► CORS              │
│  routes.py: ask / upload / delete session / stats / metrics / feedback / admin            │
│  sessions.py (TTL+LRU)   uploads.py (ephemeral per-file index)   metrics   tracing        │
└───────┬───────────────────────────────────────────┬───────────────────────────────────────┘
        │ prepare() in a thread                     │ stream() async
        ▼                                           ▼
┌──────────────────────────────┐          ┌───────────────────────────────┐
│ generation/pipeline.py       │          │ llm/resilience.py             │
│  1 understand (query.py)     │          │  retry+backoff, no retry after│
│  2 retrieve  (hybrid.py)     │          │  first token                  │
│  3 pin exact sections        │          │ llm/groq_client.py (httpx)    │
│  4 gate (abstain_threshold)  │─────────►│  Groq chat-completions, SSE   │
│  5 build_messages (prompts)  │          └───────────────────────────────┘
│  6 validate citations        │
└───────┬──────────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────┐
│ retrieval/                                                   │
│  Retriever: dense + sparse → RRF (k0=60) → per-doc cap 3     │
│  → optional rerank (lexical | cross-encoder) → top-k (5)     │
│  VectorStore interface: InMemoryVectorStore | QdrantVectorStore
│  Embedder: SentenceTransformer (all-MiniLM-L6-v2) | Hashing-512
└──────────────────────────────────────────────────────────────┘
        ▲ index built by
┌──────────────────────────────────────────────────────────────┐
│ ingestion/  (CLI: python -m legal_rag.ingestion sync)        │
│  data/raw/{legal,banking}/*.pdf ─► classify.py ─► manifest   │
│  parser.py (PyMuPDF) ─► normalize.py (JSON) ─► chunking.py   │
│  ─► embed via SQLite cache (cache.py) ─► store.upsert        │
│  fingerprint check, safe-order replace, hash-based skipping  │
└──────────────────────────────────────────────────────────────┘
   data/manifest.yaml · data/processed/{<doc>.json,index.json} · data/cache/embeddings.sqlite
```

Request path in one line: **question → understand → hybrid retrieve → (pin exact sections) → abstain gate → numbered prompt → streamed LLM tokens → citation validation → SSE `citations` + `done`.**

---

## 3. Every module, explained

### 3.1 `config/settings.py` (248 lines)
- One `Settings(BaseSettings)` class; ~70 fields, all overridable by environment variable or `.env`. `env_ignore_empty=True` so blank lines in `.env.example` do not override defaults.
- `cors_origins` uses `Annotated[list[str], NoDecode]` plus a validator, so `CORS_ORIGINS=http://a,http://b` and a JSON list both work. This fixed a real startup crash; there is a regression test.
- Secrets (`llm_api_key`, `admin_api_key`) are `SecretStr`, so model dumps print `**********`.
- Notable defaults: `chunk_size 800`, `chunk_overlap 150`, `chunk_max_tokens 256`, `retrieval_top_k 5`, `retrieval_candidates 30`, `retrieval_per_doc_cap 3`, `abstain_threshold 0.0` (`settings.py:68-70`), `domain_router False`, `auto_sync_on_startup False`.
- `get_settings()` is cached, so the app reads config once.

### 3.2 `ingestion/` (ingestion pipeline)
| File | Lines | What it does |
|---|---|---|
| `schema.py` | 105 | Pydantic `DocumentRecord` (the manifest row) with enums `DocType` (10 values), `Domain`, `Status` (in_force/amended/superseded/repealed), `SourceType`, `TextQuality`; `check_consistency()` rules (superseded needs `superseded_by`; judgments need a court). |
| `manifest.py` | 75 | `load_manifest`, `validate_manifest` (duplicates, missing files, hash drift, `superseded_by` links), `sha256_file`, `draft_entries`. |
| `manifest_edit.py` | 81 | Text-level edits: `append_entries`, `remove_entry`, `replace_file_hash`. The manifest is edited as text, not re-dumped, so comments and manual edits survive. |
| `classify.py` | 194 | Infers `doc_id`, `domain`, `doc_type`, `title`, `year` from folder, file name, PDF metadata and the first 3 pages (PyMuPDF). Ordered keyword rules (`_NAME_RULES`, `_TEXT_RULES`); defaults `statute` (legal) / `circular` (banking). |
| `parser.py` | 78 | `parse_pdf` (PyMuPDF per page, flags `low_text` pages under 50 chars), `parse_text`, `clean_pages` (removes repeated headers/footers, bare page numbers, de-hyphenates). `PARSER_VERSION = "pymupdf-1"`. |
| `normalize.py` | 108 | Turns pages into typed blocks (heading / list / paragraph) saved as `data/processed/<doc_id>.json`; re-parsed only if file hash or parser version changes (`ensure_normalized`, atomic temp-file rename). |
| `chunking.py` | 253 | Four strategies. **s0** fixed windows (800 chars, 150 overlap, snapping to paragraph/sentence/space); **s1** per `Section N` / `Article N` (statutes, constitution, rules, schemes); **s2** numbered-paragraph grouping for judgments; **s3** heading-based (circulars, directions, FAQs, reports). `is_toc_page` drops contents pages; `_enforce_max` shrinks any chunk over 256 tokens. Chunk ID = `{doc_id}:{strategy_version}:{i:05d}`. Embedded text gets a contextual prefix `Title > Section` (`chunking.py:251`). |
| `cache.py` | 48 | SQLite embedding cache keyed `(model, sha256(text))`; batches of 500 on read. |
| `indexer.py` | 199 | `Indexer.index_document`: skip if stored hash equals file hash; chunk, embed through the cache in batches of 64, `upsert` new chunks first, then `delete_except` stale ones ("safe order"). `config_fingerprint()` hashes embedder, dim, strategy, sizes, prefix flag, sparse method and parser version. |
| `sync.py` | 189 | `run_sync`: classify new files and append to manifest, refresh hash of modified files, validate, fingerprint check (refuse with "run sync --force"), index only new/changed, warn or prune deleted files, summary. `startup_sync` never raises. |
| `cli.py` | 238 | Commands: `sync [--prune --force]`, `validate`, `draft`, `list`, `status`, `stats`, `verify`, `parse`, `add`, `update`, `remove`, `reindex --all`, `report`, `lookup`; global `--dry-run`, `--yes`. |
| `report.py` | 88 | Writes `docs/ingestion_report.md` and a section lookup by metadata. |

### 3.3 `retrieval/`
- `types.py` (59): frozen dataclasses `Chunk`, `SearchResult`, `Filters`. `Filters.matches()` defaults to `status=("in_force","amended")`.
- `base.py` (120): abstract `Embedder` and `VectorStore` interfaces plus placeholder implementations for "not ready" states (so `/ready` can report honestly).
- `embedder.py` (77): `HashingEmbedder` (signed bag-of-words hashing, 512-d, L2-normalised; used in tests and CI) and `SentenceTransformerEmbedder` (lazy import; `encode(..., normalize_embeddings=True, batch_size=32)`). `tokenize` lowercases and removes a stopword list.
- `memory_store.py` (180): dense = dot product over normalised vectors (so cosine), full scan; sparse = BM25 with `k1=1.5, b=0.75` computed per query (`memory_store.py:121-143`); JSON persistence; `check_fingerprint` raises `FingerprintMismatchError`.
- `qdrant_store.py` (232): named vectors `dense` (cosine) and `sparse` (hashed term counts, server-side IDF modifier); deterministic UUID5 point IDs from chunk IDs; keyword payload indexes on `domain, doc_type, status, doc_id, chunk_id, section_or_article_no`; fingerprint stored in collection metadata.
- `hybrid.py` (106): `rrf_fuse` (`score = Σ weight/(60+rank)`, keeps the max dense score per chunk for the abstain gate), `cap_per_doc` (3), `Retriever.retrieve` (dense on `query + extra_terms`, sparse on the same), optional rerank, optional parent-child expansion (neighbouring chunks, 2,400-char cap, off by default).
- `rerank.py` (66): `LexicalReranker` (query-term coverage), `CrossEncoderReranker` (sentence-transformers, lazy), strategy `none|lexical|cross_encoder`.
- `query.py` (169): rule-based query understanding: follow-up rewrite, `Section/Article` detection, abbreviation table (IPC, BNS, CrPC, BNSS, IEA, BSA, RBI, KYC, RTI, PMJDY), `derive_aliases()` (title-derived acronyms such as `dpdp`, `pmla`, `it act`), verified old-to-new mapping (`MappingTable`), keyword `route_domain`.
- `factory.py` (45): `build_store` (memory loads `index.json`; qdrant local path or URL) and `build_retriever` (single place that wires settings into `Retriever`, shared by API and eval).

### 3.4 `generation/`
- `pipeline.py` (173): `RAGPipeline.prepare()` (`pipeline.py:87-150`) is the heart: understand, status filter (`in_force`/`amended` unless the question mentions IPC, CrPC or IEA), optional domain routing with a no-filter retry, pin up to 2 exact-section chunks, abstain gate on the best `dense_score`, build messages, currency notes. `stream()` yields tokens; `answer()` is the non-streaming path used by the eval runner.
- `prompts.py` (91): loads `config/prompts/answer_v{1,2}.txt`; `format_passage` renders `[i] Title | Section | p.N` and a `SUPERSEDED (current: ...)` tag; `_escape` neutralises `</passages>`; Jaccard dedupe (0.9), budget 12,000 chars, optional "edges" ordering.
- `citations.py` (59): `[n]` / `[n, m]` regex, `resolve_citations` returns valid citations and invalid numbers, `strip_invalid_citations` removes markers that point past the passage list.
- Prompt v2 rules: answer only from numbered passages; cite every claim; say so when evidence is missing; treat passages as untrusted data; flag superseded sources; be concise.

### 3.5 `llm/`
- `llm_client.py` (125): `LLMClient` ABC plus typed errors (`LLMAuthError`, `LLMRateLimitError(retry_after)`, `LLMTimeoutError`, `LLMProviderError(status_code)`).
- `groq_client.py` (159): direct `httpx` calls to `https://api.groq.com/openai/v1` (no SDK dependency), maps HTTP status to typed errors, parses the streaming SSE body.
- `resilience.py` (165): `ResilientLLMClient` wraps any client. Retries only timeouts, 429 and 5xx (`is_transient_error`, `resilience.py:25`); backoff `min(min_wait·2^attempt + jitter, max_wait)` honouring `Retry-After` (`calculate_backoff`, `resilience.py:43`); defaults 3 retries, 0.5 s to 8 s. **Streaming is never retried once a token has been yielded** (`resilience.py:123-136`).

### 3.6 `api/`
- `main.py` (326): `create_app()` factory with dependency injection (settings, llm, embedder, store, session store, pipeline) for tests. `lifespan` builds the shared `httpx.AsyncClient`, wraps the LLM, loads index/embedder (a failure marks components "not ready" instead of crashing), optionally runs `startup_sync` in a thread, builds the pipeline (with title aliases), session store plus a 60 s cleanup task, metrics, tracer, upload store. Global handlers return a uniform `ErrorResponse`. `/health` (liveness), `/ready` (503 until llm+embedder+store are ready; no network calls).
- `routes.py` (265): `POST /v1/ask` (SSE: `token`, `citations`, `abstained`, `done`, `error`), upload, delete upload, stats, metrics, feedback, two admin routes.
- `security.py` (88): `AbuseControlMiddleware` (pure ASGI): `Content-Length` check (413), sliding 60 s window per client IP (429 + `Retry-After`); `require_admin` uses `secrets.compare_digest`, returns 403 when no key is configured.
- `middleware.py` (79): request ID via `contextvars`, validated `X-Request-ID`.
- `sessions.py` (169): `InMemorySessionStore` with TTL (3600 s), cap 1000, LRU eviction, `asyncio.Lock`, injectable clock for instant tests.
- `uploads.py` (140): ephemeral per-file index: `.pdf`/`.txt` only, `%PDF-` magic check, 5 MB, 100 pages, 2,000 chunks, temp dir with a fixed file name, `s0` chunking, `secrets.token_urlsafe(24)` ID, TTL 1800 s, max 50 sessions.
- `streaming.py` (100): `format_sse`, SSE response headers.

### 3.7 `observability/`, `eval/`, `scripts/`, `web/`
- `observability/`: `Metrics` (bounded deques, p50/p95 per stage), `FileTracer` (opt-in JSONL, no prompt text), JSON logging with `SecretScrubber`.
- `eval/`: golden-set schema (`golden.py`), metrics (`hit_at_k, recall_at_k, mrr, ndcg_at_k, percentile, location_hit`), LLM judge (`faithfulness`, `relevance`, `citation_support`), experiment configs `B0_baseline ... G1_prompt_v2, C3/C5/C8` as dicts of settings overrides so the eval runs the production code path, `tune_abstain`, `bakeoff`, `freeze` (hashes the test split), `author.py` (LLM-assisted question drafting with human review).
- `scripts/`: `demo_server.py` (fake LLM, 4-document fixture), `load_test.py`, `check_staged.py` (pre-commit guard that blocks `.env`), `check_api_keys.py`.
- `web/` (1,253 lines TS/TSX): Next.js 16 / React 19 app: `app/page.tsx` (Ask), `app/metrics`, `app/documents`, `components/` (`Shell`, `Sidebar`, `UploadCard`, `AnswerPreview`, `AppContext`), `lib/api.ts` (SSE parsing over `fetch` + `ReadableStream`, not `EventSource`, because the request is a POST).

---

## 4. Tech stack

| Technology | Role | Why this over the alternative |
|---|---|---|
| Python 3.11 | Backend language | `requires-python = ">=3.11,<3.12"`; modern typing, `asyncio` TaskGroup-era stdlib |
| FastAPI + Uvicorn | HTTP/SSE API | Async streaming, Pydantic validation, OpenAPI docs for free (vs Flask: no native async streaming) |
| Pydantic v2 / pydantic-settings | Schemas, config | One typed source for manifest, requests, settings; `SecretStr` masks keys |
| PyMuPDF | PDF text extraction | Fast, page-aware, no system deps (vs pdfminer: slower; vs OCR stacks: out of scope) |
| sentence-transformers `all-MiniLM-L6-v2` | Dense embeddings (384-d) | Small, CPU-friendly, strong baseline; swappable behind `Embedder` |
| Hashing embedder (512-d) | Test/CI embedder | Deterministic, dependency-free, so CI needs no model download |
| In-memory store (JSON) | Default vector store | Zero infrastructure; honest about scale limits |
| Qdrant | Optional vector DB | Named dense+sparse vectors, payload filters, local or server mode |
| BM25 / hashed TF-IDF | Sparse retrieval | Exact identifiers ("Section 66A") that embeddings blur |
| Reciprocal Rank Fusion | Fusion | No score calibration between cosine and BM25 |
| Groq (llama-3.3-70b-versatile) | LLM | OpenAI-compatible, fast streaming, free tier; direct `httpx` keeps the dependency surface small |
| SQLite | Embedding cache | Stdlib, durable, keyed by model + content hash |
| YAML manifest | Document registry | Human-editable source of truth |
| Next.js 16 / React 19 / TypeScript | UI | Streaming UI with type-checked API client |
| pytest (+asyncio), ruff, pre-commit, detect-secrets | Quality | 218 tests, lint, secret scanning |
| GitHub Actions | CI | 3 jobs: test (with eval gate), web, docker |
| Docker / compose | Packaging | Written, never built here (see weaknesses) |

---

## 5. Storage and schemas

**Manifest row (`data/manifest.yaml`, `ingestion/schema.py:50`)**

| Field | Type | Notes |
|---|---|---|
| `doc_id` | str, `^[a-z0-9][a-z0-9_\-]*$` | stable slug |
| `title` | str | |
| `doc_type` | enum | statute, constitution, judgment, rule, circular, master_direction, scheme, faq, mapping_table, report |
| `domain` | enum | legal, banking |
| `path` | str | relative to `data/` (e.g. `raw/legal/x.pdf`) |
| `jurisdiction, authority_or_court, year, date, case_citation, act_name` | optional | |
| `status` | enum | in_force, amended, superseded, repealed |
| `effective_from, superseded_by, version, language, source_url, source_type` | optional | |
| `file_hash` | 64 hex | SHA-256 of the file |
| `page_count, text_quality` | int, enum | good, fair, poor, needs_ocr, unknown |

**Chunk (`retrieval/types.py:10`)**: `chunk_id`, `doc_id`, `text`, `embed_text` (prefix + text), `metadata` = manifest fields plus `section_or_article_no, heading_path[], page_start, page_end, paragraph_no, chunk_type, token_count, strategy_id, strategy_version, file_hash, config_fingerprint, ingest_run`.

**Files**
| Path | Content |
|---|---|
| `data/manifest.yaml` | document registry |
| `data/processed/<doc_id>.json` | normalized pages and blocks (parse cache) |
| `data/processed/index.json` | memory index: `{fingerprint, chunks:[{chunk_id, doc_id, text, embed_text, metadata, vector}]}` |
| `data/cache/embeddings.sqlite` | table `emb(model, key, vec)`, PK `(model, key)` |
| `data/processed/feedback.jsonl` | one JSON line per feedback |
| Qdrant collection `legal_rag` | vectors `dense` (cosine), `sparse` (IDF); payload indexes on 6 keyword fields |

---

## 6. API contracts

**`POST /v1/ask`** → `text/event-stream`
```json
// request
{"question": "What is the punishment for theft?", "domain": "legal",
 "conversation_id": null, "upload_id": null}
```
```
event: token      data: {"content": "Theft is ..."}
event: citations  data: {"citations":[{"index":1,"doc_id":"...","title":"...","section_or_article":"Section 4",
                                         "page":3,"status":"in_force","snippet":"..."}],
                         "invalid":[], "answer":"<sanitised text>", "notes":[]}
event: done       data: {"conversation_id":"...","request_id":"...","finish_reason":"stop",
                         "disclaimer":"Informational only; not legal advice.",
                         "latency_ms":1144,"understand_ms":0,"retrieve_ms":38,"rerank_ms":0,"generate_ms":1105}
event: abstained  data: {"conversation_id":"...","request_id":"...","message":"...","disclaimer":"..."}
event: error      data: {"code":"LLM_ERROR","request_id":"...","message":"The language model is unavailable."}
```
Validation: `question` 1 to 1,000 chars (422 beyond), `domain` is `legal|banking`, upload ID unknown or expired → 404, no index → 503.

**`POST /v1/documents/upload?filename=x.pdf`** (raw body) → `201`
```json
{"upload_id":"<token_urlsafe(24)>","filename":"x.pdf","pages":21,"chunks":108,"expires_in_seconds":1800}
```
Errors: 415 (type or magic), 413 (size/pages/chunks), 422 (empty or unreadable or no text), 503 (embedder not ready).

**`DELETE /v1/documents/session/{upload_id}`** → `204` or `404`.
**`GET /v1/stats`** → `{"documents":21,"chunks":26974}`.
**`GET /v1/metrics`** → `{"requests":3,"answered":3,"abstained":0,"errors":0,"error_rate":0.0,"abstention_rate":0.0,"approx_output_tokens_per_request":..,"latency_ms":{"retrieve_ms":{"p50":..,"p95":..,"n":3},...}}`.
**`POST /v1/feedback`** `{"request_id":"<8-64 chars>","rating":"up|down","comment":"<=1000"}` → `202 {"status":"recorded"}`.
**`GET /v1/admin/traces/{request_id}`**, **`DELETE /v1/admin/documents/{doc_id}`** need header `X-API-Key`; `403` if no admin key is configured, `401` if wrong; delete returns `{"doc_id":..., "chunks_removed":N}`.
**`GET /health`** → `{"status":"ok"}`. **`GET /ready`** → `{"status":"ready|not_ready","components":{"llm":"ready","embedder":"ready (all-MiniLM-L6-v2)","vector_store":"ready (memory, 26974 chunks)"}}`, `503` if any is not ready.
All errors: `{"code","message","request_id"}`; every response carries `X-Request-ID`.

---

## 7. Key design decisions

| Decision | Why | Alternative considered |
|---|---|---|
| Hybrid dense + sparse with RRF | Legal queries mix concepts and exact identifiers; RRF needs no score calibration | Weighted score sum (needs normalising cosine vs BM25) |
| Structure-aware chunking (s1/s2/s3) | A legal answer lives in one section or paragraph group; fixed windows cut provisos from their sections | Single recursive character splitter |
| Contextual prefix `Title > Section` in the embedded text | Disambiguates "Section 3" across Acts | Embed raw text only |
| Store the manifest as source of truth, index as derived | Index can always be rebuilt; edits are reviewable | Index as the only record |
| Config fingerprint on the index | Prevents silently mixing vectors from two embedders | Trust the operator |
| Safe-order replacement (upsert new, then delete stale) | A crash mid-update never leaves a document with no chunks | Delete then insert |
| Abstention gate before calling the LLM | No tokens spent, no hallucination on empty evidence | Always call the LLM and rely on the prompt |
| Validate citation numbers server-side | The model can cite `[9]` when only 5 passages exist; those markers are stripped | Trust the model |
| Never retry a stream after the first token | A retry would duplicate or corrupt text already shown | Retry everything |
| Pure ASGI abuse middleware | No dependency, runs before routing | slowapi / Redis rate limiter |
| Ephemeral in-memory upload index with unguessable ID | Per-user isolation without a database | Persist uploads in the main corpus |
| Eval configs are settings dicts | Experiments and production share one code path | A separate eval pipeline |
| Direct `httpx` to Groq | One less SDK; easy to mock with `httpx.MockTransport` | `groq` or `openai` SDK |
| Text-level manifest editing | Keeps comments and manual edits | `yaml.dump` the whole file |

---

## 8. "How to present this project" script (about 2 min 15 s)

**Opening (30 s).** "This is Legal Assist, a RAG system for Indian legal and banking documents. Users ask in plain English and get a streamed answer with numbered citations they can click to see the source passage, a note when a law has been superseded, and an explicit refusal when the corpus does not support an answer."

**Problem (15 s).** "General LLMs state legal conclusions with confidence and no source. Statutes and judgments are long and structured, so naive chunking breaks provisos away from their sections. I wanted answers that are grounded, checkable and honest about gaps."

**Architecture (45 s).** "Ingestion turns PDFs into structure-aware chunks, three of four strategies split on sections, numbered paragraphs or headings, and indexes them with a dense embedding plus BM25. At query time I rewrite and expand the question, retrieve with both, fuse with reciprocal rank fusion, cap per document, and pin exact section matches. A gate checks the best dense score; if it passes, a numbered-passage prompt goes to Groq and tokens stream to the browser over SSE. After generation I validate every citation number against the passages actually supplied."

**Hard parts (30 s).** "Three things. First, making ingestion safe and incremental: file hashes, an embedding cache, a config fingerprint that refuses mixed vectors, and a sync command that classifies new files automatically without overwriting my manual manifest edits. Second, streaming resilience: retry only before the first token. Third, finding that acronym queries like 'DPDP Act' failed because the Act never spells the acronym; I fixed it by deriving acronyms from manifest titles and embedding the expansion."

**Impact (15 s).** "218 tests, a CI gate on a fixture corpus, 21 documents and about 27,000 chunks in my real run, plus honest limits: I have not yet measured retrieval quality on the real corpus."

---

## 9. Weakness awareness (what I would say)

| Weakness | What I'd say |
|---|---|
| No real-corpus quality numbers (`eval/runs/` empty, `eval/experiment_log.md` blank) | "The harness, golden-set schema and CI gate exist and run on a fixture corpus; I have not frozen a real golden set yet. I won't quote Hit@5 on the real corpus because I haven't measured it." |
| `abstain_threshold` defaults to `0.0` (`settings.py:68`) | "Out of the box it only abstains on an empty result. There's a `tune-abstain` command to pick a threshold on a dev split; I haven't run it on real data." |
| In-memory store scans every vector, and BM25 statistics are rebuilt per query (`memory_store.py:126-129`) | "Fine at 27k chunks for a demo, it's O(N) per query. At scale I'd cache the BM25 index or move to Qdrant, which is already supported." |
| Retrieval latency spread in my 3-request sample (median 38 ms, p95 2.5 s) | "Three samples prove nothing. I'd profile before explaining it." |
| Duplicate documents in the real corpus (Companies Act, DPDP, PMLA appear twice; Constitution twice with different counts) | "Sync classifies by file; it doesn't deduplicate by content. A hash or title duplicate check is the next fix." |
| Process-local state: sessions, uploads, metrics, rate limit | "Single-process by design; for replicas these move to Redis." |
| `data/mappings/ipc_to_bns.json` is not in the repo, so the old-to-new mapping returns nothing | "The mechanism and tests exist; the verified mapping data is a to-do." |
| Docker and compose written but never built; the Dockerfile installs without the `embeddings` extra | "Unverified. First fix is adding the extra and running the build in CI." |
| No OCR in this repository; scanned pages are flagged (`low_text`) and uploads with no text are rejected | "OCR isn't in the code. If a report or slide says otherwise, be ready to show the commit." |
| No agentic layer in this repository (pipeline is linear) | "The implemented system is a single-pass RAG pipeline with limited query understanding. Do not describe an agent loop unless you can point to the code." |
| Citation check is structural, not semantic | "I check that `[n]` exists, not that passage n supports the sentence. The judge in `eval/judge.py` does that offline." |
| UI renders model markdown as literal `**` | "Cosmetic; render markdown in `AnswerPreview`." |

---

## 10. Numbers to memorise

| Item | Value | Where |
|---|---|---|
| Python in `src/legal_rag` | about 6,500 lines, 55 files | `wc -l` |
| Tests | 218 in 16 files, about 3,600 lines | `pytest --collect-only` |
| Web (TS/TSX/CSS) | about 1,250 lines | `web/` |
| Tracked project lines (py/ts/js/html/yml/toml/md) | 12,758 | git-tracked files |
| Real run (your machine, MiniLM) | 21 documents, 26,974 chunks | Documents page screenshot |
| Repo manifest (hashing embedder test) | 10 documents, 12,828 chunks | `docs/ingestion_report.md` |
| Chunk size / overlap / max tokens | 800 chars / 150 / 256 | `settings.py:72-74` |
| Observed chunk length | avg 75 to 169 tokens, max 200 | ingestion `stats` screenshot |
| Top-k / candidates / per-doc cap | 5 / 30 / 3 | `settings.py:63-67` |
| RRF constant | 60 | `hybrid.py:14` |
| BM25 | k1 1.5, b 0.75 | `memory_store.py:130` |
| Embedding dims | MiniLM 384, hashing 512 | model, `embedder.py:31` |
| Context budget / dedupe | 12,000 chars / Jaccard 0.9 | `settings.py` |
| LLM | llama-3.3-70b-versatile, temp 0.1, max 2,048 tokens, 30 s timeout | `settings.py:29-39` |
| Retry | 3 attempts, 0.5 to 8 s backoff, 0.5×min jitter | `resilience.py:43` |
| Rate limit / body / question | 60 req/min per IP / 1 MB / 1,000 chars | `settings.py` |
| Upload limits | 5 MB, 100 pages, 2,000 chunks, 30 min TTL, 50 sessions | `settings.py` |
| Sessions | TTL 3,600 s, max 1,000, history 4 turns | `settings.py` |
| API routes | 8 under `/v1` plus `/health`, `/ready` | `routes.py`, `main.py` |
| Strategy map | statute, constitution, rule, scheme → s1; judgment → s2; circular, master_direction, faq, mapping_table, report → s3 | `chunking.py:15-26` |
| Eval configs | 16 (`B0_baseline` ... `C5_edges`) | `eval/runner.py:70-143` |
| Smoke golden set | 8 questions, 6 categories | `eval/golden/smoke.jsonl` |
| CI gate | hit@5 ≥ 0.9, mrr ≥ 0.8, abstention_acc ≥ 0.9, citation_valid ≥ 0.9 | `.github/workflows/ci.yml` |
| Load test (fake LLM, 4-doc fixture) | 398 requests, 0 errors, about 8.8 req/s, retrieve p50 2.5 ms | `docs/load_test.md` |
| Metrics page (3 real requests) | total p50 1,144 ms, generation p50 1,105 ms, retrieval p50 38 ms | screenshot |
