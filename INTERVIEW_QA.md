# Interview Q&A (78 questions, answers ready to speak)

All answers are first person and match the code in this repository. Where a limitation is real, the answer says so.

---

## A. Project overview

**A1. What is this project?**
> I built Legal Assist, a retrieval-augmented generation system for Indian legal and banking questions. It searches a corpus of statutes, judgments and RBI material, passes the best passages to an LLM, and streams back an answer with numbered citations. If the corpus doesn't support an answer, the system is designed to say so rather than guess.

**A2. Why did you build it? What problem does it solve?**
> Plain LLMs answer legal questions confidently with no source, and Indian statutes and judgments are long and structured, so people struggle to find the exact provision. I wanted answers that are grounded, checkable through citations, and honest about gaps and about superseded law.

**A3. Who is it for?**
> Non-lawyers who want a first, sourced pointer to the right provision. The prompt says "plain language," the interface shows the source passage for every citation, and every response carries "Informational only; not legal advice."

**A4. What is the end-to-end flow of a question?**
> The browser POSTs to `/v1/ask`. I rewrite and expand the question, retrieve with dense and BM25 search, fuse the lists with reciprocal rank fusion, cap results per document, and pin exact section matches. A gate checks the best dense score; if it passes I build a numbered-passage prompt and stream Groq's tokens back over server-sent events. At the end I validate the citation numbers against the passages I actually supplied and send a `citations` event and a `done` event.

**A5. What's the scale of the project?**
> About 6,500 lines of Python across 55 files in `src/legal_rag`, around 1,250 lines of TypeScript in the Next.js app, and 218 tests in 16 files. On my real run it indexed 21 documents into about 27,000 chunks.

**A6. What was your role?**
> I designed and built the pipeline end to end: ingestion, retrieval, generation, the API, the evaluation harness, the CI, and the interface. I can walk through any module in detail.

**A7. What are you most proud of?**
> The ingestion design. It's incremental and safe: hashes decide what to re-process, a config fingerprint refuses to mix vectors from two embedders, new chunks are written before stale ones are deleted, and a `sync` command classifies new PDFs and updates the manifest without overwriting my manual edits.

---

## B. Architecture and design

**B1. Why RAG instead of fine-tuning?**
> Law changes, and I need to show sources. With RAG I update the corpus by dropping in a PDF, I can cite the exact passage, and I can mark a source superseded. Fine-tuning bakes knowledge into weights, can't cite, and goes stale.

**B2. Why hybrid retrieval?**
> Legal questions mix concepts, like "what is data fiduciary," and exact identifiers, like "Section 66A." Dense embeddings handle paraphrase but blur identifiers; BM25 matches exact terms but misses synonyms. Running both and fusing them covers both.

**B3. Why reciprocal rank fusion instead of a weighted sum?**
> Cosine similarity and BM25 scores live on different scales. RRF uses only rank positions, `weight / (60 + rank)`, so I don't have to calibrate scores. The constant 60 is the standard value from the original paper and it's `k0` in `hybrid.py`.

**B4. Why structure-aware chunking?**
> Fixed windows cut a proviso away from its section and mix two sections in one chunk. I split statutes on `Section N` and `Article N`, judgments on numbered paragraphs grouped up to 800 characters, and circulars and reports on headings. A recursive character splitter would be simpler but loses the structure that legal citations depend on.

**B5. What is the contextual prefix?**
> Each chunk's embedded text starts with `Title > Section`, for example "The Digital Personal Data Protection Act, 2023 > Section 9". It disambiguates "Section 3" across different Acts. The displayed text stays the original chunk text; only the embedded text carries the prefix.

**B6. How does the abstention gate work?**
> After retrieval I take the maximum dense score across the results. If there are no results, or the best score is below `abstain_threshold`, I skip the LLM and return an explanation of what was searched. One honest caveat: the default threshold is 0.0, so by default it only abstains on an empty result. There's a `tune-abstain` command to pick a value on a dev split.

**B7. How do you handle laws that have been replaced?**
> Each document has a `status`: in_force, amended, superseded or repealed, plus an optional `superseded_by`. By default retrieval filters to in-force and amended sources, unless the question mentions IPC, CrPC or the Evidence Act. If a superseded passage is used, it's tagged in the prompt and returned as a currency note.

**B8. Why an interface for embedders and vector stores?**
> It lets me swap implementations without touching the pipeline: a hashing embedder for tests, MiniLM for real use; an in-memory store by default, Qdrant optionally. The eval runner and the API share the same `build_retriever` factory, so experiments test the production code.

**B9. Why is the pipeline linear rather than an agent?**
> For a fixed corpus, most legal questions are answered by one or two passages, and a linear pipeline is faster, cheaper and easier to test. An agent loop adds latency and failure modes. I kept the stages as separate callables so a loop could wrap them later.

**B10. How does the upload feature work without touching the corpus?**
> An uploaded PDF or text file is validated, parsed in a temp directory, chunked and embedded into its own in-memory store, and registered under an unguessable `secrets.token_urlsafe(24)` ID with a 30-minute TTL. A question with that `upload_id` searches only that store, so one user's file never appears in another user's answer.

**B11. What does `create_app` do and why is it a factory?**
> It builds the FastAPI app and lets me inject settings, an LLM client, an embedder, a vector store, a session store or a whole pipeline. Tests pass fakes, so none of them need a network or a model download.

---

## C. Security

**C1. How do you defend against prompt injection in retrieved passages?**
> The prompt tells the model the passages are untrusted data and to never follow instructions in them. In code, `_escape` neutralises `<passages>` and `</passages>` inside passage text so a document can't close the data block. I also keep five injection fixtures in `eval/injection/`, such as "ignore previous" and "fake citation." I'd say this reduces risk; I haven't proven resistance against a real model.

**C2. How do you stop the model from citing sources that don't exist?**
> I parse every `[n]` marker, check `1 <= n <= number of passages`, report invalid numbers, and strip the invalid markers from the final answer. The streamed tokens may briefly contain a bad marker, so the `citations` event carries a sanitised `answer` that the client uses to replace the displayed text.

**C3. How are secrets handled?**
> `llm_api_key` and `admin_api_key` are `SecretStr`, so dumps show asterisks. The JSON logger runs a `SecretScrubber`, and a pre-commit hook plus `scripts/check_staged.py` block committing `.env`. CI verifies that guard with a self-check.

**C4. How is the admin API protected?**
> Admin routes need an `X-API-Key` header, compared with `secrets.compare_digest` to avoid timing leaks. If no admin key is configured, the routes return 403, so they're off by default.

**C5. How do you protect against abuse?**
> A pure-ASGI middleware rejects oversized bodies from the `Content-Length` header with 413, and applies a sliding 60-second window per client IP, 60 requests per minute by default, returning 429 with `Retry-After`. Uploads are bounded separately: 5 MB, 100 pages, 2,000 chunks, `%PDF-` magic check, and the file name is never used as a path.

**C6. How do you handle CORS?**
> `CORS_ORIGINS` is configurable, and I never set `allow_credentials` when the wildcard is used. I also fixed a startup crash where a comma-separated list wasn't parsed, with a regression test.

**C7. What about privacy of uploads?**
> They live only in memory under an unguessable ID, expire after 30 minutes, and can be deleted immediately with a DELETE call. Tracing is opt-in and stores metadata but not prompt text. The honest caveat is that uploaded text and questions do go to an external LLM provider, Groq.

**C8. What error information do you expose?**
> Provider failures become a generic `LLM_ERROR` event with a request ID, with no provider details. Unhandled exceptions return the same `ErrorResponse` shape. Logs carry the request ID so I can trace a failure.

---

## D. Performance

**D1. Why stream the answer?**
> LLM generation is the slowest stage, around 1.1 seconds median in my three-request sample. Streaming with SSE shows the first token almost immediately instead of waiting for the full answer, and it lets me keep the connection open for the citation and done events.

**D2. Why SSE and not WebSockets?**
> The data flows one way, server to client, so SSE is simpler and works over plain HTTP. In the browser I read it with `fetch` and a `ReadableStream` reader rather than `EventSource`, because `EventSource` can't send a POST body.

**D3. Where are the bottlenecks?**
> The LLM call dominates latency. In the in-memory store, dense search scans every vector and BM25 statistics are rebuilt on every query in `sparse_search`, so retrieval is O(N) per question. That's fine at 27k chunks for a demo; for more I'd cache the BM25 index or use Qdrant.

**D4. How do you avoid blocking the event loop?**
> Retrieval and embedding are CPU-bound, so `prepare()` runs in `asyncio.to_thread`, as do uploads' parsing and feedback file writes. The LLM stream is natively async through `httpx`.

**D5. How does the embedding cache help?**
> It's a SQLite table keyed by model and the SHA-256 of the text. Re-indexing a modified document only embeds chunks whose text changed; the rest are cache hits. Switching models doesn't mix entries because the model name is part of the key.

**D6. How would you reduce LLM cost and latency?**
> Cap `max_tokens`, keep the context budget at 12,000 characters, de-duplicate near-identical passages with a Jaccard threshold of 0.9, and abstain before calling the model when evidence is missing. I also track approximate output tokens per request in the metrics.

**D7. What latency numbers do you have?**
> In a real three-request session the Metrics page showed a total median of about 1.14 seconds, generation about 1.1 seconds and retrieval about 38 ms, but retrieval p95 was 2.5 seconds. Three samples aren't a benchmark and I'd profile before explaining the spread. Separately, a load test with a fake LLM and a four-document fixture did 398 requests with zero errors at about 8.8 requests per second.

---

## E. Data

**E1. What is the manifest and why does it exist?**
> `data/manifest.yaml` is the source of truth: one record per document with ID, title, type, domain, status, path and file hash. The index is derived from it, so I can rebuild the index at any time and review metadata changes in git.

**E2. How does the drop-in `sync` work?**
> I drop a PDF into `data/raw/legal` or `data/raw/banking`. `sync` classifies it from the folder, file name, PDF metadata and first pages, appends an entry to the manifest as raw text, validates, then parses, chunks, embeds and indexes only that file. Unchanged files are skipped by hash; modified files are re-indexed and their stale chunks removed; deleted files only warn unless I pass `--prune`.

**E3. Why edit the manifest as text?**
> If I re-dumped the YAML, I'd lose comments and could overwrite fields I edited by hand, like marking an Act superseded or adding a source URL. Appending raw text preserves everything; the only automatic change to an existing entry is refreshing its file hash when the file changes.

**E4. What's the config fingerprint?**
> A 16-character SHA-256 over everything that changes index contents: embedder name and dimension, chunk strategy and its version, sizes, prefix flag, sparse method and parser version. The store records it, and a mismatch raises `FingerprintMismatchError`, telling me to run `sync --force`. It prevents mixing 512-dimension hashing vectors with 384-dimension MiniLM vectors.

**E5. Why write new chunks before deleting old ones?**
> If the process dies between the two steps, the document still has its old chunks instead of none. `index_document` upserts the new chunks, then calls `delete_except` with the new chunk IDs.

**E6. How do chunk IDs work?**
> `{doc_id}:{strategy_version}:{index:05d}`. They're deterministic, so re-indexing overwrites; in Qdrant I derive a UUID5 point ID from the chunk ID so upserts are idempotent.

**E7. Memory store or Qdrant, how do they differ?**
> The memory store keeps chunks and vectors in dicts and persists a JSON file: dense search is a dot product over normalised vectors, sparse is BM25 computed in Python. Qdrant keeps named vectors, `dense` with cosine distance and `sparse` with hashed term counts and a server-side IDF modifier, plus keyword payload indexes for filters. Both implement the same interface.

**E8. What did you learn from your real ingestion run?**
> 21 documents gave 26,974 chunks, averaging 75 to 169 tokens with a maximum of 200. It also showed duplicates: the Companies Act, DPDP Act and PMLA each appear twice with identical counts, and the Constitution twice with different counts. Sync classifies by file, not content, so a duplicate check is the next fix.

**E9. How do you handle scanned PDFs?**
> There's no OCR in this code. `parse_pdf` flags pages with under 50 characters as `low_text`, the sync summary lists them, and an upload with no extractable text is rejected.

---

## F. Language and framework specifics (Python, FastAPI, TypeScript)

**F1. How do you do dependency injection in FastAPI here?**
> Mostly through the `create_app` factory and `app.state`, not `Depends`. Shared objects, such as the LLM client, embedder, store and session store, are built once in the lifespan and read from `request.app.state`. `Depends` is used where it fits naturally, for `require_admin`.

**F2. What does the lifespan do?**
> It creates one shared `httpx.AsyncClient`, wraps the LLM client in the resilience layer, loads the index and embedder (a failure flips readiness to not-ready instead of crashing), optionally runs the startup sync in a thread, builds the pipeline, starts a background session-cleanup task, and cleans everything up on shutdown.

**F3. How do you handle async and sync code together?**
> The pipeline's `prepare()` is synchronous because retrieval is CPU work, so I call it through `asyncio.to_thread`. Streaming is an `async for` over the LLM's async generator. The session store uses an `asyncio.Lock`, while the upload store and metrics use `threading.Lock` because they're touched from threads.

**F4. How do you use Pydantic?**
> For the manifest schema with enums and validators, request and response models, and settings with `pydantic-settings`. I used `Annotated[list[str], NoDecode]` plus a validator so list-valued environment variables accept comma-separated or JSON input.

**F5. How do you request IDs propagate?**
> A `ContextVar` set by `RequestIDMiddleware` (`api/middleware.py`) holds the ID. The middleware validates an incoming `X-Request-ID`, generates one otherwise, and adds it to every response, error body and log line.

**F6. How does the frontend consume the stream?**
> `web/lib/api.ts` POSTs with `fetch`, gets `res.body.getReader()`, decodes chunks, splits on blank lines, and parses `event:` and `data:` lines into typed events. The Ask page updates the answer as tokens arrive, then replaces the text with the sanitised answer from the `citations` event.

**F7. How do you test code that depends on time?**
> I inject the clock. `InMemorySessionStore` and `UploadStore` take a `time_fn`, and the resilience client takes sleep functions, so TTL expiry and backoff are tested instantly, with no real waiting.

**F8. How do you mock the LLM?**
> The Groq client uses `httpx`, so tests use `httpx.MockTransport` to return canned status codes and SSE bodies. There's also an `ExtractiveFakeLLM` for pipeline and eval smoke tests.

---

## G. DevOps and deployment

**G1. What does CI do?**
> Three GitHub Actions jobs. `test` runs `uv sync`, ruff, pytest, a staged-file guard self-check, an advisory `pip-audit`, and an eval smoke run on the fixture corpus with thresholds `hit@5 ≥ 0.9, mrr ≥ 0.8, abstention_acc ≥ 0.9, citation_valid ≥ 0.9`. `web` runs `npm ci`, the TypeScript type check and the build. `docker` builds the image after tests pass.

**G2. Is it containerised?**
> There's a multi-stage `python:3.11-slim` Dockerfile that runs as a non-root user with a `/health` healthcheck, and a compose file with a volume for the index and an `ingest` profile. I have to be upfront: I haven't been able to run the build here, and the Dockerfile doesn't install the `embeddings` extra yet, so a container with the default MiniLM model would fail until I add it.

**G3. How would you deploy it for free?**
> API on Hugging Face Spaces (Docker, CPU, enough RAM for the model and index) with the index baked into the image at build time, the UI on Vercel pointing at the API URL, and a Groq free-tier key. The index must be built at image build because the free tier has no persistent disk.

**G4. What monitoring exists?**
> `/health` and `/ready` probes, a `/v1/metrics` endpoint with p50 and p95 per stage, JSON logs with request IDs and stage timings on every ask, and optional JSONL traces that I can look up by request ID through an admin route. They're process-local; a production setup would export to Prometheus or similar.

**G5. How do you manage configuration across environments?**
> Everything is an environment variable with a safe default, validated by Pydantic at start-up. `.env.example` lists the knobs; `AUTO_SYNC_ON_STARTUP=true` is set there for local development and defaults to false in code.

**G6. What happens on start-up if the index is missing?**
> The app still starts. The embedder and store are placeholders marked not-ready, `/ready` returns 503, and `/v1/ask` returns 503 with "no index loaded." If `AUTO_SYNC_ON_STARTUP` is true, it builds the index from `data/raw` first.

---

## H. Improvements

**H1. What would you add next?**
> First, a real golden set and measured baselines, because I haven't measured retrieval quality on the real corpus. Then tune the abstention threshold, de-duplicate documents during sync, cache the BM25 index, and fix the Docker build.

**H2. What technical debt do you have?**
> The default abstention threshold of 0.0, per-query BM25 rebuilds in the memory store, process-local sessions and rate limits, a Dockerfile missing the embeddings extra, a stale `uv.lock`, and the old-to-new law mapping file not being populated.

**H3. How would you add OCR?**
> I already detect low-text pages during parsing. I'd route those pages through an OCR step before normalisation, keep the OCR confidence in the page stats, and add a `text_quality` of `needs_ocr` handling plus a test with a scanned fixture.

**H4. How would you add multilingual support?**
> Add a language selector in the request, translate the query or use a multilingual embedder, and translate the answer while preserving citation markers, then verify that the marker set is identical before and after. I'd want bilingual legal review for terminology.

**H5. How would you check that a citation actually supports a claim?**
> Today I only check that the number exists. The offline judge in `eval/judge.py` already scores citation support with an LLM; I'd run a lightweight version online for high-risk answers, or sample it, trading latency for assurance.

**H6. How would you scale the corpus to 100k+ documents?**
> Switch to Qdrant server mode, add payload indexes for filters (already created for six fields), shard by domain, batch ingestion in workers, and put the embedding cache on shared storage. I'd also add a cross-encoder reranker over the top 30 to keep precision.

**H7. What would your v2 be?**
> A measured baseline, deduplicated corpus, an agentic mode for multi-step questions behind a bounded loop, user accounts for document ownership, and Redis-backed sessions so I can run several replicas.

---

## I. Behavioural

**I1. Tell me about the hardest bug.**
> A user asked "what id dpdp act" and the system answered that the passages had nothing about a DPDP Act, citing the Motor Vehicles Act. The Act's text never spells the acronym, so neither BM25 nor MiniLM matched it. I derived acronyms from manifest titles, expanded them in query understanding, and made the dense search embed the expansion too. I'm candid that I fixed the mechanism and added tests, but haven't measured its effect on a real golden set.

**I2. Describe a trade-off you made.**
> Zero-infrastructure in-memory storage versus scalability. I accepted O(N) scans and Python BM25 to keep local setup trivial, and kept a Qdrant adapter behind the same interface so I can scale later without rewriting the pipeline.

**I3. Tell me about a mistake you caught.**
> While testing the sync command I built an index with the offline hashing embedder, 512 dimensions, in the same data folder my real backend reads, which uses MiniLM at 384. Queries were then compared against vectors from a different embedder, so retrieval was meaningless. The fingerprint exists to refuse mixing embedders at write time, but this showed that a stale index can still be loaded and served silently. My fix list includes making a fingerprint mismatch show as not-ready, and I now keep test indexes in temp directories.

**I4. How did you handle uncertainty or missing information?**
> I separated what I can show from what I can't. For example, I don't quote retrieval-quality numbers because `eval/runs` is empty, and I mark the real-run figures, 21 documents and 26,974 chunks, as coming from my own run's screenshots.

**I5. How do you test your own work?**
> Tests first for the risky parts: fingerprint mismatches, safe-order replacement, streaming retry rules, upload limits, and sync behaviours like preserving manual manifest edits. Then lint and the CI eval gate. I also run the real thing by hand and compare against the screenshots.

**I6. How did you handle disagreement or feedback?**
> When the interface looked broken I took the screenshot as the spec, reproduced it, and fixed it rather than defending the first version. Similarly, feedback that manual manifest editing was painful led directly to the drop-in sync workflow.

---

## J. Theory

**J1. What is cosine similarity and why normalise vectors?**
> It's the cosine of the angle between two vectors. If both are L2-normalised, cosine equals the dot product, so I can compute it as a plain dot product, which is what the memory store does. MiniLM is encoded with `normalize_embeddings=True` and my hashing embedder normalises explicitly.

**J2. Explain BM25.**
> A term-frequency scoring function with inverse document frequency and length normalisation: `idf · tf·(k1+1) / (tf + k1·(1 − b + b·len/avglen))`. I use `k1 = 1.5` and `b = 0.75`. `idf = log(1 + (N − df + 0.5)/(df + 0.5))`.

**J3. Why does RRF work?**
> Each list contributes `1/(60 + rank)`, so an item ranked high in either list scores well, and an item high in both scores best. It ignores raw scores, so it's robust to different scales. The constant dampens the advantage of the very top ranks.

**J4. What's the complexity of your retrieval?**
> In-memory dense search is O(N·d) for N chunks of dimension d; BM25 as written is O(total tokens) because statistics are rebuilt per call, plus O(N log N) sorting. Qdrant replaces the dense scan with an approximate-nearest-neighbour index, giving sub-linear search.

**J5. CAP theorem, where does your system sit?**
> It's a single-node system, so the question mostly doesn't apply yet. Sessions and uploads are process-local, so with replicas I'd trade consistency for availability unless I add shared state. The index is read-mostly and rebuilt offline, so eventual consistency across replicas is acceptable if they all load the same fingerprint.

**J6. What are precision@k, recall@k, MRR and nDCG?**
> Hit@k is whether any gold document appears in the top k; recall@k is the share of gold documents found; MRR is the mean of `1/rank` of the first gold hit; nDCG@k rewards putting gold items higher with a logarithmic discount. They're implemented in `eval/metrics.py`.

**J7. What's the difference between hallucination and unsupported citation?**
> Hallucination is stating something not in the sources. An unsupported citation is attaching a real passage number to a sentence the passage doesn't support. My online check catches non-existent numbers; checking support needs a judge, which I run offline.

**J8. Why exponential backoff with jitter?**
> To avoid synchronised retries from many clients hammering a rate-limited service. I use `min_wait · 2^attempt` plus random jitter capped at `max_wait`, and honour a `Retry-After` header when the provider sends one. I only retry timeouts, 429 and 5xx.

**J9. What is idempotency and where do you rely on it?**
> Repeating an operation gives the same result. Ingestion relies on it: deterministic chunk IDs, content-hash skipping and upserts mean re-running `sync` on the same files changes nothing.
