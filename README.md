# Legal Assist RAG

A production-style Legal & Banking RAG service (Part 1 of the roadmap in `docs/roadmap/`): structure-aware
chunking, hybrid retrieval (dense + BM25 with reciprocal rank fusion), metadata filtering, grounded answers
with validated passage-level citations, abstention, SSE streaming, an eval harness, and abuse controls.

> Informational only; not legal advice.

## What exists today

| Area | Status |
|---|---|
| Config, LLM client (Groq) with retries/backoff, sessions with TTL + cap, SSE streaming, request IDs | done, tested |
| Typed manifest schema, `validate`, draft helper, hash-based idempotent ingestion CLI | done, tested |
| Parser (PyMuPDF + text), header/footer cleaning, chunkers S0–S3 with contextual prefix | done, tested |
| In-memory vector store (dense + BM25, filters, fingerprint guard, JSON persistence) | done, tested |
| Hybrid retrieval (RRF), per-doc cap, query understanding, old→new mapping lookup | done, tested |
| Versioned prompts, citation validation, status-aware context, confidence gate | done, tested |
| `/v1/ask` (SSE: token, citations, abstained, done, error), `/v1/stats`, `/v1/feedback`, admin delete | done, tested |
| Rate limiting, body-size limit, admin API key, prompt-injection delimiting | done, tested |
| Eval: golden schema, retrieval/abstention/citation metrics, runner, compare, CI regression gate | done, tested on a **synthetic fixture corpus only** |
| Dockerfile, compose, GitHub Actions CI | written; CI not yet run on GitHub |

## What is NOT done (needs your data / keys / services)

* **The ~100-document corpus** (Phase 1): `data/manifest.yaml` and the PDFs must be collected from official
  sources by you. `docs/corpus_plan.md` lists the targets.
* **The real golden set** (Phase 3): `eval/golden/smoke.jsonl` is 8 synthetic questions for CI. Dev/test splits
  must be authored and verified against real PDFs.
* **Measured results**: no B0→final ablation exists; `eval/experiment_log.md` is empty on purpose. No claim in
  this repo is backed by a real-corpus experiment yet.
* Not built: Qdrant adapter, cross-encoder reranker, LLM-judge faithfulness metrics, Langfuse tracing,
  `/metrics`, document-upload chat, frontend, `/v1/documents/upload`, `/stats` at the root path (it is `/v1/stats`).
* Embedding quality: the offline `hashing-512` embedder is for tests/CI. For real use install the extra
  (`uv sync --extra embeddings`) and set `EMBEDDING_MODEL=all-MiniLM-L6-v2` (or a model you choose).

## Quickstart

```bash
uv sync                      # (or: pip install -e . pytest pytest-asyncio ruff)
cp .env.example .env         # set LLM_API_KEY (Groq) and ADMIN_API_KEY
uv run pytest && uv run ruff check .
uv run python -m legal_rag.eval run --config G1_prompt_v2 --golden eval/golden/smoke.jsonl \
    --data-dir eval/fixtures --manifest eval/fixtures/manifest.yaml --embedder hashing-512
```

Index your own corpus and serve it:

```bash
uv run python -m legal_rag.ingestion draft      # draft manifest entries from data/raw
uv run python -m legal_rag.ingestion validate
uv run python -m legal_rag.ingestion reindex --all
uv run uvicorn legal_rag.api.main:create_app --factory
curl -N -X POST localhost:8000/v1/ask -H 'content-type: application/json' \
     -d '{"question":"What is the punishment for theft?"}'
```

Windows without `make`: use `.\make.ps1 <target>`; Docker: `docker compose up -d --build`.

## Layout

See `docs/roadmap/PART1_PRODUCTION_RAG_ROADMAP_v3.md` for the plan and `DESIGN.md` / `EVAL.md` for decisions and
eval methodology. Code lives in `src/legal_rag/{api,config,generation,ingestion,retrieval,eval,llm,observability}`.
