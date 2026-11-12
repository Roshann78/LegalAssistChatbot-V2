# Resume and interview kit

Rule: use only numbers you have *measured on real data*. Everything in `[brackets]` is a placeholder you must
fill from `eval/experiment_log.md`; do not publish a bullet whose number you cannot reproduce.

## Resume bullets (fill the brackets; drop any you cannot support)
1. Built a production-style RAG system for Indian legal and banking documents (FastAPI, Qdrant, hybrid dense +
   BM25 retrieval with RRF, structure-aware chunking for statutes, judgments and circulars) over a
   [N]-document versioned corpus with idempotent, fingerprinted ingestion.
2. Designed an evaluation harness (golden set of [N] verified questions across [9] categories, frozen test
   split, retrieval/abstention/citation metrics, LLM-judge faithfulness) and drove retrieval from
   hit@5 [X] to [Y] through single-variable ablations, keeping only changes the numbers supported.
3. Implemented grounded generation with validated passage-level citations, legal-currency handling
   (superseded law to current equivalent) and an abstention gate tuned on dev data: [X]% of unanswerable
   questions refused at [Y]% false refusals.
4. Hardened the service (rate limits, size caps, admin auth, prompt-injection delimiting, isolated TTL upload
   sessions), with [N] tests, CI regression gate and request-level metrics/tracing; load-tested at
   [N] concurrent users with p95 [X] ms and [Y]% errors.

**Supported today (real, but small):** 205 tests, 94% line coverage; a CI eval gate on a synthetic fixture; an
API-layer load test of 398 requests with 0 errors on a fake LLM (see `docs/load_test.md`). None of the
quality numbers above exist yet.

## Interview stories (write each in your own words; tag = what your code does today)
1. **Idempotent ingestion:** hash-based skip, deterministic chunk IDs, write-new-then-delete-old, fingerprint
   guard refusing mixed embeddings. (Tested; failure injected after upsert.)
2. **A bug the tests caught:** pick one from your history (e.g. the superseded-document filter, the
   number-of-passages vs citation-index mismatch after deduplication).
3. **An experiment that failed:** [fill from your ablation; none exist yet].
4. **Trade-off: abstention before generation** (cheap, deterministic, but dense-score-only) versus a
   post-generation faithfulness check (costly; kept eval-only).
5. **Trade-off: in-memory store default vs Qdrant** behind one interface.
6. **Security:** untrusted-data delimiting and escaping, upload isolation, why this is not a proof against
   injection and what experiment would measure it.
7. **Why a thin custom judge** and how you would audit it (hand-check ~20 items).
8. **Legal currency:** only verified old-to-new mappings are used; unverified entries are ignored.

## Likely interview questions with outlines (ground each in your results)
Retrieval and chunking
1. *Why section-aware chunking?* Provisos and sub-clauses must stay with their section; tested boundaries; gain
   over S0 is **[unmeasured]**.
2. *Why hybrid?* Exact identifiers ("Section 66A") need lexical match; RRF avoids score calibration; gain **[unmeasured]**.
3. *What does the contextual prefix do?* Embed-only context (title + heading path); raw text is displayed/cited.
4. *How do you stop one long judgment dominating?* Per-document cap (and optional neighbour expansion).
5. *Why a reranker and what did it cost?* Interface + lexical stand-in + cross-encoder adapter; latency/gain **[unmeasured]**.
6. *How do you handle "Section 302 IPC"?* Reference detection, abbreviation expansion, verified mapping, status filter.
7. *How does the embedding model get chosen?* `eval bakeoff` on identical chunking; **[no result yet]**.
8. *What if the embedding model changes?* Fingerprint changes; store refuses to mix; `reindex --all`.

Evaluation
9. *How is the golden set built and protected from leakage?* LLM drafts, human verifies against PDFs; frozen
   test split with hash lock; test run once; dev-only tuning.
10. *Metrics?* hit@k, recall@k, MRR, nDCG, abstention accuracy, citation validity/support, latency, tokens/cost.
11. *How do you trust the LLM judge?* Pinned model, temperature 0, versioned prompts, manual audit **[not done]**.
12. *How do you prevent regressions?* CI smoke eval with thresholds on a fixture; full eval manual.
13. *What are the weaknesses of your evaluation?* Synthetic fixture only so far; small N; gold at document level.

Generation and trust
14. *How are citations validated?* Markers parsed, checked against provided passages, invalid ones stripped and logged.
15. *When does the system refuse?* Best dense score below a threshold tuned on dev unanswerables (`tune-abstain`).
16. *How do you handle superseded law?* Status filter by default, included when the question names an old Act;
    citation status and currency notes.
17. *Faithfulness guard: online or offline?* Offline until numbers justify one judge call per sentence.

Systems and security
18. *Concurrency model?* Heavy components loaded once at startup; blocking work in threads; SSE with disconnect handling.
19. *What happens when the LLM fails mid-stream?* Safe `error` event, no provider detail; retries with backoff before that.
20. *Prompt injection?* Delimited data block, escaped closing tag, rules; effect on a real LLM **[untested]**.
21. *Upload safety?* Type/magic-byte/size/page/chunk caps, fixed temp filename, isolation by unguessable ID, TTL.
22. *Rate limiting limits?* Per-process, per-IP; use a shared store to scale out.
23. *What would you monitor in production?* `/v1/metrics` (p95 per stage, error and abstention rate), traces, feedback.

Wrap-up
24. *What would you do next?* Real corpus + golden set, run B0 to final, audit the judge, OCR, multilingual, Part 2 agents.
25. *What are you least sure about?* Everything marked unmeasured; say so plainly.
