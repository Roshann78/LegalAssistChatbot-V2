# Demo script (run on the real corpus; targets, not yet verified results)

Start: `uvicorn legal_rag.api.main:create_app --factory`, open `/ui/`. Set `TRACE_PATH=data/processed/traces.jsonl`
to show traces. Everything below is the *intended* behavior; confirm each on your real corpus before presenting
and replace any that fail with an honest "known limitation" beat.

| # | Question | Expected behavior | Point out |
|---|---|---|---|
| 1 | "What does Section 66A of the IT Act say?" (exact lookup) | Pinned chunk for that section first; cited with section and page | Section pinning; citation list with status |
| 2 | "What is the punishment for theft under the IPC?" (old vs new) | Answer cites the current law and shows a currency note that IPC is superseded and what replaced it (needs verified entries in `data/mappings/ipc_to_bns.json`) | `notes` in the citations event; status tag |
| 3 | "What documents are needed for KYC and how often is it updated?" (multi-source) | Citations from at least two documents | Several numbered citations; domain routing |
| 4 | "Recipe for chocolate cake" (out of scope) | No LLM call; abstention message saying what was searched and the closest documents | `abstained` event; zero generation latency |
| 5 | Upload a PDF containing "ignore previous instructions", then ask a question about it | Answer stays grounded in the document and ignores the instruction **(unverified on a real LLM: run `eval/injection/` cases)** | Delimited data block; upload isolation and delete button |
| 6 | A follow-up: "What about its punishment?" | Resolved using conversation history | Conversation ID; trace shows the standalone query |

For each, open the trace (`GET /v1/admin/traces/{request_id}` with `X-API-Key`) to show retrieved chunks,
scores and per-stage timings, and `/v1/metrics` for p50/p95 and abstention rate.

Honest limits to mention: no OCR for scanned PDFs, no multilingual support, judge metrics are unaudited,
results come from one small corpus.
