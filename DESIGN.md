# Design decisions

Short ADRs. "Evidence" is only filled where a real experiment exists; none exist yet on a real corpus.

| # | Decision | Why | Status / evidence |
|---|---|---|---|
| 1 | In-memory vector store with JSON persistence behind a `VectorStore` interface | Zero-dependency local run and fast tests; Qdrant adapter implemented (`VECTOR_STORE_BACKEND=qdrant`); sparse = hashed term counts + server-side IDF | Tested in-process only; no benchmark |
| 2 | PyMuPDF parser | Light, fast, page-aware; OCR for scanned PDFs is **not implemented** (low-text pages are flagged only) | Docling comparison **not run** |
| 3 | Chunking per doc type (S1 statutes, S2 judgments, S3 headings) with S0 baseline | Keeps sections/provisos and numbered paragraphs intact; unit-tested boundaries | Gain over S0 **unmeasured** on real data |
| 4 | Contextual prefix is embed-only | Better matching without polluting displayed/generated text | Unmeasured |
| 5 | Hybrid = dense + BM25 fused by RRF | Exact identifiers ("Section 66A") need lexical match | Unmeasured; B0/B3 configs exist in `eval/runner.py` |
| 6 | Index records a config fingerprint and refuses mixing | Prevents silent embedding/chunking mismatch | Tested |
| 7 | Retrieval returns structured `SearchResult`s; pipeline is stateless with separate stages | Keeps Part 2 agent tools easy; history is injected | Tested |
| 8 | Abstain when best dense score < threshold, before calling the LLM | Cheap, deterministic; threshold must be tuned on dev unanswerables | Default 0.0 (off); fixture uses 0.2 |
| 9 | Citations are `[n]` markers validated against provided passages; invalid ones are dropped and logged | Invalid citations never reach clients silently | Tested |
| 10 | Retrieved text is delimited as untrusted data; closing tags are escaped | Basic prompt-injection hardening | Tested at prompt level; **no live-model injection pass rate measured** |
| 11 | Sessions: in-memory with TTL + LRU cap | Simple; Redis later | Tested |
| 12 | Per-IP rate limit and Content-Length cap in ASGI middleware | No extra dependency | Single-process only; chunked bodies not size-checked |
| 13 | Admin endpoints disabled unless `ADMIN_API_KEY` set; constant-time compare | Safe default | Tested |

## Rejected or deferred
Scraping, fine-tuning, Graph RAG, agents (Part 2), cloud deployment (Part 3), cross-encoder reranking and
LLM-judge metrics (need models/keys and a real golden set), Langfuse tracing, upload chat, frontend.

## Phase 4 (retrieval upgrades): built, not yet measured
Every upgrade is a `Settings` switch and an experiment config in `eval/runner.py::CONFIGS`, differing from
its predecessor in one knob (enforced by a test): chunking (B1), contextual prefix (B1b), parent-child (B1c),
hybrid (B3), domain router (B4), rerank (B5), query-understanding components (B6a-d). Tools:
`eval bakeoff --models ...` (retrieval-only embedding comparison) and `eval ablation` (B0 -> final table from
saved runs). **No decision is made yet**: nothing has been run on a real corpus, so nothing is kept or dropped.
* Reranker: a dependency-free lexical reranker (test/CI stand-in) and a lazy cross-encoder adapter that needs
  `RERANK_MODEL` (no model is assumed). No hosted reranker.
* `R_final` is a **provisional** composition, not chosen from data; redefine it from your ablation.
* Multilingual (4.7): out of scope for now; no Hindi set exists. Revisit with a 10-question Hindi dev set.
* Parent-child here = the hit plus its neighbouring chunks (same document, +/-1); overlapping parents are not merged.
