# Whiteboard and Visual Playbook

## Quick reference: what to draw when

| Interviewer says | Draw | Time |
|---|---|---|
| "Give me the architecture" | Diagram 1: system architecture | 2 min |
| "Walk me through a request" | Diagram 2: request lifecycle (sequence) | 2 min |
| "How does data get in?" | Diagram 3: ingestion and sync flow | 2 min |
| "What does your data look like?" | Diagram 4: manifest, chunk and index schema | 1.5 min |
| "How would this scale?" | Diagram 5: scaling path | 2.5 min |
| "What if X fails?" | Diagram 6: failure handling | 2 min |
| "Why hybrid / why this store?" | Diagram 7: trade-off comparison | 1.5 min |
| "How do you chunk legal text?" | Diagram 8: chunking decision tree | 1.5 min |
| "Explain rank fusion" | Diagram 9: RRF worked example | 1 min |

Rule: always draw boxes left to right (user → API → pipeline → stores), name each box with the file that implements it, and finish by marking the one weak spot you'd fix first.

---

## Diagram 1: System architecture

```
   ┌──────────┐   SSE    ┌───────────────────────────┐        ┌──────────────┐
   │ Browser  │◄────────►│ FastAPI (api/main.py)     │───────►│ Groq LLM     │
   │ Next.js  │  POST    │  middleware → routes      │ stream │ (external)   │
   └──────────┘          └───────────┬───────────────┘        └──────────────┘
                                     │ prepare() in thread
                          ┌──────────▼───────────┐
                          │ RAGPipeline          │
                          │ understand→retrieve  │
                          │ →gate→prompt→validate│
                          └──────────┬───────────┘
                    ┌────────────────┴───────────────┐
              ┌─────▼──────┐                   ┌─────▼──────┐
              │ Embedder   │                   │ VectorStore│
              │ MiniLM     │                   │ memory|Qdrant
              └────────────┘                   └─────▲──────┘
                                                     │ upsert
   data/raw/*.pdf ─► sync ─► classify ─► chunk ─► embed (SQLite cache)
```

**Drawing order**
1. Draw the Browser box on the left and the Groq box on the far right.
2. Draw the FastAPI box in the middle; write "middleware: request ID, rate limit, CORS" inside it.
3. Draw the arrow Browser ↔ FastAPI labelled "POST /v1/ask, SSE tokens back".
4. Draw RAGPipeline under FastAPI with its five stages.
5. Draw Embedder and VectorStore under the pipeline; write "memory | Qdrant" on the store.
6. Draw the ingestion path along the bottom feeding the store.

**Narration.** "A question goes from the browser to FastAPI. After rate-limit and size checks, the pipeline rewrites the question, embeds it, searches the store, decides whether the evidence is good enough, and only then calls Groq. Tokens stream back over SSE. Offline, `sync` turns PDFs into chunks and vectors in the same store."

---

## Diagram 2: Request lifecycle

```
Browser        API/middleware      Pipeline            Store        Groq
  │ POST /v1/ask    │                 │                  │            │
  ├────────────────►│ id, rate, size  │                  │            │
  │                 ├────────────────►│ understand       │            │
  │                 │                 ├─ dense+BM25 ────►│            │
  │                 │                 │◄── results ──────┤            │
  │                 │                 │ RRF→cap→pin      │            │
  │                 │                 │ best dense < thr?│            │
  │◄─ abstained ────┤◄── abstain ─────┤  (yes: stop)     │            │
  │                 │                 │ build prompt [1..n]           │
  │                 ├──────────────── stream ───────────────────────►│
  │◄── token ×N ────┤◄──────────── tokens ──────────────────────────┤
  │                 │ validate [n] against passages                  │
  │◄─ citations ────┤  (strip invalid, add currency notes)           │
  │◄─ done ─────────┤  (latency, request_id, disclaimer)             │
```

**Drawing order**
1. Draw five vertical lifelines: Browser, API, Pipeline, Store, Groq.
2. Draw the request arrow and label the middleware checks.
3. Draw understand → retrieve → results.
4. Draw the diamond-style decision "best dense score < threshold?" with the early `abstained` branch.
5. Draw the Groq stream and the token arrows back to the browser.
6. Draw the last two events: `citations` then `done`.

**Narration.** "Everything before the LLM is deterministic and cheap: rules, an embedding, two searches, a gate. That's why abstention costs nothing in tokens. After streaming, I validate citation numbers server-side and send the cleaned answer in the `citations` event."

---

## Diagram 3: Ingestion and sync flow

```
 data/raw/{legal,banking}/*.pdf|txt        (drop files here)
            │
            ▼
   discover (recursive) ──► known? (manifest path)
            │ no                       │ yes
            ▼                          ▼
   classify.py                 hash changed?
   domain/type/title/year        │yes        │no
            │                    ▼           ▼
   append to manifest      refresh hash     SKIP
   (text, keeps edits)           │
            └─────────┬──────────┘
                      ▼
              validate manifest
                      ▼
          fingerprint == index?  ──no──► "run sync --force"
                      │yes
                      ▼
   parse (PyMuPDF) → normalize JSON → chunk (s0..s3) → embed (SQLite cache)
                      ▼
           upsert NEW chunks → delete STALE chunks → persist index
                      ▼
        summary: added / updated / skipped / removed / low-text / chunks
```

**Drawing order.** (1) the folder at the top; (2) the "known?" fork; (3) the classify branch on the left; (4) the hash branch on the right; (5) validation and the fingerprint diamond; (6) the four processing boxes in a row; (7) the safe-order write; (8) the summary.

**Narration.** "I made ingestion idempotent and incremental. A file is processed only if it's new or its hash changed. A config fingerprint refuses to mix embedders. New chunks are written before stale ones are deleted, so a crash never leaves a document empty."

---

## Diagram 4: Data model

```
 manifest.yaml (source of truth)           data/processed/<doc_id>.json
 ┌───────────────────────────┐             ┌────────────────────────┐
 │ doc_id (slug, PK)         │             │ pages[ blocks[type,text]]
 │ title, doc_type, domain   │──parse────► │ stats{low_text_pages}  │
 │ path, file_hash (sha256)  │             │ source_file_hash       │
 │ status, superseded_by     │             │ parser_version         │
 │ year, source_url, ...     │             └───────────┬────────────┘
 └─────────────┬─────────────┘                         │ chunk
               │ metadata copied                       ▼
               │                      ┌──────────────────────────────────┐
               └─────────────────────►│ Chunk                            │
                                      │ chunk_id = doc:strategy:00042    │
                                      │ text, embed_text ("Title > Sec") │
                                      │ metadata: section_or_article_no, │
                                      │  heading_path, page_start/end,   │
                                      │  chunk_type, status, domain,     │
                                      │  file_hash, config_fingerprint   │
                                      │ vector (384-d)                   │
                                      └───────────────┬──────────────────┘
                                                      │
                  index.json (memory)  |  Qdrant: vectors {dense cosine, sparse IDF}
                                       |  payload indexes: domain, doc_type, status,
                                       |  doc_id, chunk_id, section_or_article_no
 data/cache/embeddings.sqlite: emb(model, key=sha256(text), vec)  PK (model,key)
```

**Drawing order.** (1) manifest box; (2) arrow to the normalized JSON; (3) arrow to the Chunk box; (4) list the six metadata fields that drive filters and citations; (5) the two storage backends; (6) the SQLite cache off to the side.

**Narration.** "The manifest is the source of truth and the index is derived. Every chunk carries the document's metadata plus its section and page, which is what makes filtering by status or domain and citing the exact page possible."

---

## Diagram 5: Scaling path

```
 TODAY                       10x                         100x / 1000x
 ┌───────────────┐   ┌──────────────────────┐   ┌────────────────────────────────┐
 │ 1 process     │   │ LB → API ×3          │   │ Gateway → API fleet (autoscale)│
 │ memory index  │──►│ Qdrant (1 node, HNSW)│──►│ Qdrant cluster (shard by domain)│
 │ in-proc state │   │ Redis: sessions,     │   │ Redis cluster                  │
 │ MiniLM in-proc│   │  rate limit          │   │ Embedding service (GPU/ONNX)   │
 └───────────────┘   │ worker: ingestion    │   │ LLM gateway: queue, fallback,  │
   bottleneck:       └──────────────────────┘   │  response cache                │
   O(N) scan, BM25      bottleneck: LLM          └────────────────────────────────┘
   rebuilt per query,   latency + rate limits        bottleneck: cost + provider limits
   state in memory
```

**Drawing order.** Three columns left to right. Under each, write its bottleneck in one line. Add the fix arrow between columns.

**Narration.** "Today's limits are in the in-memory store and process-local state. First fix is Qdrant plus Redis, which my interfaces already allow. After that the bottleneck is the LLM: queueing, caching, fallback providers, and cost."

---

## Diagram 6: Failure handling

```
 LLM call ──► transient? (timeout, 429, 5xx)
                 │yes                         │no (401/403/4xx)
                 ▼                            ▼
          before first token?            raise typed error
            │yes        │no                    │
            ▼           ▼                      ▼
   backoff+jitter    abort, no retry     SSE error LLM_ERROR
   (≤3 tries,        (partial text        (generic message +
    Retry-After)      already shown)        request_id)

 Startup ──► embedder/index load fails? ──► placeholders, /ready 503, /v1/ask 503
 Request ──► rate > 60/min/IP ──► 429 + Retry-After
          ──► body too large ──► 413
 Upload  ──► bad type/magic/size/pages/chunks ──► 415/413/422 (temp dir removed)
 Index   ──► fingerprint mismatch at sync ──► refuse, "run sync --force"
```

**Drawing order.** (1) the LLM decision tree first, since it's the interesting one; (2) add the four one-line guards beneath.

**Narration.** "Retries are conditional: only transient errors, only before the first token. After that I can't un-send text, so I surface an error event instead. Startup failures never crash the server; they flip readiness so an orchestrator stops routing traffic."

---

## Diagram 7: Trade-off comparison

```
 RETRIEVAL                      STORE
 ┌────────┬────────┬────────┐   ┌──────────────┬──────────────┬───────────────┐
 │        │ Dense  │ BM25   │   │              │ In-memory    │ Qdrant        │
 ├────────┼────────┼────────┤   ├──────────────┼──────────────┼───────────────┤
 │Synonyms│  good  │  poor  │   │ Setup        │ none         │ service/local │
 │Exact id│  poor  │  good  │   │ Search       │ O(N) scan    │ HNSW ANN      │
 │Acronym │  poor* │  poor* │   │ Filters      │ Python loop  │ payload index │
 │Cost    │  embed │  cheap │   │ Scale        │ ~10^4-10^5   │ 10^6+         │
 └────────┴────────┴────────┘   │ Persistence  │ JSON file    │ on disk/server│
   Hybrid = both lists + RRF    └──────────────┴──────────────┴───────────────┘
   *fixed by title-derived acronym expansion
```

**Drawing order.** Draw the left table and fill the three rows; then the right table. Circle "Hybrid = both + RRF".

**Narration.** "Neither retrieval mode wins on its own, so I fuse them. For storage I defaulted to zero setup and kept the interface so Qdrant is a config change when scale or filter speed matters."

---

## Diagram 8: Chunking decision tree

```
 doc_type
   ├─ statute | constitution | rule | scheme ─► s1: split at "Section N" / "Article N"
   ├─ judgment ───────────────────────────────► s2: group numbered paragraphs (≤ 800 chars)
   ├─ circular | master_direction | faq |
   │  mapping_table | report ─────────────────► s3: split at headings
   └─ anything else ──────────────────────────► s0: 800-char windows, 150 overlap
 every strategy: drop contents pages → cap 256 tokens → add prefix "Title > Section"
```

**Drawing order.** Root node `doc_type`, four branches, then the shared post-processing line.

**Narration.** "Chunk boundaries follow how lawyers cite: sections for statutes, numbered paragraphs for judgments, headings for directions and reports. If no markers are found, a strategy falls back to plain windows."

---

## Diagram 9: RRF worked example

```
 dense list:  A(1)  B(2)  C(3)  D(4)
 sparse list: C(1)  A(2)  E(3)  B(4)

 score(A) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
 score(C) = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
 score(B) = 1/(60+2) + 1/(60+4) = 0.01613 + 0.01563 = 0.03176
 score(D) = 1/(60+4)            = 0.01563
 score(E) = 1/(60+3)            = 0.01587

 fused: A, C, B, E, D  → cap 3 per doc → top-5
```

**Drawing order.** Two ranked columns, then the formula, then the computed scores, then the fused order.

**Narration.** "A chunk in both lists beats one that's top in a single list, and I never compare a cosine score to a BM25 score."
