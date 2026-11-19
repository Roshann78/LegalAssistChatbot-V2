# Data Folder Guide

How the `data/` folder is organised, where each kind of file goes, and how the system handles it. There are two separate paths: the **permanent corpus** (files you place in `data/raw/` and ingest) and **user uploads** (files chatted with through the UI or API, never saved to disk).

## 1. Folder layout

```
data/
├── manifest.yaml            # source of truth: one record per corpus document (you create it)
├── raw/                     # ORIGINAL files you add (never modified by the system)
│   ├── legal/               # all legal PDFs go here, no subfolders  e.g. bns_2023.pdf
│   └── banking/             # all banking PDFs go here, no subfolders e.g. rbi_kyc_md_2016.pdf
├── processed/               # GENERATED; git-ignored (only .gitkeep is tracked)
│   ├── <doc_id>.json        # normalized text per document (parse cache)
│   ├── index.json           # in-memory index snapshot (default backend)
│   ├── qdrant/              # only if VECTOR_STORE_BACKEND=qdrant in local mode
│   └── feedback.jsonl       # thumbs up/down submitted from the UI
├── mappings/
│   └── ipc_to_bns.json      # optional IPC→BNS section mapping used for query understanding
└── cache/
    └── embeddings.sqlite    # GENERATED; embedding cache keyed by (model, text hash)
```

`doc_type` is set per document in `manifest.yaml` (not by folder). Allowed values: `statute`, `constitution`, `judgment`, `rule`, `circular`, `master_direction`, `scheme`, `faq`, `mapping_table`, `report`.

Configurable paths (in `.env` / `Settings`): `DATA_DIR`, `MANIFEST_PATH`, `PROCESSED_DIR`, `INDEX_PATH`, `MAPPINGS_PATH`, `FEEDBACK_PATH`, `EMBEDDING_CACHE_PATH`, `QDRANT_PATH`.

## 2. Where do I put files?

### A. Permanent knowledge base (laws, judgments, RBI circulars…)
1. Put the PDF in `data/raw/legal/` or `data/raw/banking/` (no further subfolders).
   - The file name (without extension) becomes the `doc_id`: a lowercase slug (`a-z`, `0-9`, `_`, `-`), e.g. `rbi_kyc_md_2016.pdf`.
2. Run `legal-rag-ingest draft` to print draft manifest entries for files that are not yet registered. Paste them into `data/manifest.yaml` and fill in `doc_type` (left blank, since folders no longer encode it), `title`, `year`, `status`, `source_url`, etc.
3. Run `legal-rag-ingest validate` (checks duplicates, missing files, hash mismatches, `superseded_by` links, field consistency).
4. Run `legal-rag-ingest add <doc_id>` (or `reindex --all`) to parse, chunk, embed and index.

Manifest `path` is **relative to `data/`**, e.g. `path: raw/legal/bns_2023.pdf`.

### B. Files a user wants to chat with (temporary)
Use the **paperclip / upload** control in the UI, or the API:

```
POST /v1/documents/upload?filename=contract.pdf     (raw file bytes as the body)
```
Nothing is written into `data/`. See section 4.

## 3. How corpus data is handled (ingestion pipeline)

| Step | What happens | Output |
|---|---|---|
| 1. Manifest | Each record is validated against a typed schema (domain, doc_type, status, year, hash…). | `manifest.yaml` is the source of truth; the index is derived from it |
| 2. Hash | SHA-256 of the raw file is computed. | Detects changes; unchanged files are skipped |
| 3. Parse | PDF/text is extracted page by page and cleaned (headers, footers, whitespace). Pages with very little text are flagged (`low_text`) — scanned PDFs are not OCR'd. | `data/processed/<doc_id>.json` (reused while hash and parser version are unchanged) |
| 4. Chunk | Text is split by the chosen strategy (default `auto`; size 800 chars, overlap 150) and each chunk gets contextual metadata (title, section, domain, status). | Chunks |
| 5. Embed | Chunks are embedded with the configured model; vectors are cached by text hash so re-runs are fast. | `data/cache/embeddings.sqlite` |
| 6. Index | Chunks are written to the vector store (in-memory by default, optional Qdrant) plus a BM25 (keyword) index for hybrid search. New chunks are written first, then stale chunks for that document are removed. | `data/processed/index.json` or Qdrant |

Key behaviours:
- **Raw files are never edited.** Everything under `processed/` and `cache/` can be deleted and rebuilt.
- **Config fingerprint:** the index records the embedding model and chunk settings. Starting with different settings raises a mismatch error until you run `reindex --all`.
- **Updating a document:** replace the file in `raw/`, run `legal-rag-ingest update <doc_id>` (hash change is detected and only that document is re-indexed).
- **Removing:** `legal-rag-ingest remove <doc_id>` (or admin API `DELETE /v1/admin/documents/{doc_id}`, requires `ADMIN_API_KEY`).
- **Legal currency:** `status` (`in_force`, `amended`, `superseded`, `repealed`) and `superseded_by` travel with every chunk, so answers can warn when a source is out of date.
- **Other useful commands:** `list`, `status`, `stats`, `verify`, `parse --all`, `lookup <doc_id> <section>`, `report`, and `--dry-run` on any command.

## 4. How uploaded files are handled

Uploads are **ephemeral and isolated**:

1. **Validation:** only `.pdf` and `.txt`; PDFs must start with `%PDF-`; max **5 MB** (`UPLOAD_MAX_BYTES`), max **100 pages** (`UPLOAD_MAX_PAGES`), max **2000 chunks** (`UPLOAD_MAX_CHUNKS`). The filename is reduced to a safe basename and never used as a disk path.
2. **Temporary parse:** the bytes are written to a temp directory under a fixed name, parsed, and the directory is deleted immediately (even on failure).
3. **In-memory index:** chunks are embedded and stored in an in-memory vector store only. Nothing goes to `data/raw`, `data/processed` or the cache.
4. **Upload ID:** the API returns an unguessable `upload_id`. Questions that include it (`upload_id` in `/v1/ask`) search only that document, not the corpus, so one user's file never leaks into another's answers.
5. **Expiry:** sessions expire after 30 minutes (`UPLOAD_TTL_SECONDS`); at most 50 live sessions (`UPLOAD_MAX_SESSIONS`), oldest evicted first. `DELETE /v1/documents/session/{upload_id}` removes one immediately. A server restart clears all uploads.
6. **Limits:** scanned/image-only PDFs are rejected ("No extractable text"); there is no OCR.

## 5. Other files the system writes
- `data/processed/feedback.jsonl` — one JSON line per feedback submission.
- Optional request traces (JSONL) if `TRACE_PATH` is set.
- Eval runs write to a temporary directory, never into `data/processed`.

## 6. Git and privacy
- `data/processed/*` and `data/cache/` are git-ignored. `data/raw/` PDFs and `manifest.yaml` are normal files; commit them only if you have the right to redistribute those documents (most are public-domain government texts, but check each source).
- Never put secrets in `data/`. API keys go in `.env` (git-ignored).

## 7. Quick reference

| I want to… | Do this |
|---|---|
| Add a law/circular to the knowledge base | Put PDF in `data/raw/legal/` or `data/raw/banking/`, `draft` → edit manifest → `validate` → `add <doc_id>` |
| Chat with my own PDF once | Upload in the UI (stored in memory for 30 min) |
| Fix a document after replacing the file | `legal-rag-ingest update <doc_id>` |
| Rebuild everything | `legal-rag-ingest reindex --all` |
| Free disk space | Delete `data/processed/*` and `data/cache/`, then reindex |
| Remove a document | `legal-rag-ingest remove <doc_id>` |
