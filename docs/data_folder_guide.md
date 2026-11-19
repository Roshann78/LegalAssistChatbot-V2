# Data Folder Guide

How the `data/` folder is organised, where files go, and how the system handles them. There are two separate paths: the **permanent corpus** (PDFs you drop into `data/raw/`, ingested automatically by `sync`) and **user uploads** (files chatted with through the UI or API, never saved to disk).

## 1. Folder layout

```
data/
├── manifest.yaml            # one record per corpus document (auto-appended by sync; you may edit)
├── raw/                     # ORIGINAL files you add (never modified by the system)
│   ├── legal/               # legal PDFs, no subfolders needed
│   ├── banking/             # banking PDFs, no subfolders needed
│   └── (or drop files directly here; subfolders of any depth are scanned)
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

Configurable paths (`.env` / `Settings`): `DATA_DIR`, `MANIFEST_PATH`, `PROCESSED_DIR`, `INDEX_PATH`, `MAPPINGS_PATH`, `FEEDBACK_PATH`, `EMBEDDING_CACHE_PATH`, `QDRANT_PATH`.

## 2. Adding documents: the drop-in workflow

1. **Drop** a `.pdf` or `.txt` into `data/raw/legal/` or `data/raw/banking/` (or directly into `data/raw/`).
2. **Sync**, either way:
   - CLI: `legal-rag-ingest sync` (or `python -m legal_rag.ingestion sync`)
   - Automatic: set `AUTO_SYNC_ON_STARTUP=true` in `.env` and start (or restart) the API. The sync runs before the first request is served.

That is all. No manual manifest editing is needed.

### What sync does

| Situation | Behaviour |
|---|---|
| **New file** | Classified, appended to `data/manifest.yaml`, validated, then parsed, chunked, embedded and indexed (only that file). |
| **Unchanged file** | Skipped entirely (SHA-256 matches the index; parse cache and embedding cache are reused). |
| **Modified file** | Re-parsed, re-chunked, re-embedded (embedding cache reused for unchanged text), new chunks upserted, stale chunks of that `doc_id` removed. Only the entry's machine-owned `file_hash` is refreshed in the manifest. |
| **Deleted file** | Clear warning; nothing is removed unless you pass `--prune`, which drops its chunks, parse cache and manifest entry. |
| **Changed config** (embedding model, chunk size, parser) | Refuses with a message telling you to run `legal-rag-ingest sync --force`. |

Flags: `--prune` (apply deletions), `--force` (rebuild everything), `--dry-run` (global flag; report only, writes nothing).

It ends with a summary: `added`, `updated`, `skipped`, `removed`, `low_text_warnings` and the total indexed chunks. Low-text warnings flag pages with little or no extractable text (scanned pages; there is no OCR).

### Auto-classification

New files get their metadata inferred (`src/legal_rag/ingestion/classify.py`):

| Field | How it is chosen |
|---|---|
| `doc_id` | Lowercase slug of the file name (`a-z`, `0-9`, `_`, `-`), made unique with `-2`, `-3`… |
| `domain` | The folder if the file is under `legal/` or `banking/`; otherwise keywords in the file name and first pages |
| `doc_type` | Keywords in the file name, then the first page (`statute`, `constitution`, `judgment`, `rule`, `circular`, `master_direction`, `scheme`, `faq`, `mapping_table`, `report`); default `statute` (legal) or `circular` (banking) |
| `title` | PDF metadata title, else an "The … Act, YYYY" line on the first page, else the first meaningful line, else the file name |
| `year` | First 4-digit year in the file name, title or first page |
| defaults | `status: in_force`, `jurisdiction: India`, `language: en`, file hash, page count, text quality; judgments get a court (`Supreme Court of India`, `High Court` or `Court (to be verified)`) |

These are best guesses. **Review new entries**, each is preceded by a `# auto-added by sync` comment. Edit anything in the manifest (for example `status: superseded` plus `superseded_by`, `source_url`, a better `title`). Sync **never overwrites** fields of existing entries; it only appends new ones, refreshes `file_hash` for a modified file, and removes an entry on `--prune`. Comments and ordering are preserved because the manifest is edited as text.

## 3. How corpus data is handled (what happens per document)

| Step | What happens | Output |
|---|---|---|
| 1. Manifest | Each record is validated against a typed schema (domain, doc_type, status, year, hash…). | `manifest.yaml` is the source of truth; the index is derived from it |
| 2. Hash | SHA-256 of the raw file is computed. | Detects changes; unchanged files are skipped |
| 3. Parse | PDF/text is extracted page by page and cleaned (headers, footers, whitespace). Low-text pages are flagged. | `data/processed/<doc_id>.json` (reused while hash and parser version are unchanged) |
| 4. Chunk | Text is split by the chosen strategy (default `auto`; size 800 chars, overlap 150) and each chunk gets contextual metadata. | Chunks |
| 5. Embed | Chunks are embedded with the configured model; vectors are cached by text hash. | `data/cache/embeddings.sqlite` |
| 6. Index | Chunks go to the vector store (in-memory by default, optional Qdrant) plus a BM25 index for hybrid search. New chunks are written first, then stale ones removed. | `data/processed/index.json` or Qdrant |

Other behaviours:
- **Raw files are never edited.** Everything under `processed/` and `cache/` can be deleted and rebuilt.
- **Config fingerprint:** the index records the embedding model and chunk settings; a mismatch is refused until `sync --force` (or `reindex --all`).
- **Legal currency:** `status` (`in_force`, `amended`, `superseded`, `repealed`) and `superseded_by` travel with every chunk so answers can warn about out-of-date sources.
- **Manual commands still exist:** `validate`, `draft` (preview classification without writing), `list`, `status`, `stats`, `verify`, `parse`, `add`, `update`, `remove`, `reindex --all`, `lookup`, `report`.

## 4. Files a user wants to chat with (temporary uploads)

Use the paperclip / upload control in the UI, or `POST /v1/documents/upload?filename=contract.pdf` with the file bytes as the body. Nothing is written into `data/`.

1. **Validation:** only `.pdf` and `.txt`; PDFs must start with `%PDF-`; max **5 MB** (`UPLOAD_MAX_BYTES`), **100 pages** (`UPLOAD_MAX_PAGES`), **2000 chunks** (`UPLOAD_MAX_CHUNKS`). The filename is reduced to a safe basename and never used as a path.
2. **Temporary parse:** bytes go to a temp directory under a fixed name, are parsed, and the directory is deleted immediately.
3. **In-memory index:** chunks are embedded into an in-memory store only.
4. **Upload ID:** an unguessable `upload_id` is returned; questions that include it search only that document, so files never leak between users.
5. **Expiry:** 30 minutes (`UPLOAD_TTL_SECONDS`), at most 50 live sessions, oldest evicted first. `DELETE /v1/documents/session/{upload_id}` removes one immediately. A restart clears all uploads.
6. Scanned/image-only PDFs are rejected (no OCR).

## 5. Other files the system writes
- `data/processed/feedback.jsonl`: one JSON line per feedback submission.
- Optional request traces (JSONL) if `TRACE_PATH` is set.
- Eval runs use a temporary directory, never `data/processed`.

## 6. Git and privacy
- `data/processed/*` and `data/cache/` are git-ignored. `data/raw/` PDFs and `manifest.yaml` are normal files; commit them only if you may redistribute those documents.
- Never put secrets in `data/`. API keys go in `.env` (git-ignored).

## 7. Quick reference

| I want to… | Do this |
|---|---|
| Add a law/circular | Drop the PDF in `data/raw/legal/` or `data/raw/banking/`, run `legal-rag-ingest sync` (or restart the API with `AUTO_SYNC_ON_STARTUP=true`) |
| See what sync would do | `legal-rag-ingest --dry-run sync` |
| Replace a document | Overwrite the file in `data/raw/`, run `sync` |
| Remove a document | Delete the file, run `sync --prune` |
| Fix a wrong doc_type/title/status | Edit `data/manifest.yaml` (sync preserves your edit), then `legal-rag-ingest update <doc_id>` if the metadata should be re-indexed |
| Switched embedding model / chunk size | `legal-rag-ingest sync --force` |
| Chat with my own PDF once | Upload in the UI (kept in memory for 30 min) |
| Free disk space | Delete `data/processed/*` and `data/cache/`, then `sync --force` |
