# Metadata schema

Defined in `src/legal_rag/ingestion/schema.py` (`DocumentRecord`, `ChunkMetadata`).

**Document fields:** `doc_id` (stable slug), `title`, `doc_type`, `domain`, `path` (relative to data dir),
`jurisdiction`, `authority_or_court`, `year`, `date`, `case_citation`, `act_name`, `status`
(in_force/amended/superseded/repealed), `effective_from`, `superseded_by`, `version`, `language`, `source_url`,
`source_type` (pdf/text/api_text), `file_hash` (SHA-256), `page_count`, `text_quality`.
Why: status/superseded_by drive legal-currency handling; file_hash drives idempotent re-ingestion;
domain/doc_type are retrieval filters; source_url keeps provenance.

**Chunk fields:** `doc_id`, `chunk_id` (`<doc>:<strategy-version>:<n>`), `section_or_article_no`,
`heading_path`, `page_start/end`, `paragraph_no`, `chunk_type` (strategy), `parent_id` (reserved),
`config_fingerprint`. Every chunk also carries the document fields above in its payload.

Cross-field rules: superseded docs need `superseded_by`; judgments need `authority_or_court`;
`superseded_by` must resolve to a manifest `doc_id` (checked by `validate`).
