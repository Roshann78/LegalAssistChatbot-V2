# Ingestion report

Generated 2026-11-18T18:56:41+00:00  fingerprint `7a55b7cf2cc9542e`

- documents: 10  chunks: 12828
- last run: indexed 0, skipped 10, cache hits 0, 0.06s
- tokens/chunk: min 1, median 169, p95 198, max 200

## Chunks per document type

| doc_type | docs | chunks |
|---|---|---|
| constitution | 1 | 4245 |
| judgment | 2 | 2856 |
| report | 2 | 1033 |
| statute | 5 | 4694 |

## Per document

| doc_id | chunks | min tok | max tok |
|---|---|---|---|
| in-constitution-of-india | 4245 | 3 | 200 |
| in-companies-act-2013 | 3082 | 3 | 200 |
| in-dpdp-act-2023 | 114 | 39 | 200 |
| in-pmla-2002 | 424 | 4 | 200 |
| sc-puttaswamy-2017 | 1578 | 24 | 200 |
| sc-navtej-johar-2018 | 1278 | 26 | 200 |
| in-dicgc-act-1961 | 242 | 5 | 200 |
| in-ibc-2016 | 832 | 9 | 200 |
| dicgc-annual-report-2025-26 | 605 | 1 | 200 |
| rbi-nsfi-2025-30 | 428 | 1 | 200 |

## Needs attention

- zero or >3000 chunks: ['in-constitution-of-india', 'in-companies-act-2013']
- low-text pages (OCR candidates): none
- chunk types: {'s1': 8939, 's2': 2856, 's3': 1033}
