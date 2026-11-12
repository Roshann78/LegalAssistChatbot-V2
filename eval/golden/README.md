# Golden set

JSONL, one question per line, validated by `legal_rag.eval.golden.GoldenQuestion`.

| field | meaning |
|---|---|
| `id` | unique within the file (`dev-conceptual-004`) |
| `question` | what a user would ask |
| `category` | `section_lookup, conceptual, case_law, old_vs_new, multi_hop, banking_procedural, definitions, out_of_scope, adversarial` |
| `reference_answer` | short, verified against the source PDF |
| `gold_doc_ids` | manifest `doc_id`s that contain the answer (required when `answerable`) |
| `gold_locations` | section/article/paragraph/page labels |
| `answerable` | `false` for out-of-scope questions (system should abstain) |
| `difficulty` | `easy|medium|hard` |
| `split` | `dev` or `test` |
| `domain` | optional `legal`/`banking` filter |

## Target mix (~100)
section lookup 15 · conceptual 15 · case law 15 · old-vs-new 10 · multi-hop 10 · banking procedural 15 ·
definitions 5 · out-of-scope 10 · adversarial/injection 5.

## Split policy
~70/30 dev/test. **dev** is for tuning. **test** is frozen: run `python -m legal_rag.eval freeze` once the
test file is final; test runs then refuse a modified file, and `promote` refuses to append to a frozen file
(new questions go in a new versioned file such as `test_v2.jsonl`). Never tune on test results; run it once
per final config. No question text may appear in both splits (`split_overlap`).

## Authoring workflow
1. `python -m legal_rag.eval.author draft --per-category 5` samples chunks (round-robin across documents) and
   asks an LLM for candidates into `eval/golden/drafts/` (never into dev/test).
2. `python -m legal_rag.eval.author review <drafts file>` shows each draft beside its source passage:
   accept / edit / reject. **You must check every question and gold reference against the actual PDF.**
3. `python -m legal_rag.eval.author promote --split dev` validates accepted items and appends to `dev.jsonl`.
