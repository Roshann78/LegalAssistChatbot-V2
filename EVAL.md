# Evaluation

## Method
* Golden set JSONL (`src/legal_rag/eval/golden.py`): `id, question, category, reference_answer, gold_doc_ids,
  gold_locations, answerable, difficulty, split`. Answerable questions must name gold documents.
* Dev/test split ~70/30; the test split is frozen and run once per final config.
* Metrics (`src/legal_rag/eval/metrics.py`): hit@5, recall@8, MRR, nDCG@5 against gold doc IDs; abstention
  accuracy; citation validity (answered, has citations, none refer to non-existent passages); latency p50/p95.
* Not implemented: LLM-judged faithfulness/answer relevance, citation *support* accuracy, token cost.
* Configs (`eval/runner.py::CONFIGS`): `B0_baseline` (S0 + dense + prompt v1), `B1_structure`, `B3_hybrid`,
  `G1_prompt_v2` (adds prompt v2 and abstention gate). Change one thing per row.
* `python -m legal_rag.eval run ... --min hit@5=0.9` exits non-zero on regression (used in CI).

## Current results
**Only a synthetic smoke fixture** (4 invented documents, 8 questions, hashing embedder, extractive fake LLM)
is evaluated. It proves the harness and gate work; it says nothing about real-world quality. The ablation table
in `eval/experiment_log.md` is intentionally empty until the real corpus and golden set exist.

## Next
Collect corpus → author and verify golden questions → run B0 → one change at a time → freeze → test once.
