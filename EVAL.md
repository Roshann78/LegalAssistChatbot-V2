# Evaluation

## Method
* Golden set JSONL (`src/legal_rag/eval/golden.py`): `id, question, category, reference_answer, gold_doc_ids,
  gold_locations, answerable, difficulty, split`. Answerable questions must name gold documents.
* Dev/test split ~70/30; the test split is frozen (`python -m legal_rag.eval freeze`, hash lock enforced on test runs) and run once per final config. See `eval/golden/README.md` for authoring (`legal_rag.eval.author`).
* Metrics (`src/legal_rag/eval/metrics.py`): hit@5, recall@8, MRR, nDCG@5 against gold doc IDs; abstention
  accuracy; citation validity (answered, has citations, none refer to non-existent passages); latency p50/p95.
* LLM-judge metrics (`eval/judge.py`, run with `--judge`): faithfulness, answer relevance, citation support.
  The judge model is pinned in `JUDGE_MODEL` (never defaulted or guessed), runs at temperature 0, uses versioned
  prompts (`config/prompts/judge_*_v1.txt`), and its model and prompt version are recorded in every run.
  Unusable judge outputs are excluded from means rather than counted as 0. Judge scores are **unaudited**:
  hand-check ~20 judged items before trusting them.
* Framework choice (Ragas vs DeepEval vs custom): **the 10-question trial was not run** (needs keys and a real
  corpus). The code uses a thin custom judge, which has no extra dependency and works offline in tests.
* Cost: tokens are estimated at ~4 chars/token (the streaming path returns no provider usage); cost uses
  `PRICE_IN_PER_1K` / `PRICE_OUT_PER_1K` (0 if unset, so cost shows $0 rather than a guess).
* Runs record git commit, config fingerprint, prompt version, generator, judge, embedder and golden-file hash.
* `--llm real` uses the configured Groq model; the default `fake` generator is for CI only.
* Configs (`eval/runner.py::CONFIGS`): `B0_baseline` (S0 + dense + prompt v1), `B1_structure`, `B3_hybrid`,
  `G1_prompt_v2` (adds prompt v2 and abstention gate). Change one thing per row.
* `python -m legal_rag.eval run ... --min hit@5=0.9` exits non-zero on regression (used in CI).

## Current results
**B0 has not been run on a real corpus** (there is none yet). Only a synthetic smoke fixture (4 invented documents, 8 questions, hashing embedder, extractive fake LLM)
is evaluated. It proves the harness and gate work; it says nothing about real-world quality. The ablation table
in `eval/experiment_log.md` is intentionally empty until the real corpus and golden set exist.

## Next
Collect corpus → author and verify golden questions → run B0 → one change at a time → freeze → test once.
