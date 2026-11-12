.PHONY: ingest-check setup test lint format run check clean ingest reindex eval-smoke eval-dev eval-test up down

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

run:
	uv run uvicorn legal_rag.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage

ingest:
	uv run python -m legal_rag.ingestion validate
	uv run python -m legal_rag.ingestion verify

reindex:
	uv run python -m legal_rag.ingestion reindex --all

eval-smoke:
	uv run python -m legal_rag.eval run --config G1_prompt_v2 --golden eval/golden/smoke.jsonl --data-dir eval/fixtures --manifest eval/fixtures/manifest.yaml --embedder hashing-512 --min hit@5=0.9 abstention_acc=0.9

eval-dev:
	uv run python -m legal_rag.eval run --config B0_baseline --split dev --golden eval/golden/dev.jsonl --manifest data/manifest.yaml --log

eval-test:
	uv run python -m legal_rag.eval run --config R_final --split test --golden eval/golden/test.jsonl --manifest data/manifest.yaml

up:
	docker compose up -d --build

down:
	docker compose down

ingest-check:
	uv run python -m legal_rag.ingestion --yes report --out docs/ingestion_report.md
