# make.ps1 - Windows PowerShell task runner matching the Makefile targets
param(
    [Parameter(Position=0)]
    [ValidateSet("setup", "test", "lint", "format", "run", "check", "clean", "ingest", "reindex",
                 "ingest-check", "eval-smoke", "eval-dev", "eval-test", "author-draft",
                 "eval-freeze", "demo", "load-test", "up", "down")]
    [string]$Target = "test"
)

$ErrorActionPreference = "Stop"
$ing = "python -m legal_rag.ingestion"
$smoke = "--config G1_prompt_v2 --golden eval/golden/smoke.jsonl --data-dir eval/fixtures --manifest eval/fixtures/manifest.yaml --embedder hashing-512"

switch ($Target) {
    "setup"        { uv sync }
    "test"         { uv run pytest }
    "lint"         { uv run ruff check . }
    "format"       { uv run ruff format . }
    "run"          { uv run uvicorn legal_rag.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000 }
    "check"        { uv run ruff check .; uv run pytest }
    "clean"        {
        foreach ($d in ".pytest_cache", ".ruff_cache", "htmlcov", ".coverage") {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
    }
    "ingest"       { Invoke-Expression "uv run $ing validate"; Invoke-Expression "uv run $ing verify" }
    "reindex"      { Invoke-Expression "uv run $ing --yes reindex --all" }
    "ingest-check" { Invoke-Expression "uv run $ing --yes report --out docs/ingestion_report.md" }
    "eval-smoke"   { Invoke-Expression "uv run python -m legal_rag.eval run $smoke --min hit@5=0.9 abstention_acc=0.9" }
    "eval-dev"     { uv run python -m legal_rag.eval run --config B0_baseline --split dev --golden eval/golden/dev.jsonl --manifest data/manifest.yaml --log }
    "eval-test"    { uv run python -m legal_rag.eval run --config R_final --split test --golden eval/golden/test.jsonl --manifest data/manifest.yaml }
    "author-draft" { uv run python -m legal_rag.eval.author draft --per-category 5 }
    "eval-freeze"  { uv run python -m legal_rag.eval freeze --golden eval/golden/test.jsonl }
    "demo"         { uv run python scripts/demo_server.py }
    "load-test"    { uv run python scripts/load_test.py --concurrency 10 --seconds 60 }
    "up"           { docker compose up -d --build }
    "down"         { docker compose down }
}
