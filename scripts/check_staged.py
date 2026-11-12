"""Pre-commit guard: fail if generated artifacts, secrets files or vector DBs are staged."""

from __future__ import annotations

import re
import subprocess
import sys

BLOCKED = [
    (re.compile(r"^data/processed/(?!\.gitkeep$)"), "processed data"),
    (re.compile(r"^data/cache/"), "cache"),
    (re.compile(r"(^|/)qdrant(_data)?/"), "vector DB folder"),
    (re.compile(r"(^|/)\.env($|\.(?!example$))"), ".env file"),
    (re.compile(r"\.(log|sqlite|sqlite3)$"), "log/db file"),
    (re.compile(r"^eval/runs/(?!\.gitkeep$)"), "generated eval run"),
    (re.compile(r"\.(safetensors|ckpt|gguf)$|(^|/)pytorch_model\.bin$"), "model weights"),
]


def blocked(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        for pattern, label in BLOCKED:
            if pattern.search(p):
                out.append(f"{p}  ({label})")
                break
    return out


def main() -> int:
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    bad = blocked(staged)
    if bad:
        print("Refusing to commit generated/secret artifacts:\n  " + "\n  ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
