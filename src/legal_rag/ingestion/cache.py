"""On-disk embedding cache keyed by (model name, text hash). SQLite, stdlib only."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, path: Path | None, model: str) -> None:
        self.model = model
        target = ":memory:"
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            target = str(path)
        self._db = sqlite3.connect(target)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS emb (model TEXT, key TEXT, vec TEXT, "
            "PRIMARY KEY (model, key))"
        )

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT key, vec FROM emb WHERE model=? AND key IN ({marks})",  # noqa: S608
                [self.model, *chunk],
            ).fetchall()
            out.update({k: json.loads(v) for k, v in rows})
        return out

    def put_many(self, items: dict[str, list[float]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
            [(self.model, k, json.dumps(v)) for k, v in items.items()],
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
