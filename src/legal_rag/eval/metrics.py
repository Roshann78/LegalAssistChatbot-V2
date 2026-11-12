"""Retrieval and system metrics computed from gold IDs (no LLM judge needed)."""

from __future__ import annotations

import math


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return float(any(d in gold for d in retrieved[:k]))


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG; each gold doc is counted once at its first rank."""
    seen: set[str] = set()
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in gold and d not in seen:
            seen.add(d)
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def location_hit(results_sections: list[str | None], gold_locations: list[str]) -> float:
    """1.0 if any retrieved section label contains a gold location (case-insensitive)."""
    if not gold_locations:
        return 0.0
    labels = [s.lower() for s in results_sections if s]
    return float(
        any(g.lower() in lab or lab in g.lower() for g in gold_locations for lab in labels)
    )
