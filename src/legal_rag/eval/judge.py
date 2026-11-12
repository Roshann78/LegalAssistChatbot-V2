"""LLM-as-judge metrics (faithfulness, answer relevance, citation support).

The judge model is pinned via settings (never guessed), runs at temperature 0, and its model
name and prompt versions are recorded in every run's output.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from legal_rag.generation.prompts import PROMPT_DIR
from legal_rag.llm import LLMClient
from legal_rag.retrieval.types import SearchResult

JUDGE_PROMPT_VERSION = "v1"
_JSON = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class JudgeScore:
    score: float | None  # None = judge output unusable (excluded from means)
    reason: str = ""


def parse_judge_output(raw: str) -> JudgeScore:
    m = _JSON.search(raw or "")
    if not m:
        return JudgeScore(None, "unparseable")
    try:
        data = json.loads(m.group(0))
        score = float(data["score"])
    except (ValueError, KeyError, TypeError):
        return JudgeScore(None, "unparseable")
    if not 0.0 <= score <= 1.0:
        return JudgeScore(None, "out of range")
    return JudgeScore(score, str(data.get("reason", ""))[:200])


def _passages(results: list[SearchResult]) -> str:
    return "\n\n".join(f"[{i}] {r.text}" for i, r in enumerate(results, 1))


class Judge:
    def __init__(self, llm: LLMClient, model_name: str, max_tokens: int = 200) -> None:
        self.llm, self.model_name, self.max_tokens = llm, model_name, max_tokens
        self.version = JUDGE_PROMPT_VERSION
        self.calls = 0

    def _prompt(self, name: str) -> str:
        return Path(PROMPT_DIR / f"judge_{name}_{self.version}.txt").read_text(encoding="utf-8")

    async def _ask(self, name: str, **fields: str) -> JudgeScore:
        messages = [{"role": "user", "content": self._prompt(name).format(**fields)}]
        self.calls += 1
        try:
            resp = await asyncio.to_thread(
                self.llm.generate, messages, temperature=0.0, max_tokens=self.max_tokens
            )
        except Exception as exc:  # a failed judge call must not abort the run
            return JudgeScore(None, f"judge error: {type(exc).__name__}")
        return parse_judge_output(resp.content)

    async def faithfulness(self, answer: str, results: list[SearchResult]) -> JudgeScore:
        return await self._ask("faithfulness", passages=_passages(results), answer=answer)

    async def relevance(self, question: str, answer: str) -> JudgeScore:
        return await self._ask("relevance", question=question, answer=answer)

    async def citation_support(self, answer: str, cited: list[SearchResult]) -> JudgeScore:
        return await self._ask("citation", passages=_passages(cited), answer=answer)


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


def split_claims(answer: str, min_chars: int = 25) -> list[str]:
    """Sentence-level claims (citation markers kept; very short fragments skipped)."""
    return [s.strip() for s in _SENTENCE.split(answer) if len(s.strip()) >= min_chars]


@dataclass(frozen=True)
class GuardResult:
    unsupported: list[str]
    checked: int
    unjudged: int  # claims whose judge output was unusable
    judge_calls: int


async def faithfulness_guard(
    judge: Judge, answer: str, results: list[SearchResult], threshold: float = 0.5
) -> GuardResult:
    """Flag sentences not supported by the passages (one judge call per sentence).

    Prototype for the roadmap's 5.6 decision: costly, so it defaults to offline/eval-only use.
    """
    claims = split_claims(answer)
    unsupported, unjudged = [], 0
    for claim in claims:
        score = (await judge._ask("claim", passages=_passages(results), claim=claim)).score
        if score is None:
            unjudged += 1
        elif score < threshold:
            unsupported.append(claim)
    return GuardResult(unsupported, len(claims), unjudged, len(claims))
