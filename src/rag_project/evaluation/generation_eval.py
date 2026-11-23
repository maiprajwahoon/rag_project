"""Generation evaluation: refusal accuracy, citation hit and judged faithfulness."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from rag_project.evaluation.golden import GoldenItem
from rag_project.generation.generator import LLMClient, format_context
from rag_project.generation.prompts import load_prompt
from rag_project.generation.schema import AnswerResult


class Asker(Protocol):
    """Anything that can answer a question."""

    def ask(self, question: str) -> AnswerResult:
        """Return a cited answer or a refusal."""
        ...


class CachedLLM:
    """Disk-cached, throttled wrapper around an LLMClient."""

    def __init__(
        self,
        inner: LLMClient,
        cache_path: Path,
        min_interval: float = 0.0,
        namespace: str = "",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._path = cache_path
        self._min_interval = min_interval
        self._namespace = namespace
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self._cache: dict[str, str] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.real_calls = 0

    def _key(self, system: str, user: str) -> str:
        raw = f"{self._namespace}\x00{system}\x00{user}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def complete(self, system: str, user: str) -> str:
        """Return a cached reply, or call the model (throttled) and cache it."""
        key = self._key(system, user)
        if key in self._cache:
            return self._cache[key]
        if self._last_call is not None:
            wait = self._min_interval - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        reply = self._inner.complete(system, user)
        self._last_call = self._clock()
        self.real_calls += 1
        self._cache[key] = reply
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache), encoding="utf-8")
        return reply


class JudgeVerdict(BaseModel):
    """Faithfulness verdict for one answer."""

    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)


def parse_verdict(raw: str) -> JudgeVerdict:
    """Extract and validate the JSON verdict in a judge reply."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in judge reply")
    return JudgeVerdict.model_validate_json(raw[start : end + 1])


def judge_answer(llm: LLMClient, result: AnswerResult) -> JudgeVerdict | None:
    """Ask the judge whether the answer is supported by its cited chunks."""
    prompt = load_prompt("judge_v1")
    user = prompt.user_template.format(
        context=format_context(result.citations), answer=result.answer
    )
    try:
        return parse_verdict(llm.complete(prompt.system, user))
    except ValueError:
        return None


class ItemOutcome(BaseModel):
    """What happened for one golden question."""

    id: str
    answerable: bool
    refused: bool
    refusal_reason: str | None = None
    attempts: int = 1
    cited_ids: list[str] = Field(default_factory=list)
    cited_page_hit: bool | None = None
    faithful: bool | None = None
    unsupported_claims: list[str] = Field(default_factory=list)


class GenerationSummary(BaseModel):
    """Aggregate generation metrics. None means not measurable."""

    n_answerable: int
    n_unanswerable: int
    answered_when_answerable: float | None
    refused_when_unanswerable: float | None
    accuracy: float | None
    cited_page_hit_rate: float | None
    faithfulness_rate: float | None
    n_judged: int


class GenerationReport(BaseModel):
    """Full generation evaluation report."""

    created: str
    golden_sha: str
    model: str
    prompt_version: str
    k: int
    summary: GenerationSummary
    outcomes: list[ItemOutcome]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(outcomes: Sequence[ItemOutcome]) -> GenerationSummary:
    """Compute aggregate metrics from per-question outcomes."""
    answerable = [o for o in outcomes if o.answerable]
    unanswerable = [o for o in outcomes if not o.answerable]
    answered_ok = sum(not o.refused for o in answerable)
    refused_ok = sum(o.refused for o in unanswerable)
    hits = [o.cited_page_hit for o in answerable if o.cited_page_hit is not None]
    judged = [o.faithful for o in outcomes if o.faithful is not None]
    return GenerationSummary(
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        answered_when_answerable=_rate(answered_ok, len(answerable)),
        refused_when_unanswerable=_rate(refused_ok, len(unanswerable)),
        accuracy=_rate(answered_ok + refused_ok, len(outcomes)),
        cited_page_hit_rate=_rate(sum(hits), len(hits)),
        faithfulness_rate=_rate(sum(judged), len(judged)),
        n_judged=len(judged),
    )


def evaluate_generation(
    asker: Asker,
    items: Sequence[GoldenItem],
    judge: LLMClient | None,
    golden_sha: str,
    model: str,
    k: int,
    progress: Callable[[int, int, ItemOutcome], None] | None = None,
) -> GenerationReport:
    """Run every golden question through the pipeline and score the results."""
    outcomes: list[ItemOutcome] = []
    prompt_version = ""
    for index, item in enumerate(items, start=1):
        result = asker.ask(item.question)
        prompt_version = result.prompt_version or prompt_version
        outcome = ItemOutcome(
            id=item.id,
            answerable=item.answerable,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            attempts=result.attempts,
            cited_ids=[c.chunk_id for c in result.citations],
        )
        if item.answerable and not result.refused:
            outcome.cited_page_hit = any(
                c.source_path in item.pages for c in result.citations
            )
        if judge is not None and not result.refused:
            verdict = judge_answer(judge, result)
            if verdict is not None:
                outcome.faithful = verdict.supported
                outcome.unsupported_claims = verdict.unsupported_claims
        outcomes.append(outcome)
        if progress is not None:
            progress(index, len(items), outcome)
    return GenerationReport(
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        golden_sha=golden_sha,
        model=model,
        prompt_version=prompt_version,
        k=k,
        summary=summarize(outcomes),
        outcomes=outcomes,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_generation_report(report: GenerationReport) -> str:
    """Render a plain-text summary with the questions worth inspecting."""
    s = report.summary
    reasons: dict[str, int] = {}
    for o in report.outcomes:
        if o.refused:
            key = o.refusal_reason or "unknown"
            reasons[key] = reasons.get(key, 0) + 1
    wrongly_answered = [
        o.id for o in report.outcomes if not o.answerable and not o.refused
    ]
    wrongly_refused = [o.id for o in report.outcomes if o.answerable and o.refused]
    wrong_page = [o.id for o in report.outcomes if o.cited_page_hit is False]
    unfaithful = [o.id for o in report.outcomes if o.faithful is False]
    lines = [
        f"model={report.model} prompt={report.prompt_version} k={report.k} golden={report.golden_sha}",
        "",
        f"answered when answerable ({s.n_answerable}): {_fmt(s.answered_when_answerable)}",
        f"refused when unanswerable ({s.n_unanswerable}): {_fmt(s.refused_when_unanswerable)}",
        f"overall accuracy: {_fmt(s.accuracy)}",
        f"cited an expected page (of answered): {_fmt(s.cited_page_hit_rate)}",
        f"faithfulness (judged {s.n_judged}): {_fmt(s.faithfulness_rate)}",
        "",
        f"refusal reasons: {reasons or 'none'}",
        f"answered but should refuse: {', '.join(wrongly_answered) or 'none'}",
        f"refused but answerable: {', '.join(wrongly_refused) or 'none'}",
        f"answered with no expected page cited: {', '.join(wrong_page) or 'none'}",
        f"judged unfaithful: {', '.join(unfaithful) or 'none'}",
    ]
    return "\n".join(lines)


def save_generation_report(report: GenerationReport, directory: Path) -> Path:
    """Write the report as JSON and return the file path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report.created).strftime("%Y%m%d_%H%M%S")
    path = directory / f"generation_{stamp}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
