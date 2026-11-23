"""Tests for the generation evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_project.evaluation.generation_eval import (
    CachedLLM,
    evaluate_generation,
    format_generation_report,
    parse_verdict,
    save_generation_report,
)
from rag_project.evaluation.golden import GoldenItem
from rag_project.generation.schema import AnswerResult
from rag_project.ingestion.models import Chunk


class CountingLLM:
    def __init__(self, reply: str = "r") -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def test_cache_hit_avoids_second_call(tmp_path: Path) -> None:
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path / "c.json")
    assert llm.complete("s", "u") == "r"
    assert llm.complete("s", "u") == "r"
    assert inner.calls == 1
    assert llm.real_calls == 1


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    CachedLLM(CountingLLM("first"), path).complete("s", "u")
    second = CountingLLM("second")
    assert CachedLLM(second, path).complete("s", "u") == "first"
    assert second.calls == 0


def test_namespace_separates_entries(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    CachedLLM(CountingLLM("a"), path, namespace="m1").complete("s", "u")
    other = CountingLLM("b")
    assert CachedLLM(other, path, namespace="m2").complete("s", "u") == "b"


def test_throttle_sleeps_between_real_calls(tmp_path: Path) -> None:
    sleeps: list[float] = []
    llm = CachedLLM(
        CountingLLM(),
        tmp_path / "c.json",
        min_interval=5.0,
        sleep=sleeps.append,
        clock=lambda: 100.0,
    )
    llm.complete("s", "one")
    llm.complete("s", "two")
    assert sleeps == [5.0]


def test_parse_verdict() -> None:
    verdict = parse_verdict('x {"supported": false, "unsupported_claims": ["c"]} y')
    assert verdict.supported is False
    assert verdict.unsupported_claims == ["c"]
    with pytest.raises(ValueError):
        parse_verdict("no json here")


def chunk(page: str) -> Chunk:
    return Chunk(
        chunk_id=f"{page}::0000",
        source_path=page,
        title="t",
        heading_path="t",
        chunk_index=0,
        text="x",
        token_count=1,
    )


class FakeAsker:
    def ask(self, question: str) -> AnswerResult:
        if question in ("q1", "q4"):
            page = "a.md" if question == "q1" else "z.md"
            return AnswerResult(
                question=question,
                answer="ans",
                refused=False,
                citations=[chunk(page)],
                prompt_version="v1",
            )
        return AnswerResult(
            question=question,
            answer="no",
            refused=True,
            refusal_reason="insufficient_evidence",
            prompt_version="v1",
        )


ITEMS = [
    GoldenItem(id="a1", question="q1", pages=["a.md"], answerable=True),
    GoldenItem(id="a2", question="q2", pages=["b.md"], answerable=True),
    GoldenItem(id="u1", question="q3", pages=[], answerable=False),
    GoldenItem(id="u2", question="q4", pages=[], answerable=False),
]


def test_evaluate_generation_metrics(tmp_path: Path) -> None:
    judge = CountingLLM('{"supported": true, "unsupported_claims": []}')
    report = evaluate_generation(FakeAsker(), ITEMS, judge, "sha", "m", 5)
    s = report.summary
    assert s.answered_when_answerable == 0.5
    assert s.refused_when_unanswerable == 0.5
    assert s.accuracy == 0.5
    assert s.cited_page_hit_rate == 1.0
    assert s.faithfulness_rate == 1.0
    assert s.n_judged == 2
    assert report.prompt_version == "v1"
    assert judge.calls == 2


def test_without_judge_faithfulness_is_not_measured() -> None:
    report = evaluate_generation(FakeAsker(), ITEMS, None, "sha", "m", 5)
    assert report.summary.faithfulness_rate is None
    assert report.summary.n_judged == 0


def test_report_text_and_saving(tmp_path: Path) -> None:
    report = evaluate_generation(FakeAsker(), ITEMS, None, "sha", "m", 5)
    text = format_generation_report(report)
    assert "answered but should refuse: u2" in text
    assert "refused but answerable: a2" in text
    assert save_generation_report(report, tmp_path).exists()
