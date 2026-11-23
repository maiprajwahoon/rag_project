"""Tests for the retrieval evaluation runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_project.evaluation.golden import GoldenItem
from rag_project.evaluation.runner import (
    evaluate_retrieval,
    format_report,
    save_results,
)
from rag_project.ingestion.models import Chunk
from rag_project.retrieval.retriever import Mode


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


class FakeRetriever:
    def retrieve(self, query: str, mode: Mode) -> list[tuple[Chunk, float]]:
        if mode == "vector":
            table = {
                "q1": [("x.md", 0.9), ("a.md", 0.8)],
                "q2": [("b.md", 0.7)],
                "q3": [("z.md", 0.4)],
            }
        else:
            table = {
                "q1": [("a.md", 1.0)],
                "q2": [("b.md", 1.0)],
                "q3": [("z.md", 0.1)],
            }
        return [(chunk(page), score) for page, score in table[query]]


ITEMS = [
    GoldenItem(id="a1", question="q1", pages=["a.md"], answerable=True),
    GoldenItem(id="a2", question="q2", pages=["b.md"], answerable=True),
    GoldenItem(id="u1", question="q3", pages=[], answerable=False),
]


def test_evaluate_retrieval_scores_each_mode() -> None:
    report = evaluate_retrieval(FakeRetriever(), ITEMS, k=5, golden_sha="abc")
    by_mode = {m.mode: m for m in report.modes}
    assert by_mode["vector"].mrr == 0.75
    assert by_mode["vector"].ranks == {"a1": 2, "a2": 1}
    assert by_mode["hybrid"].mrr == 1.0
    assert by_mode["hybrid_rerank"].hit_rate == 1.0
    assert report.top_vector_scores_answerable == {"a1": 0.9, "a2": 0.7}
    assert report.top_vector_scores_unanswerable == {"u1": 0.4}


def test_evaluate_requires_answerable_questions() -> None:
    only_unanswerable = [ITEMS[2]]
    with pytest.raises(ValueError):
        evaluate_retrieval(FakeRetriever(), only_unanswerable, k=5, golden_sha="abc")


def test_report_text_and_saving(tmp_path: Path) -> None:
    report = evaluate_retrieval(FakeRetriever(), ITEMS, k=5, golden_sha="abc")
    text = format_report(report)
    assert "hybrid_rerank" in text
    assert "gap" in text
    path = save_results(report, tmp_path)
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("{")
