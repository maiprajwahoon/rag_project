"""Tests for golden-set loading and retrieval metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_project.evaluation.golden import load_golden
from rag_project.evaluation.metrics import (
    first_hit_rank,
    page_recall,
    reciprocal_rank,
)


def test_first_hit_rank() -> None:
    assert first_hit_rank(["x", "a", "a"], ["a"]) == 2
    assert first_hit_rank(["x", "y"], ["a"]) is None
    assert first_hit_rank([], ["a"]) is None


def test_page_recall_partial() -> None:
    assert page_recall(["a", "x"], ["a", "b"]) == 0.5
    assert page_recall(["a", "b"], ["a", "b"]) == 1.0


def test_page_recall_requires_expected_pages() -> None:
    with pytest.raises(ValueError):
        page_recall(["a"], [])


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def write_rows(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


GOOD_A = '{"id": "q1", "question": "Q?", "pages": ["a.md"], "answerable": true}'
GOOD_U = '{"id": "u1", "question": "Q?", "pages": [], "answerable": false}'


def test_load_golden_reads_valid_rows(tmp_path: Path) -> None:
    items = load_golden(write_rows(tmp_path, [GOOD_A, "", GOOD_U]))
    assert [i.id for i in items] == ["q1", "u1"]


def test_load_golden_rejects_duplicate_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(write_rows(tmp_path, [GOOD_A, GOOD_A]))


def test_load_golden_rejects_answerable_without_pages(tmp_path: Path) -> None:
    bad = '{"id": "q2", "question": "Q?", "pages": [], "answerable": true}'
    with pytest.raises(ValueError, match="needs pages"):
        load_golden(write_rows(tmp_path, [bad]))


def test_load_golden_rejects_unanswerable_with_pages(tmp_path: Path) -> None:
    bad = '{"id": "u2", "question": "Q?", "pages": ["a.md"], "answerable": false}'
    with pytest.raises(ValueError, match="no pages"):
        load_golden(write_rows(tmp_path, [bad]))


def test_load_golden_rejects_bad_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_golden(write_rows(tmp_path, ["{not json"]))
