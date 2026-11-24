"""Tests for the CI quality gate."""

from __future__ import annotations

from rag_project.evaluation.gate import check_thresholds


def test_passes_when_all_metrics_meet_minimums() -> None:
    assert (
        check_thresholds({"hit": 0.93, "mrr": 0.75}, {"hit": 0.90, "mrr": 0.70}) == []
    )


def test_exact_minimum_passes() -> None:
    assert check_thresholds({"hit": 0.9}, {"hit": 0.9}) == []


def test_reports_each_failing_metric() -> None:
    failures = check_thresholds({"hit": 0.80, "mrr": 0.75}, {"hit": 0.90, "mrr": 0.70})
    assert len(failures) == 1
    assert failures[0].startswith("hit:")


def test_missing_metric_fails() -> None:
    assert check_thresholds({}, {"hit": 0.9}) == ["hit: metric missing"]
