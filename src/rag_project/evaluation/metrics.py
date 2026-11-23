"""Retrieval metrics computed at page level."""

from __future__ import annotations

from collections.abc import Sequence


def first_hit_rank(
    retrieved_pages: Sequence[str], expected_pages: Sequence[str]
) -> int | None:
    """Return the 1-based rank of the first chunk on an expected page, or None."""
    expected = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return rank
    return None


def page_recall(retrieved_pages: Sequence[str], expected_pages: Sequence[str]) -> float:
    """Fraction of expected pages that appear among the retrieved pages."""
    expected = set(expected_pages)
    if not expected:
        raise ValueError("expected_pages must not be empty")
    return len(expected & set(retrieved_pages)) / len(expected)


def reciprocal_rank(
    retrieved_pages: Sequence[str], expected_pages: Sequence[str]
) -> float:
    """Return 1/rank of the first hit, or 0.0 when nothing matches."""
    rank = first_hit_rank(retrieved_pages, expected_pages)
    return 0.0 if rank is None else 1.0 / rank
