"""Golden question set: loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


class GoldenItem(BaseModel):
    """One evaluation question with its expected source pages."""

    id: str
    question: str
    pages: list[str] = Field(default_factory=list)
    answerable: bool

    @model_validator(mode="after")
    def _pages_match_answerable(self) -> Self:
        if self.answerable and not self.pages:
            raise ValueError(f"{self.id}: answerable question needs pages")
        if not self.answerable and self.pages:
            raise ValueError(f"{self.id}: unanswerable question must have no pages")
        return self


def load_golden(path: Path) -> list[GoldenItem]:
    """Load a JSONL golden set, rejecting invalid rows and duplicate ids."""
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = GoldenItem.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path}:{number}: duplicate id {item.id}")
        seen.add(item.id)
        items.append(item)
    if not items:
        raise ValueError(f"{path}: no questions found")
    return items
