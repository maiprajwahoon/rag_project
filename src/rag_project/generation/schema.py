"""Pydantic models for generated answers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_project.ingestion.models import Chunk


class LLMAnswer(BaseModel):
    """Structured output the LLM must return."""

    refused: bool
    answer: str = ""
    citations: list[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    """Final result returned to callers."""

    question: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[Chunk] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    attempts: int = 1
    prompt_version: str = ""
