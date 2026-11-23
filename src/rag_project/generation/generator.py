"""Grounded answer generation with citation enforcement."""

from __future__ import annotations

from typing import Protocol

from rag_project.generation.prompts import PromptConfig, load_prompt
from rag_project.generation.schema import AnswerResult, LLMAnswer
from rag_project.ingestion.models import Chunk

REFUSAL_TEXT = (
    "I could not find enough support in the FastAPI documentation "
    "to answer this question."
)


class LLMClient(Protocol):
    """Minimal interface the generator needs from a language model."""

    def complete(self, system: str, user: str) -> str:
        """Return the model's text reply."""
        ...


def format_context(chunks: list[Chunk]) -> str:
    """Render chunks with their ids so the model can cite them."""
    blocks = [f"[{c.chunk_id}] ({c.heading_path})\n{c.text}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


def parse_llm_answer(raw: str) -> LLMAnswer:
    """Extract and validate the JSON object in a model reply."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in reply")
    return LLMAnswer.model_validate_json(raw[start : end + 1])


def check_answer(parsed: LLMAnswer, allowed_ids: set[str]) -> str | None:
    """Return a problem description, or None if the answer is acceptable."""
    if parsed.refused:
        return None
    if not parsed.answer.strip():
        return "answer is empty but refused is false"
    if not parsed.citations:
        return "answer has no citations"
    unknown = [c for c in parsed.citations if c not in allowed_ids]
    if unknown:
        return f"unknown citation ids: {unknown}"
    return None


class Generator:
    """Generate cited answers and refuse when evidence is missing."""

    def __init__(
        self,
        llm: LLMClient,
        prompt: PromptConfig | None = None,
        max_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._prompt = prompt or load_prompt()
        self._max_retries = max_retries

    def generate(self, question: str, chunks: list[Chunk]) -> AnswerResult:
        """Answer from the given chunks, with validated citations."""
        ids = [c.chunk_id for c in chunks]
        if not chunks:
            return self._refuse(question, ids, "no_context", 0)
        user = self._prompt.user_template.format(
            context=format_context(chunks), question=question
        )
        allowed = set(ids)
        note = ""
        parsed: LLMAnswer | None = None
        attempts = 0
        for _ in range(self._max_retries + 1):
            attempts += 1
            raw = self._llm.complete(self._prompt.system, user + note)
            try:
                candidate = parse_llm_answer(raw)
            except ValueError as exc:  # pydantic ValidationError is a ValueError
                problem: str | None = f"invalid output ({str(exc)[:150]})"
            else:
                problem = check_answer(candidate, allowed)
                if problem is None:
                    parsed = candidate
                    break
            note = (
                f"\n\nYour previous reply was rejected: {problem}. "
                "Reply again with one valid JSON object that follows the rules."
            )
        if parsed is None:
            return self._refuse(question, ids, "invalid_output", attempts)
        if parsed.refused:
            return self._refuse(question, ids, "insufficient_evidence", attempts)
        by_id = {c.chunk_id: c for c in chunks}
        cited = [by_id[i] for i in dict.fromkeys(parsed.citations)]
        return AnswerResult(
            question=question,
            answer=parsed.answer.strip(),
            refused=False,
            citations=cited,
            retrieved_ids=ids,
            attempts=attempts,
            prompt_version=self._prompt.version,
        )

    def _refuse(
        self, question: str, ids: list[str], reason: str, attempts: int
    ) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer=REFUSAL_TEXT,
            refused=True,
            refusal_reason=reason,
            retrieved_ids=ids,
            attempts=attempts,
            prompt_version=self._prompt.version,
        )
