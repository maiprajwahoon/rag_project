"""Tests for grounded generation, prompts and the RAG service."""

from __future__ import annotations

import json

from rag_project.generation.generator import REFUSAL_TEXT, Generator, parse_llm_answer
from rag_project.generation.llm import to_text
from rag_project.generation.prompts import load_prompt
from rag_project.ingestion.models import Chunk
from rag_project.rag import RagService

FENCE = "`" * 3


def make_chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_path=cid.split("::")[0],
        title="T",
        heading_path="T > S",
        chunk_index=0,
        text="Use Query for query parameters.",
        token_count=10,
    )


CHUNKS = [make_chunk("a.md::0000"), make_chunk("b.md::0001")]


class FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.replies.pop(0)


def reply(
    refused: bool = False, answer: str = "", citations: list[str] | None = None
) -> str:
    return json.dumps(
        {"refused": refused, "answer": answer, "citations": citations or []}
    )


def test_valid_answer_maps_citations() -> None:
    llm = FakeLLM([reply(answer="Use Query.", citations=["a.md::0000"])])
    result = Generator(llm).generate("q", CHUNKS)
    assert not result.refused
    assert [c.chunk_id for c in result.citations] == ["a.md::0000"]
    assert result.attempts == 1


def test_model_refusal_is_clean() -> None:
    result = Generator(FakeLLM([reply(refused=True)])).generate("q", CHUNKS)
    assert result.refused
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert result.refusal_reason == "insufficient_evidence"


def test_unknown_citation_triggers_retry() -> None:
    llm = FakeLLM(
        [
            reply(answer="x", citations=["zzz"]),
            reply(answer="Use Query.", citations=["b.md::0001"]),
        ]
    )
    result = Generator(llm).generate("q", CHUNKS)
    assert not result.refused
    assert result.attempts == 2


def test_answer_without_citations_is_rejected() -> None:
    llm = FakeLLM([reply(answer="x"), reply(answer="x")])
    result = Generator(llm).generate("q", CHUNKS)
    assert result.refused
    assert result.refusal_reason == "invalid_output"
    assert llm.calls == 2


def test_invalid_json_twice_refuses() -> None:
    llm = FakeLLM(["not json", "still not json"])
    result = Generator(llm).generate("q", CHUNKS)
    assert result.refused
    assert result.refusal_reason == "invalid_output"


def test_no_chunks_refuses_without_calling_llm() -> None:
    llm = FakeLLM([])
    result = Generator(llm).generate("q", [])
    assert result.refusal_reason == "no_context"
    assert llm.calls == 0


def test_parse_handles_code_fences() -> None:
    raw = FENCE + "json\n" + reply(answer="A", citations=["a.md::0000"]) + "\n" + FENCE
    assert parse_llm_answer(raw).citations == ["a.md::0000"]


def test_to_text_handles_block_lists() -> None:
    blocks = [{"type": "text", "text": "OK", "extras": {"signature": "abc"}}]
    assert to_text(blocks) == "OK"
    assert to_text("plain") == "plain"


def test_prompt_loads_with_placeholders() -> None:
    prompt = load_prompt()
    assert prompt.version == "answer_v1"
    assert "{context}" in prompt.user_template
    assert "{question}" in prompt.user_template


class FakeEmbedder:
    def embed_query(self, text: str, /) -> list[float]:
        return [0.0]


class FakeSearcher:
    def search(self, query_vector: list[float], k: int, /) -> list[tuple[Chunk, float]]:
        return [(CHUNKS[0], 0.9)]


def test_service_returns_answer_and_scores() -> None:
    generator = Generator(
        FakeLLM([reply(answer="Use Query.", citations=["a.md::0000"])])
    )
    result = RagService(FakeEmbedder(), FakeSearcher(), generator, k=3).ask("q")
    assert not result.refused
    assert result.retrieval_scores == [0.9]
