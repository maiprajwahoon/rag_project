"""Tests for the HTTP API, using fake services (no Qdrant, model or LLM needed)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_project.api import app, get_service
from rag_project.generation.schema import AnswerResult
from rag_project.ingestion.models import Chunk

CHUNK = Chunk(
    chunk_id="tutorial/query-params.md::0005",
    source_path="tutorial/query-params.md",
    title="Query Parameters",
    heading_path="Query Parameters > Required query parameters",
    chunk_index=5,
    text="Do not declare a default value.",
    token_count=10,
)


class FakeService:
    def __init__(self, result: AnswerResult) -> None:
        self.result = result

    def ask(self, question: str) -> AnswerResult:
        return self.result


class FailingService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def ask(self, question: str) -> AnswerResult:
        raise self.error


def make_client(service: object) -> TestClient:
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def answered() -> AnswerResult:
    return AnswerResult(
        question="q",
        answer="Do not declare a default value.",
        refused=False,
        citations=[CHUNK],
        retrieval_scores=[0.9, 0.8],
        attempts=1,
        prompt_version="answer_v1",
    )


def test_health() -> None:
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_citations() -> None:
    client = make_client(FakeService(answered()))
    response = client.post("/ask", json={"question": "How do I require a parameter?"})
    body = response.json()
    assert response.status_code == 200
    assert body["answer"] == "Do not declare a default value."
    assert body["refused"] is False
    assert body["citations"][0]["chunk_id"] == CHUNK.chunk_id
    assert body["citations"][0]["source_path"] == CHUNK.source_path
    assert body["retrieval_scores"] == [0.9, 0.8]


def test_refusal_is_returned_with_reason() -> None:
    refusal = AnswerResult(
        question="q",
        answer="I could not find enough support.",
        refused=True,
        refusal_reason="insufficient_evidence",
        attempts=1,
        prompt_version="answer_v1",
    )
    body = (
        make_client(FakeService(refusal))
        .post("/ask", json={"question": "What is the capital of France?"})
        .json()
    )
    assert body["refused"] is True
    assert body["refusal_reason"] == "insufficient_evidence"
    assert body["citations"] == []


def test_rejects_empty_question() -> None:
    client = make_client(FakeService(answered()))
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_rejects_overlong_question() -> None:
    client = make_client(FakeService(answered()))
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


def test_vector_store_outage_returns_503() -> None:
    error = ResponseHandlingException(ConnectionError("down"))
    response = make_client(FailingService(error)).post(
        "/ask", json={"question": "How do I validate?"}
    )
    assert response.status_code == 503
    assert "down" not in response.text


def test_generation_failure_returns_502() -> None:
    response = make_client(FailingService(RuntimeError("secret detail"))).post(
        "/ask", json={"question": "How do I validate?"}
    )
    assert response.status_code == 502
    assert "secret detail" not in response.text


def test_not_ready_when_service_missing() -> None:
    response = TestClient(app).post("/ask", json={"question": "How do I validate?"})
    assert response.status_code == 503
