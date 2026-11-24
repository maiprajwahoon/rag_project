"""HTTP API for the RAG service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_project.rag import RagService

logger = logging.getLogger(__name__)


def build_service() -> RagService:
    """Build the RAG service. Heavy imports stay here so importing the API is cheap."""
    from rag_project.generation.generator import Generator
    from rag_project.generation.llm import GeminiClient
    from rag_project.indexing.embedder import Embedder
    from rag_project.indexing.store import QdrantStore

    return RagService(Embedder(), QdrantStore(), Generator(GeminiClient()))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models once at startup and keep the service on app.state."""
    app.state.service = build_service()
    yield


app = FastAPI(title="FastAPI Docs RAG", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    """A question about the FastAPI documentation."""

    question: str = Field(min_length=3, max_length=1000)


class CitationOut(BaseModel):
    """A source chunk that supports the answer."""

    chunk_id: str
    source_path: str
    heading_path: str


class AskResponse(BaseModel):
    """A cited answer, or a refusal with its reason."""

    answer: str
    refused: bool
    refusal_reason: str | None
    attempts: int
    retrieval_scores: list[float]
    citations: list[CitationOut]


def get_service(request: Request) -> RagService:
    """Return the service loaded at startup, or 503 if it is not ready."""
    service: RagService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="service is not ready")
    return service


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest, service: Annotated[RagService, Depends(get_service)]
) -> AskResponse:
    """Answer a question from the indexed documentation, with citations."""
    try:
        result = service.ask(request.question)
    except ResponseHandlingException as exc:
        logger.error("vector store unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="vector store unavailable") from exc
    except Exception as exc:
        logger.error("answer generation failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="answer generation failed") from exc
    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        attempts=result.attempts,
        retrieval_scores=result.retrieval_scores,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                source_path=c.source_path,
                heading_path=c.heading_path,
            )
            for c in result.citations
        ],
    )
