"""End-to-end service: embed, retrieve, generate."""

from __future__ import annotations

from typing import Protocol

from rag_project.generation.generator import Generator
from rag_project.generation.schema import AnswerResult
from rag_project.ingestion.models import Chunk


class QueryEmbedder(Protocol):
    """Anything that can embed a query."""

    def embed_query(self, text: str, /) -> list[float]:
        """Return the query vector."""
        ...


class ChunkSearcher(Protocol):
    """Anything that can search chunks by vector."""

    def search(self, query_vector: list[float], k: int, /) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks with scores."""
        ...


class RagService:
    """Ask a question and get a cited, validated answer."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        store: ChunkSearcher,
        generator: Generator,
        k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        self._k = k

    def ask(self, question: str) -> AnswerResult:
        """Retrieve context and generate an answer."""
        vector = self._embedder.embed_query(question)
        hits = self._store.search(vector, self._k)
        result = self._generator.generate(question, [chunk for chunk, _ in hits])
        result.retrieval_scores = [score for _, score in hits]
        return result
