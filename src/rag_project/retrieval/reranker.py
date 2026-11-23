"""Cross-encoder re-ranking of candidate chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sentence_transformers import CrossEncoder

from rag_project.ingestion.models import Chunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def order_by_scores(
    chunks: Sequence[Chunk], scores: Sequence[float], top_k: int
) -> list[tuple[Chunk, float]]:
    """Sort chunks by score (descending, stable) and keep the top_k."""
    if len(chunks) != len(scores):
        raise ValueError("chunks and scores must have the same length")
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [(chunks[i], float(scores[i])) for i in order[:top_k]]


class CrossEncoderReranker:
    """Score (query, chunk) pairs jointly and keep the best chunks.

    Scores are raw model logits: they can be negative and are not comparable
    to cosine similarities.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model: Any | None = None,
        batch_size: int = 16,
    ) -> None:
        self._model = (
            model if model is not None else CrossEncoder(model_name, max_length=512)
        )
        self._batch_size = batch_size

    def rerank(
        self, query: str, chunks: Sequence[Chunk], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks ordered by cross-encoder relevance."""
        if not chunks:
            return []
        pairs = [(query, f"{c.heading_path}\n{c.text}") for c in chunks]
        raw = self._model.predict(pairs, batch_size=self._batch_size)
        return order_by_scores(chunks, [float(s) for s in raw], top_k)
