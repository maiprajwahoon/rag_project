"""Hybrid retriever with switchable modes for ablation studies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from rag_project.ingestion.models import Chunk
from rag_project.rag import ChunkSearcher, QueryEmbedder
from rag_project.retrieval.fusion import reciprocal_rank_fusion

Mode = Literal["vector", "hybrid", "hybrid_rerank"]


class KeywordSearcher(Protocol):
    """Anything that can search chunks by keywords."""

    def search(self, query: str, k: int, /) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks with scores."""
        ...


class Reranker(Protocol):
    """Anything that can re-rank candidate chunks."""

    def rerank(
        self, query: str, chunks: Sequence[Chunk], top_k: int, /
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks, best first."""
        ...


class HybridRetriever:
    """Retrieve with vector search, BM25 + RRF fusion, and optional re-ranking."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        store: ChunkSearcher,
        keyword: KeywordSearcher,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._keyword = keyword
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._k = k
        self._rrf_k = rrf_k

    def retrieve(
        self, query: str, mode: Mode = "hybrid_rerank"
    ) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, score) pairs for the chosen mode."""
        if mode == "hybrid_rerank" and self._reranker is None:
            raise ValueError("hybrid_rerank mode needs a reranker")
        vector = self._embedder.embed_query(query)
        width = self._k if mode == "vector" else self._candidate_k
        vector_hits = self._store.search(vector, width)
        if mode == "vector":
            return vector_hits[: self._k]

        keyword_hits = self._keyword.search(query, self._candidate_k)
        by_id = {c.chunk_id: c for c, _ in [*vector_hits, *keyword_hits]}
        fused = reciprocal_rank_fusion(
            [
                [c.chunk_id for c, _ in vector_hits],
                [c.chunk_id for c, _ in keyword_hits],
            ],
            self._rrf_k,
        )
        merged = [(by_id[cid], score) for cid, score in fused]
        if mode == "hybrid" or self._reranker is None:
            return merged[: self._k]
        candidates = [c for c, _ in merged[: self._candidate_k]]
        return self._reranker.rerank(query, candidates, self._k)
