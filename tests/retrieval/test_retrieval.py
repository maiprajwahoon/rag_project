"""Tests for BM25, rank fusion, re-ranking and the hybrid retriever."""

from __future__ import annotations

import pytest

from rag_project.ingestion.models import Chunk
from rag_project.retrieval.bm25 import BM25Index, tokenize
from rag_project.retrieval.fusion import reciprocal_rank_fusion
from rag_project.retrieval.reranker import CrossEncoderReranker, order_by_scores
from rag_project.retrieval.retriever import HybridRetriever


def make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source_path=cid.split("::")[0],
        title="T",
        heading_path="T > S",
        chunk_index=0,
        text=text,
        token_count=10,
    )


A = make_chunk("a.md::0000", "Use OAuth2PasswordBearer for the password flow")
B = make_chunk("b.md::0000", "Declare query parameters with Query and validation")
C = make_chunk("c.md::0000", "Path parameters are declared in the path string")
D = make_chunk("d.md::0000", "Deploy with Docker containers and workers winner")
CORPUS = [A, B, C, D]


def test_tokenize_keeps_identifiers() -> None:
    assert tokenize("Use OAuth2PasswordBearer, now!") == [
        "use",
        "oauth2passwordbearer",
        "now",
    ]


def test_bm25_finds_exact_identifier() -> None:
    index = BM25Index(CORPUS)
    hits = index.search("OAuth2PasswordBearer", k=3)
    assert hits[0][0].chunk_id == A.chunk_id
    assert index.search("docker", k=3)[0][0].chunk_id == D.chunk_id


def test_bm25_empty_query_returns_nothing() -> None:
    assert BM25Index(CORPUS).search("   ", k=3) == []


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["w", "x"]])
    assert fused[0][0] == "x"
    assert {item for item, _ in fused} == {"x", "y", "z", "w"}


def test_rrf_ties_keep_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
    assert [item for item, _ in fused] == ["a", "c", "b", "d"]


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([]) == []


def test_order_by_scores_sorts_and_truncates() -> None:
    result = order_by_scores([A, B, C], [0.1, 0.9, 0.5], top_k=2)
    assert [c.chunk_id for c, _ in result] == [B.chunk_id, C.chunk_id]


def test_order_by_scores_length_mismatch() -> None:
    with pytest.raises(ValueError):
        order_by_scores([A, B], [0.1], top_k=1)


class FakeModel:
    def predict(
        self, pairs: list[tuple[str, str]], batch_size: int = 16
    ) -> list[float]:
        return [1.0 if "winner" in text else 0.0 for _, text in pairs]


def test_cross_encoder_reranker_orders_by_model_score() -> None:
    reranker = CrossEncoderReranker(model=FakeModel())
    result = reranker.rerank("q", [A, B, D], top_k=2)
    assert result[0][0].chunk_id == D.chunk_id
    assert len(result) == 2


class FakeEmbedder:
    def embed_query(self, text: str, /) -> list[float]:
        return [0.0]


class FakeStore:
    def __init__(self, hits: list[tuple[Chunk, float]]) -> None:
        self.hits = hits

    def search(self, query_vector: list[float], k: int, /) -> list[tuple[Chunk, float]]:
        return self.hits[:k]


class FakeKeyword:
    def __init__(self, hits: list[tuple[Chunk, float]]) -> None:
        self.hits = hits

    def search(self, query: str, k: int, /) -> list[tuple[Chunk, float]]:
        return self.hits[:k]


def make_retriever(with_reranker: bool = True) -> HybridRetriever:
    return HybridRetriever(
        embedder=FakeEmbedder(),
        store=FakeStore([(A, 0.9), (B, 0.8), (C, 0.7)]),
        keyword=FakeKeyword([(D, 5.0), (A, 3.0)]),
        reranker=CrossEncoderReranker(model=FakeModel()) if with_reranker else None,
        candidate_k=3,
        k=2,
    )


def test_vector_mode_returns_vector_hits_only() -> None:
    result = make_retriever().retrieve("q", "vector")
    assert [c.chunk_id for c, _ in result] == [A.chunk_id, B.chunk_id]


def test_hybrid_mode_includes_keyword_only_hits() -> None:
    retriever = HybridRetriever(
        FakeEmbedder(),
        FakeStore([(A, 0.9), (B, 0.8), (C, 0.7)]),
        FakeKeyword([(D, 5.0), (A, 3.0)]),
        candidate_k=3,
        k=5,
    )
    ids = [c.chunk_id for c, _ in retriever.retrieve("q", "hybrid")]
    assert ids[0] == A.chunk_id
    assert D.chunk_id in ids


def test_hybrid_rerank_uses_reranker() -> None:
    result = make_retriever().retrieve("q", "hybrid_rerank")
    assert result[0][0].chunk_id == D.chunk_id
    assert len(result) == 2


def test_hybrid_rerank_requires_reranker() -> None:
    with pytest.raises(ValueError):
        make_retriever(with_reranker=False).retrieve("q", "hybrid_rerank")
