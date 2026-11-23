"""Build the default retriever from the live Qdrant index."""

from __future__ import annotations

from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore
from rag_project.retrieval.bm25 import BM25Index
from rag_project.retrieval.reranker import CrossEncoderReranker
from rag_project.retrieval.retriever import HybridRetriever


def build_default_retriever(k: int = 5, candidate_k: int = 20) -> HybridRetriever:
    """Create a HybridRetriever backed by Qdrant, BM25 and a cross-encoder."""
    store = QdrantStore()
    return HybridRetriever(
        embedder=Embedder(),
        store=store,
        keyword=BM25Index(store.all_chunks()),
        reranker=CrossEncoderReranker(),
        candidate_k=candidate_k,
        k=k,
    )
