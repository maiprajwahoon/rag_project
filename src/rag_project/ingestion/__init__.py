"""Ingestion module for document loading, parsing, and chunking."""

from rag_project.ingestion.chunker import chunk_documents
from rag_project.ingestion.loader import load_documents, resolve_includes
from rag_project.ingestion.models import Chunk, Document

__all__ = [
    "Chunk",
    "Document",
    "chunk_documents",
    "load_documents",
    "resolve_includes",
]
