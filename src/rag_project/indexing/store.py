"""Qdrant vector store wrapper for chunk upsert, search and export."""

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from rag_project.config import get_settings
from rag_project.ingestion.models import Chunk

_UPSERT_BATCH_SIZE = 128
_SCROLL_BATCH_SIZE = 256
_NAMESPACE_UUID = uuid.UUID(
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
)  # uuid.NAMESPACE_URL


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Derive a deterministic UUID5 point ID from a chunk's string ID."""
    return str(uuid.uuid5(_NAMESPACE_UUID, chunk_id))


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from a stored Qdrant payload."""
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        source_path=str(payload["source_path"]),
        title=str(payload["title"]),
        heading_path=str(payload["heading_path"]),
        anchor=payload.get("anchor"),
        chunk_index=int(payload["chunk_index"]),
        text=str(payload["text"]),
        token_count=int(payload["token_count"]),
    )


class QdrantStore:
    """Wraps a Qdrant collection for chunk storage and retrieval.

    Args:
        collection_name: Name of the Qdrant collection to operate on.
            Defaults to ``"fastapi_docs"``.
    """

    def __init__(self, collection_name: str = "fastapi_docs") -> None:
        """Initialise the store with a Qdrant client and collection name."""
        self._collection_name = collection_name
        self._client = QdrantClient(url=get_settings().qdrant_url)

    def ensure_collection(self, vector_size: int = 384) -> None:
        """Create the collection with cosine distance if it does not exist."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> None:
        """Upsert chunks with their vectors; idempotent via UUID5 point IDs."""
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload: dict[str, object] = {
                "chunk_id": chunk.chunk_id,
                "source_path": chunk.source_path,
                "title": chunk.title,
                "heading_path": chunk.heading_path,
                "anchor": chunk.anchor,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
            points.append(
                PointStruct(
                    id=_chunk_id_to_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[start : start + _UPSERT_BATCH_SIZE]
            self._client.upsert(collection_name=self._collection_name, points=batch)

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Return the k nearest chunks with cosine similarity scores."""
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        return [
            (_payload_to_chunk(hit.payload or {}), hit.score) for hit in results.points
        ]

    def all_chunks(self) -> list[Chunk]:
        """Load every stored chunk, sorted by chunk_id for determinism."""
        chunks: list[Chunk] = []
        offset: Any = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self._collection_name,
                limit=_SCROLL_BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(_payload_to_chunk(r.payload or {}) for r in records)
            if offset is None:
                break
        chunks.sort(key=lambda c: c.chunk_id)
        return chunks
