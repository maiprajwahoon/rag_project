"""Qdrant vector store wrapper for chunk upsert and similarity search."""

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from rag_project.config import get_settings
from rag_project.ingestion.models import Chunk

_UPSERT_BATCH_SIZE = 128
_NAMESPACE_UUID = uuid.UUID(
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
)  # uuid.NAMESPACE_URL


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Derive a deterministic UUID5 point ID from a chunk's string ID.

    Args:
        chunk_id: The stable string ID of the chunk (e.g. ``"tutorial.md::0003"``).

    Returns:
        A UUID string suitable for use as a Qdrant point ID.
    """
    return str(uuid.uuid5(_NAMESPACE_UUID, chunk_id))


class QdrantStore:
    """Wraps a Qdrant collection for chunk storage and retrieval.

    The collection name is supplied at construction time so the same class
    can serve multiple index namespaces.  The Qdrant URL is taken from
    :func:`~rag_project.config.get_settings` on each instantiation.

    Args:
        collection_name: Name of the Qdrant collection to operate on.
            Defaults to ``"fastapi_docs"``.
    """

    def __init__(self, collection_name: str = "fastapi_docs") -> None:
        """Initialise the store with a Qdrant client and collection name.

        Args:
            collection_name: Name of the Qdrant collection.
        """
        self._collection_name = collection_name
        self._client = QdrantClient(url=get_settings().qdrant_url)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, vector_size: int = 384) -> None:
        """Create the collection with cosine distance if it does not yet exist.

        Calling this method on an already-existing collection is a no-op.

        Args:
            vector_size: Dimensionality of the embedding vectors (default 384
                for ``BAAI/bge-small-en-v1.5``).
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> None:
        """Upsert chunks with their embedding vectors into the collection.

        Each chunk is stored as a Qdrant point whose payload contains the
        full chunk text and all metadata fields.  Point IDs are deterministic
        UUID5 values derived from ``chunk_id``, so re-upserting the same
        chunks is idempotent.

        Args:
            chunks: Sequence of :class:`~rag_project.ingestion.models.Chunk`
                objects to store.
            vectors: Corresponding embedding vectors (must be the same length
                as *chunks*).
        """
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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Return the *k* nearest chunks and their cosine similarity scores.

        Args:
            query_vector: Embedding of the query, produced by
                :meth:`~rag_project.indexing.embedder.Embedder.embed_query`.
            k: Number of results to return (default 5).

        Returns:
            List of ``(Chunk, score)`` tuples ordered by descending similarity.
        """
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        output: list[tuple[Chunk, float]] = []
        for hit in results.points:
            payload = hit.payload or {}
            chunk = Chunk(
                chunk_id=str(payload["chunk_id"]),
                source_path=str(payload["source_path"]),
                title=str(payload["title"]),
                heading_path=str(payload["heading_path"]),
                anchor=payload.get("anchor"),  # type: ignore[arg-type]
                chunk_index=int(payload["chunk_index"]),  # type: ignore[arg-type]
                text=str(payload["text"]),
                token_count=int(payload["token_count"]),  # type: ignore[arg-type]
            )
            output.append((chunk, hit.score))
        return output
