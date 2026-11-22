"""Tests for the indexing sub-package (store, pipeline integration).

All tests use an in-memory QdrantClient and a fake embedder so they never
touch the real model or a live Qdrant server.
"""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_project.indexing.store import QdrantStore, _chunk_id_to_point_id
from rag_project.ingestion.models import Chunk

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_COLLECTION = "test_collection"
_DIM = 4  # tiny dimension for test vectors


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_in_memory_store() -> QdrantStore:
    """Return a QdrantStore backed by an in-memory Qdrant client."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config=VectorParams(size=_DIM, distance=Distance.COSINE),
    )
    store = QdrantStore.__new__(QdrantStore)
    store._collection_name = _COLLECTION
    store._client = client
    return store


def _make_chunk(
    source_path: str = "tutorial.md",
    chunk_index: int = 0,
    text: str = "Some chunk text.",
    heading_path: str = "Tutorial",
    anchor: str | None = None,
    token_count: int = 10,
) -> Chunk:
    """Construct a minimal Chunk for testing."""
    return Chunk(
        chunk_id=f"{source_path}::{chunk_index:04d}",
        source_path=source_path,
        title="Tutorial",
        heading_path=heading_path,
        chunk_index=chunk_index,
        text=text,
        token_count=token_count,
        anchor=anchor,
    )


def _unit_vector(dim: int, hot: int) -> list[float]:
    """Return a unit vector with a 1.0 at position *hot* and 0.0 elsewhere."""
    v = [0.0] * dim
    v[hot] = 1.0
    return v


# ---------------------------------------------------------------------------
# Test: upsert is idempotent
# ---------------------------------------------------------------------------


def test_upsert_is_idempotent() -> None:
    """Running upsert twice must not create duplicate points.

    Qdrant upsert semantics: if a point with the same ID already exists it is
    overwritten, so point count must remain the same after a second upsert.
    """
    store = _make_in_memory_store()
    chunks = [
        _make_chunk(chunk_index=0, text="Alpha chunk."),
        _make_chunk(chunk_index=1, text="Beta chunk."),
    ]
    vectors = [_unit_vector(_DIM, 0), _unit_vector(_DIM, 1)]

    store.upsert_chunks(chunks, vectors)
    count_after_first = store._client.count(collection_name=_COLLECTION).count

    store.upsert_chunks(chunks, vectors)
    count_after_second = store._client.count(collection_name=_COLLECTION).count

    assert count_after_first == 2
    assert count_after_second == 2, (
        "Re-upserting the same chunks must not increase point count "
        f"(got {count_after_second})"
    )


# ---------------------------------------------------------------------------
# Test: search returns the nearest chunk
# ---------------------------------------------------------------------------


def test_search_returns_nearest_chunk() -> None:
    """The top search result must be the chunk whose vector is closest to the query.

    We store three chunks with orthogonal unit vectors; the query matches the
    second chunk exactly, so it must rank first.
    """
    store = _make_in_memory_store()
    chunks = [
        _make_chunk(chunk_index=0, text="Chunk zero."),
        _make_chunk(chunk_index=1, text="Chunk one."),
        _make_chunk(chunk_index=2, text="Chunk two."),
    ]
    vectors = [
        _unit_vector(_DIM, 0),
        _unit_vector(_DIM, 1),
        _unit_vector(_DIM, 2),
    ]
    store.upsert_chunks(chunks, vectors)

    query_vector = _unit_vector(_DIM, 1)  # identical to chunk 1's vector
    results = store.search(query_vector, k=1)

    assert len(results) == 1
    top_chunk, top_score = results[0]
    assert top_chunk.chunk_id == "tutorial.md::0001", (
        f"Expected chunk index 1, got {top_chunk.chunk_id}"
    )
    assert top_score == pytest.approx(1.0, abs=1e-5), (
        f"Expected cosine score ≈ 1.0 for exact match, got {top_score}"
    )


# ---------------------------------------------------------------------------
# Test: payload round-trips into a Chunk
# ---------------------------------------------------------------------------


def test_payload_round_trips_into_chunk() -> None:
    """Every field stored in the Qdrant payload must be faithfully reconstructed.

    This verifies that ``search`` correctly deserialises the stored payload
    back into a :class:`~rag_project.ingestion.models.Chunk` with no data loss.
    """
    store = _make_in_memory_store()
    original = _make_chunk(
        source_path="advanced/guide.md",
        chunk_index=7,
        text="Detailed explanation of path parameters.",
        heading_path="Advanced Guide > Path Parameters",
        anchor="path-parameters",
        token_count=42,
    )
    vector = _unit_vector(_DIM, 3)
    store.upsert_chunks([original], [vector])

    results = store.search(vector, k=1)
    assert len(results) == 1
    reconstructed, _ = results[0]

    assert reconstructed.chunk_id == original.chunk_id
    assert reconstructed.source_path == original.source_path
    assert reconstructed.title == original.title
    assert reconstructed.heading_path == original.heading_path
    assert reconstructed.anchor == original.anchor
    assert reconstructed.chunk_index == original.chunk_index
    assert reconstructed.text == original.text
    assert reconstructed.token_count == original.token_count


# ---------------------------------------------------------------------------
# Test: point IDs are deterministic
# ---------------------------------------------------------------------------


def test_point_ids_are_deterministic() -> None:
    """The UUID5 point ID for a given chunk_id must always be the same.

    Two calls with the same chunk_id must produce the identical UUID string,
    and different chunk_ids must produce different UUID strings.
    """
    cid_a = "tutorial.md::0000"
    cid_b = "advanced/guide.md::0007"

    id_a1 = _chunk_id_to_point_id(cid_a)
    id_a2 = _chunk_id_to_point_id(cid_a)
    id_b = _chunk_id_to_point_id(cid_b)

    # Same input → same output
    assert id_a1 == id_a2

    # Different inputs → different outputs
    assert id_a1 != id_b

    # Must be a valid UUID string
    uuid.UUID(id_a1)  # raises ValueError if invalid
    uuid.UUID(id_b)

    # Must be version 5
    assert uuid.UUID(id_a1).version == 5
    assert uuid.UUID(id_b).version == 5
