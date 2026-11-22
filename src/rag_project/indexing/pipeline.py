"""Indexing pipeline: load → chunk → embed → upsert into Qdrant."""

from pathlib import Path

from rag_project.indexing.embedder import Embedder
from rag_project.indexing.store import QdrantStore
from rag_project.ingestion.chunker import chunk_documents
from rag_project.ingestion.loader import load_documents

_VECTOR_SIZE = 384


def build_index(
    docs_root: Path,
    docs_src_root: Path | None = None,
    collection_name: str = "fastapi_docs",
) -> None:
    """Load, chunk, embed, and upsert all documents under *docs_root*.

    Re-running this function on an already-indexed collection is safe: Qdrant
    upserts are idempotent because point IDs are deterministic UUID5 values
    derived from each chunk's stable ``chunk_id``.

    Steps:
        1. Load ``*.md`` documents from *docs_root* (skipping excluded files).
        2. Split each document into token-bounded chunks.
        3. Embed chunks in batches using ``BAAI/bge-small-en-v1.5``.
        4. Ensure the target collection exists.
        5. Upsert all chunk–vector pairs.

    Args:
        docs_root: Root directory containing the markdown documentation files.
        docs_src_root: Optional path to the ``docs_src`` directory used to
            resolve ``{* ... *}`` include directives during loading.
        collection_name: Qdrant collection to write into (default
            ``"fastapi_docs"``).
    """
    docs = load_documents(docs_root, docs_src_root=docs_src_root)
    chunks = chunk_documents(docs)

    embedder = Embedder()
    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)

    store = QdrantStore(collection_name=collection_name)
    store.ensure_collection(vector_size=_VECTOR_SIZE)
    store.upsert_chunks(chunks, vectors)

    print(
        f"Indexed {len(docs)} documents, {len(chunks)} chunks "
        f"→ collection '{collection_name}'."
    )


def main() -> None:
    """Entry point for ``python -m rag_project.indexing.pipeline``.

    Expects the FastAPI English docs to be present at the conventional path
    ``data/raw/fastapi/docs/en/docs`` relative to the current working directory.
    """
    cwd = Path.cwd()
    docs_root = cwd / "data" / "raw" / "fastapi" / "docs" / "en" / "docs"
    docs_src_root = cwd / "data" / "raw" / "fastapi" / "docs_src"
    build_index(docs_root=docs_root, docs_src_root=docs_src_root)


if __name__ == "__main__":
    main()
