"""Embedding utilities wrapping BAAI/bge-small-en-v1.5 via sentence-transformers."""

from sentence_transformers import SentenceTransformer

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_BATCH_SIZE = 32


class Embedder:
    """Wraps a SentenceTransformer model for document and query embedding.

    The underlying model (``BAAI/bge-small-en-v1.5``) is loaded once in
    ``__init__`` and reused for all subsequent calls, so construction is
    intentionally expensive but each ``embed_*`` call is cheap.

    Embeddings are L2-normalised, matching the retrieval setup expected by
    Qdrant with cosine similarity.
    """

    def __init__(self) -> None:
        """Load the BAAI/bge-small-en-v1.5 model from cache (or download once)."""
        self._model: SentenceTransformer = SentenceTransformer(_MODEL_NAME)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of passage texts in batches.

        No BGE instruction prefix is prepended — raw passage text is encoded
        directly.  Embeddings are L2-normalised.

        Args:
            texts: List of passage strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        vectors = self._model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string with the BGE instruction prefix.

        The prefix ``"Represent this sentence for searching relevant passages: "``
        is prepended to the query before encoding, as recommended by the BGE
        model authors for asymmetric retrieval.  The result is L2-normalised.

        Args:
            text: The raw query string.

        Returns:
            Embedding vector as a plain Python list of floats.
        """
        prefixed = _BGE_QUERY_PREFIX + text
        vector = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
