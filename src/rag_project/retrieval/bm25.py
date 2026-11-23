"""BM25 keyword index over chunks."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from rag_project.ingestion.models import Chunk

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens, keeping identifiers whole."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    """In-memory BM25 (Okapi) index. Headings are indexed with the text."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self._chunks = list(chunks)
        corpus = [tokenize(f"{c.heading_path}\n{c.text}") for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        """Return up to k chunks with a positive BM25 score, best first."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(self._chunks)), key=lambda i: (-float(scores[i]), i))
        return [(self._chunks[i], float(scores[i])) for i in order[:k] if scores[i] > 0]
