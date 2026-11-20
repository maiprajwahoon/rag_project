"""Tests for :mod:`rag_project.config`."""

from rag_project.config import Settings


def test_default_qdrant_url() -> None:
    """The default Qdrant URL should point to localhost:6333."""
    settings = Settings()
    assert settings.qdrant_url == "http://localhost:6333"


def test_api_key_not_exposed_in_repr() -> None:
    """The ``llm_api_key`` value must never appear in ``repr(settings)``."""
    secret: str = "super-secret-key-12345"
    settings = Settings(llm_api_key=secret)  # type: ignore[arg-type]
    assert secret not in repr(settings)
