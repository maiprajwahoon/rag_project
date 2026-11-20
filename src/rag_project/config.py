"""Application configuration powered by pydantic-settings."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RAG project.

    Values are loaded from environment variables and, when present, from a
    ``.env`` file located in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    llm_provider: str = ""
    """Name of the LLM provider (e.g. ``"openai"``, ``"anthropic"``)."""

    llm_api_key: SecretStr = SecretStr("")
    """API key for the configured LLM provider.  Stored as a ``SecretStr``
    so it is never leaked in logs or ``repr()`` output."""

    qdrant_url: str = "http://localhost:6333"
    """Base URL of the Qdrant vector-database instance."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    The settings object is created once and reused for the lifetime of the
    process, avoiding repeated reads of the ``.env`` file.
    """
    return Settings()
