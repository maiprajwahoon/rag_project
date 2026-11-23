"""Gemini chat client with rate-limit backoff."""

from __future__ import annotations

import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from rag_project.config import get_settings

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def to_text(content: str | list[Any]) -> str:
    """Flatten a LangChain message content (string or block list) to text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


class GeminiClient:
    """LLMClient implementation backed by Google Gemini."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_attempts: int = 4,
        base_delay: float = 5.0,
    ) -> None:
        key = get_settings().llm_api_key.get_secret_value()
        if not key:
            raise ValueError("LLM_API_KEY is empty. Set it in .env")
        self._llm = ChatGoogleGenerativeAI(model=model, google_api_key=SecretStr(key))
        self._max_attempts = max_attempts
        self._base_delay = base_delay

    def complete(self, system: str, user: str) -> str:
        """Send a system and user message, retrying on rate limits."""
        messages = [("system", system), ("human", user)]
        for attempt in range(self._max_attempts):
            try:
                return to_text(self._llm.invoke(messages).content)
            except Exception as exc:
                limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                if not limited or attempt == self._max_attempts - 1:
                    raise
                time.sleep(self._base_delay * (2**attempt))
        raise RuntimeError("unreachable")
