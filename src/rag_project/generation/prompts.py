"""Versioned prompt loading from configs/prompts."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"


class PromptConfig(BaseModel):
    """A versioned prompt pair loaded from a TOML file."""

    version: str
    system: str
    user_template: str


def load_prompt(name: str = "answer_v1") -> PromptConfig:
    """Load a prompt file by name (without extension)."""
    with (PROMPTS_DIR / f"{name}.toml").open("rb") as handle:
        return PromptConfig.model_validate(tomllib.load(handle))
