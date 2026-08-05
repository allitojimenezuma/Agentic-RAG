"""LLM helper — creates ChatOpenAI with explicit base_url/api_key."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI


def get_model(settings: Any) -> ChatOpenAI:
    """Create a ChatOpenAI instance from settings.

    Uses explicit base_url and api_key to avoid provider auto-detection issues.
    """
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )
