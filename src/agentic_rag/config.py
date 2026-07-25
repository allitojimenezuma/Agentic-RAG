"""Pydantic BaseSettings for agentic-rag configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # LLM (OpenAI-compatible)
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"

    # Paths
    wiki_path: Path = Path("./wiki")
    raw_sources_path: Path = Path("./raw")
    agents_md_path: Path = Path("./AGENTS.md")

    # Agent runtime
    recursion_limit: int = 12
    hitl_enabled: bool = True

    # Logging
    log_level: str = "DEBUG"
    log_dir: Path | None = None  # None = console only

    # Retrieval (MVP: index-only)
    retrieval_mode: str = "index"
    vector_db_path: Path | None = None

    # MarkItDown
    markitdown_llm_describe_images: bool = False

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }
