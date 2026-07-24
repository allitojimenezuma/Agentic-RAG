"""MarkItDown wrapper for source ingestion: converts files to markdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class SourceLoader:
    """Wraps markitdown.MarkItDown to convert source files to markdown."""

    def __init__(self, settings: Any) -> None:
        """Initialize with optional LLM client for image description.

        Args:
            settings: Settings object with openai_base_url, openai_api_key,
                      openai_model, and markitdown_llm_describe_images fields.
        """
        from markitdown import MarkItDown

        kwargs: dict[str, Any] = {}
        if getattr(settings, "markitdown_llm_describe_images", False):
            from openai import OpenAI

            kwargs["llm_client"] = OpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
            kwargs["llm_model"] = settings.openai_model
        self._md = MarkItDown(**kwargs)

    def load(self, source: str) -> str:
        """Convert a source file (path or URL) to markdown.

        Returns markdown with a leading "# Source: <filename>" heading.
        """
        result = self._md.convert(str(source))
        filename = Path(source).name
        return f"# Source: {filename}\n\n{result.text_content}"
