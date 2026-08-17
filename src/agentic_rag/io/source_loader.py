"""MarkItDown wrapper for source ingestion: converts files to markdown."""

from __future__ import annotations

import logging
from pathlib import Path

from markitdown import MarkItDown

logger = logging.getLogger(__name__)


class SourceLoader:
    """Wraps markitdown.MarkItDown to convert source files to markdown."""

    def __init__(self) -> None:
        self._md = MarkItDown()

    def load(self, source: str) -> str:
        """Convert a source file (path or URL) to markdown.

        Returns markdown with a leading "# Source: <filename>" heading.
        """
        source_path = Path(source)
        if source_path.exists():
            if not source_path.is_file():
                raise ValueError(f"Source path is not a file: {source}")
        elif not source.startswith("http://") and not source.startswith("https://"):
            raise FileNotFoundError(f"Source file not found: {source}")

        result = self._md.convert(str(source))
        filename = source_path.name
        content = f"# Source: {filename}\n\n{result.text_content}"
        return content