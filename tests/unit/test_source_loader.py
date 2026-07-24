"""Tests for io/source_loader.py — MarkItDown wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestSourceLoader:
    """Tests for SourceLoader.load()."""

    def test_load_markdown_file(self, tmp_path: Path) -> None:
        """Loading a markdown file returns markdown with Source heading."""
        from agentic_rag.io.source_loader import SourceLoader

        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello World\n\nSome content.\n")

        settings = SimpleNamespace(
            markitdown_llm_describe_images=False,
        )

        with patch("markitdown.MarkItDown") as MockMD:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.text_content = "# Hello World\n\nSome content."
            mock_instance.convert.return_value = mock_result
            MockMD.return_value = mock_instance

            loader = SourceLoader(settings)
            result = loader.load(str(md_file))

        assert "# Source: test.md" in result
        assert "# Hello World" in result
        assert "Some content." in result

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent file raises an error."""
        from agentic_rag.io.source_loader import SourceLoader

        settings = SimpleNamespace(
            markitdown_llm_describe_images=False,
        )

        with patch("markitdown.MarkItDown") as MockMD:
            mock_instance = MagicMock()
            mock_instance.convert.side_effect = FileNotFoundError("No such file")
            MockMD.return_value = mock_instance

            loader = SourceLoader(settings)
            with pytest.raises(FileNotFoundError):
                loader.load(str(tmp_path / "nonexistent.md"))

    def test_load_with_llm_describe_images(self, tmp_path: Path) -> None:
        """When markitdown_llm_describe_images is True, OpenAI client is created."""
        from agentic_rag.io.source_loader import SourceLoader

        settings = SimpleNamespace(
            markitdown_llm_describe_images=True,
            openai_base_url="https://api.example.com/v1",
            openai_api_key="test-key",
            openai_model="gpt-4",
        )

        with patch("markitdown.MarkItDown") as MockMD, \
             patch("openai.OpenAI") as MockOpenAI:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.text_content = "content"
            mock_instance.convert.return_value = mock_result
            MockMD.return_value = mock_instance
            MockOpenAI.return_value = MagicMock()

            loader = SourceLoader(settings)

            MockOpenAI.assert_called_once_with(
                base_url="https://api.example.com/v1",
                api_key="test-key",
            )
            MockMD.assert_called_once()
