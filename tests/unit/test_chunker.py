"""Unit tests for the pure source-section chunker (io/chunker.py)."""

from __future__ import annotations

import inspect

from agentic_rag.io.chunker import chunk_by_heading


class TestChunkByHeading:
    def test_splits_on_level2_headings(self):
        doc = "# Title\n## Section 1\nintro text\n## Section 2\ntail text"

        chunks = chunk_by_heading(doc)

        assert chunks == [
            "# Title\n## Section 1\nintro text",
            "## Section 1\n## Section 2\ntail text",
        ]

    def test_deeper_headings_split_too(self):
        doc = "## A\ntext a\n### Sub\ntext b"

        chunks = chunk_by_heading(doc)

        assert chunks == ["## A\ntext a", "## A\n### Sub\ntext b"]

    def test_breadcrumb_prepends_recent_h1_or_h2(self):
        doc = (
            "# Title\n"
            "## Section 1\n"
            "intro\n"
            "### Subsection A\n"
            "deep text\n"
            "## Section 2\n"
            "tail"
        )

        chunks = chunk_by_heading(doc)

        assert chunks[0] == "# Title\n## Section 1\nintro"
        assert chunks[1] == "## Section 1\n### Subsection A\ndeep text"
        assert chunks[2] == "## Section 1\n## Section 2\ntail"

    def test_max_chars_boundary_chunk_not_split_further(self):
        big_body = "x" * 5000
        doc = f"# T\n## Long Section\n{big_body}"

        chunks = chunk_by_heading(doc, max_chars=4000)

        assert len(chunks) == 1
        assert big_body in chunks[0]
        assert len(chunks[0]) > 4000

    def test_default_max_chars_is_4000(self):
        assert inspect.signature(chunk_by_heading).parameters["max_chars"].default == 4000

    def test_empty_input_returns_empty_list(self):
        assert chunk_by_heading("") == []
        assert chunk_by_heading("   \n\t ") == []

    def test_no_level2_heading_returns_single_chunk(self):
        doc = "# Title\nJust a paragraph, no sections."

        assert chunk_by_heading(doc) == [doc]
