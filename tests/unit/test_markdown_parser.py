"""Tests for io/markdown_parser.py — markdown parsing utilities."""

from __future__ import annotations

import pytest

from agentic_rag.io.markdown_parser import (
    extract_headings,
    extract_links,
    parse_frontmatter,
    serialize_frontmatter,
    slugify,
)
from agentic_rag.schemas.wiki import Frontmatter
from datetime import date


class TestExtractLinks:
    def test_simple_link(self) -> None:
        content = "See [[Python]] for details."
        links = extract_links(content)
        assert len(links) == 1
        assert links[0].target == "Python"
        assert links[0].alias is None

    def test_link_with_alias(self) -> None:
        content = "See [[Python|Py]] for details."
        links = extract_links(content)
        assert len(links) == 1
        assert links[0].target == "Python"
        assert links[0].alias == "Py"

    def test_multiple_links(self) -> None:
        content = "[[Python]] and [[Machine Learning]] and [[PyTorch|torch]]"
        links = extract_links(content)
        assert len(links) == 3
        assert links[0].target == "Python"
        assert links[1].target == "Machine Learning"
        assert links[2].target == "PyTorch"
        assert links[2].alias == "torch"

    def test_no_links(self) -> None:
        content = "No links here."
        links = extract_links(content)
        assert links == []

    def test_link_in_list(self) -> None:
        content = "- [[A]]\n- [[B|alias]]\n- plain text"
        links = extract_links(content)
        assert len(links) == 2


class TestExtractHeadings:
    def test_headings(self) -> None:
        content = "# Title\n\nSome text.\n\n## Subtitle\n\nMore text.\n\n### Deep"
        headings = extract_headings(content)
        assert len(headings) == 3
        assert headings[0].level == 1
        assert headings[0].text == "Title"
        assert headings[1].level == 2
        assert headings[1].text == "Subtitle"
        assert headings[2].level == 3
        assert headings[2].text == "Deep"

    def test_no_headings(self) -> None:
        content = "Just some text.\nNo headings."
        headings = extract_headings(content)
        assert headings == []


class TestFrontmatter:
    def test_parse_frontmatter(self) -> None:
        content = "---\nslug: test\ntype: entity\ntitle: Test\nsources:\n  - src.md\nupdated: 2025-01-15\ntags:\n  - tag1\n---\n\nBody."
        fm = parse_frontmatter(content)
        assert fm.slug == "test"
        assert fm.type == "entity"
        assert fm.title == "Test"
        assert fm.sources == ["src.md"]
        assert fm.updated == date(2025, 1, 15)
        assert fm.tags == ["tag1"]

    def test_parse_frontmatter_missing_delimiter(self) -> None:
        with pytest.raises(ValueError, match="does not start with"):
            parse_frontmatter("No frontmatter here.")

    def test_parse_frontmatter_malformed(self) -> None:
        with pytest.raises(ValueError, match="missing closing"):
            parse_frontmatter("---\nonly opening")

    def test_roundtrip(self) -> None:
        fm = Frontmatter(
            slug="python",
            type="entity",
            title="Python",
            sources=["manual"],
            updated=date(2025, 1, 1),
            tags=["language"],
        )
        serialized = serialize_frontmatter(fm)
        assert serialized.startswith("---\n")
        assert serialized.endswith("---\n")
        parsed = parse_frontmatter(serialized + "Body.")
        assert parsed.slug == "python"
        assert parsed.type == "entity"
        assert parsed.sources == ["manual"]


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("3D Gaussian Splatting") == "3d-gaussian-splatting"

    def test_unicode(self) -> None:
        result = slugify("Álvaro Jiménez")
        assert result == "alvaro-jimenez"

    def test_special_chars(self) -> None:
        assert slugify("Hello, World!") == "hello-world"

    def test_multiple_hyphens(self) -> None:
        assert slugify("a---b") == "a-b"

    def test_empty(self) -> None:
        assert slugify("") == ""

    def test_already_slug(self) -> None:
        assert slugify("machine-learning") == "machine-learning"
