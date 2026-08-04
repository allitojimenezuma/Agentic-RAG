"""Tests for io/wiki_io.py — wiki filesystem operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.io.wiki_io import (
    delete_page,
    list_pages,
    page_exists,
    read_page,
    read_page_with_frontmatter,
    write_page,
)
from agentic_rag.io.index_manager import read_index, write_index
from agentic_rag.schemas.wiki import Frontmatter, IndexEntry
from datetime import date


@pytest.fixture
def wiki(tmp_path: Path) -> Path:
    """Create a temporary wiki directory with sample index."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "entities").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n"
        "## Entities\n"
        "- [[Python]] - High-level programming language | Source: manual | Updated: 2025-01-01\n\n"
        "## Concepts\n\n"
        "## Sources\n"
        "- [Cv](sources/cv.md) - Ingested: 2026-07-20\n\n"
        "## Comparisons\n"
    )
    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


class TestListPages:
    def test_lists_md_files(self, wiki: Path) -> None:
        (wiki / "entities" / "python.md").write_text("# Python")
        (wiki / "concepts" / "ml.md").write_text("# ML")

        pages = list_pages(wiki)
        names = [p.name for p in pages]
        assert "python.md" in names
        assert "ml.md" in names

    def test_excludes_index_and_log(self, wiki: Path) -> None:
        (wiki / "entities" / "python.md").write_text("# Python")

        pages = list_pages(wiki)
        names = [p.name for p in pages]
        assert "index.md" not in names
        assert "log.md" not in names

    def test_empty_wiki(self, wiki: Path) -> None:
        pages = list_pages(wiki)
        assert pages == []


class TestReadWritePage:
    def test_write_and_read_roundtrip(self, wiki: Path) -> None:
        content = "# Python\n\nA programming language."
        write_page(wiki, "entities/python", content)

        result = read_page(wiki, "entities/python")
        assert result == content

    def test_write_creates_parent_dirs(self, wiki: Path) -> None:
        write_page(wiki, "new-category/test", "content")
        assert (wiki / "new-category" / "test.md").is_file()

    def test_write_with_frontmatter(self, wiki: Path) -> None:
        fm = Frontmatter(
            slug="python",
            type="entity",
            title="Python",
            sources=["manual"],
            updated=date(2025, 1, 1),
            tags=["language"],
        )
        write_page(wiki, "entities/python", "A language.", frontmatter=fm)

        raw = read_page(wiki, "entities/python")
        assert raw.startswith("---")
        assert "slug: python" in raw
        assert "A language." in raw

    def test_read_nonexistent_raises(self, wiki: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_page(wiki, "entities/nonexistent")

    def test_read_page_with_frontmatter(self, wiki: Path) -> None:
        fm = Frontmatter(
            slug="test",
            type="entity",
            title="Test",
            sources=["src.md"],
            updated=date(2025, 6, 15),
            tags=[],
        )
        write_page(wiki, "entities/test", "Body content.", frontmatter=fm)

        parsed_fm, body = read_page_with_frontmatter(wiki, "entities/test")
        assert parsed_fm.slug == "test"
        assert parsed_fm.type == "entity"
        assert "Body content." in body


class TestDeletePage:
    def test_delete_existing(self, wiki: Path) -> None:
        write_page(wiki, "entities/python", "content")
        assert page_exists(wiki, "entities/python")

        delete_page(wiki, "entities/python")
        assert not page_exists(wiki, "entities/python")

    def test_delete_nonexistent_no_error(self, wiki: Path) -> None:
        # Should not raise
        delete_page(wiki, "entities/nonexistent")


class TestPageExists:
    def test_exists(self, wiki: Path) -> None:
        write_page(wiki, "entities/python", "content")
        assert page_exists(wiki, "entities/python") is True

    def test_not_exists(self, wiki: Path) -> None:
        assert page_exists(wiki, "entities/nonexistent") is False


class TestSlugValidation:
    """Test _validate_slug rejects dangerous slugs."""

    def test_dotdot_rejected(self, wiki: Path) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            read_page(wiki, "../../etc/passwd")

    def test_dotdot_in_middle_rejected(self, wiki: Path) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            write_page(wiki, "entities/../../etc/passwd", "evil")

    def test_absolute_path_rejected(self, wiki: Path) -> None:
        with pytest.raises(ValueError, match="absolute path"):
            delete_page(wiki, "/etc/passwd")

    def test_empty_slug_rejected(self, wiki: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            read_page(wiki, "")

    def test_valid_nested_slug_accepted(self, wiki: Path) -> None:
        write_page(wiki, "entities/python", "content")
        assert page_exists(wiki, "entities/python")

    def test_simple_slug_accepted(self, wiki: Path) -> None:
        write_page(wiki, "python", "content")
        assert page_exists(wiki, "python")


class TestIndexDisplayNames:
    """Test that index preserves original display names."""

    def test_display_name_preserved_on_parse(self, wiki: Path) -> None:
        index = read_index(wiki)
        python_entry = index.categories["entities"][0]
        assert python_entry.display_name == "Python"

    def test_source_entry_uses_sources_prefix(self, wiki: Path) -> None:
        index = read_index(wiki)
        source_entry = index.categories["sources"][0]
        assert source_entry.slug == "sources/cv"

    def test_write_and_read_preserves_display_name(self, wiki: Path) -> None:
        index = read_index(wiki)
        write_index(wiki, index)
        read_back = read_index(wiki)
        assert read_back.categories["entities"][0].display_name == "Python"

    def test_source_path_format_in_output(self, wiki: Path) -> None:
        """Verify _format_entry links source entries at their full slug path."""
        from agentic_rag.io.index_manager import _format_entry
        entry = IndexEntry(
            slug="sources/cv",
            summary="Cv",
            type="source",
            sources=["cv.pdf"],
            updated=date(2026, 7, 20),
            display_name="Cv",
        )
        line = _format_entry(entry)
        # entry.slug already includes the sources/ prefix — no double prefix
        assert "(sources/cv.md)" in line
        assert "sources/sources" not in line

    def test_section_to_type_entities(self) -> None:
        from agentic_rag.io.index_manager import _SECTION_TO_TYPE
        assert _SECTION_TO_TYPE["entities"] == "entity"
        assert _SECTION_TO_TYPE["concepts"] == "concept"
        assert _SECTION_TO_TYPE["sources"] == "source"
        assert _SECTION_TO_TYPE["comparisons"] == "comparison"
