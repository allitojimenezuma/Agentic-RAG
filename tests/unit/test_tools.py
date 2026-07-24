"""Unit tests for all LangChain tools in agentic_rag.tools."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agentic_rag.io.index_manager import upsert_entry, read_index
from agentic_rag.io.log_manager import append_log
from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.io.wiki_io import write_page
from agentic_rag.schemas.wiki import Frontmatter, IndexEntry, LogEntry
from agentic_rag.tools.shared import read_index as tool_read_index
from agentic_rag.tools.shared import read_wiki_page as tool_read_wiki_page
from agentic_rag.tools.shared import search_index as tool_search_index
from agentic_rag.tools.ingest_tools import (
    read_source,
    create_page,
    update_page,
    delete_wiki_page,
    update_index,
    append_log as tool_append_log,
    flag_contradiction,
)
from agentic_rag.tools.query_tools import find_relevant_pages
from agentic_rag.tools.lint_tools import (
    read_all_pages,
    find_inbound_links,
    extract_concepts,
    write_lint_report,
)


# --- Helpers ---


def _create_test_page(wiki_path: Path, slug: str = "entities/python", content: str = "# Python\n\nA language.") -> None:
    """Create a test wiki page with frontmatter."""
    fm = Frontmatter(
        slug=slug,
        type="entity",
        title="Python",
        sources=["manual"],
        updated=date(2025, 1, 1),
    )
    full = serialize_frontmatter(fm) + content
    write_page(wiki_path, slug, full)


def _create_test_index(wiki_path: Path) -> None:
    """Populate the index with a sample entry."""
    entry = IndexEntry(
        slug="python",
        summary="High-level programming language",
        type="entity",
        sources=["manual"],
        updated=date(2025, 1, 1),
    )
    upsert_entry(wiki_path, entry)


def _create_test_log(wiki_path: Path) -> None:
    """Add a log entry."""
    entry = LogEntry(
        timestamp=date(2025, 1, 15).isoformat(),  # type: ignore[arg-type]
        op="ingest",
        title="manual",
        details="Created: [[Python]]",
    )
    append_log(wiki_path, entry)


# --- Shared tools ---


class TestReadIndex:
    def test_returns_formatted_index(self, wiki_path: Path) -> None:
        _create_test_index(wiki_path)
        result = tool_read_index.invoke({"wiki_path": str(wiki_path)})
        assert "python" in result
        assert "High-level programming language" in result

    def test_empty_index(self, wiki_path: Path) -> None:
        result = tool_read_index.invoke({"wiki_path": str(wiki_path)})
        assert "Index is empty" in result


class TestReadWikiPage:
    def test_reads_existing_page(self, wiki_path: Path) -> None:
        _create_test_page(wiki_path, "entities/python")
        result = tool_read_wiki_page.invoke({"wiki_path": str(wiki_path), "slug": "entities/python"})
        assert "Python" in result
        assert "A language" in result

    def test_errors_on_missing(self, wiki_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            tool_read_wiki_page.invoke({"wiki_path": str(wiki_path), "slug": "nonexistent"})


class TestSearchIndex:
    def test_keyword_match(self, wiki_path: Path) -> None:
        _create_test_index(wiki_path)
        result = tool_search_index.invoke({"wiki_path": str(wiki_path), "query": "python"})
        assert "python" in result

    def test_no_match(self, wiki_path: Path) -> None:
        result = tool_search_index.invoke({"wiki_path": str(wiki_path), "query": "rustlang"})
        assert "No results" in result


# --- Ingest tools ---


class TestReadSource:
    def test_loads_markdown_file(self, tmp_path: Path) -> None:
        src = tmp_path / "sample.md"
        src.write_text("# Hello\n\nThis is a test.")
        result = read_source.invoke({"source_path": str(src)})
        assert "Hello" in result
        assert "This is a test." in result


class TestCreatePage:
    def test_creates_new_page(self, wiki_path: Path) -> None:
        result = create_page.invoke({
            "wiki_path": str(wiki_path),
            "slug": "concepts/ai",
            "page_type": "concept",
            "title": "Artificial Intelligence",
            "content": "# AI\n\nMachine intelligence.",
            "sources": ["test.pdf"],
            "tags": ["ml"],
        })
        assert "Created" in result
        assert (wiki_path / "concepts/ai.md").is_file()

    def test_errors_if_exists(self, wiki_path: Path) -> None:
        _create_test_page(wiki_path, "entities/python")
        result = create_page.invoke({
            "wiki_path": str(wiki_path),
            "slug": "entities/python",
            "page_type": "entity",
            "title": "Python",
            "content": "duplicate",
        })
        assert "Error" in result
        assert "already exists" in result


class TestUpdatePage:
    def test_updates_existing_page(self, wiki_path: Path) -> None:
        _create_test_page(wiki_path, "entities/python")
        result = update_page.invoke({
            "wiki_path": str(wiki_path),
            "slug": "entities/python",
            "content": "# Python\n\nUpdated content.",
            "sources": ["updated.pdf"],
        })
        assert "Updated" in result
        content = (wiki_path / "entities/python.md").read_text()
        assert "Updated content." in content

    def test_errors_if_missing(self, wiki_path: Path) -> None:
        result = update_page.invoke({
            "wiki_path": str(wiki_path),
            "slug": "nonexistent",
            "content": "nope",
        })
        assert "Error" in result
        assert "does not exist" in result


class TestDeleteWikiPage:
    def test_deletes_existing(self, wiki_path: Path) -> None:
        _create_test_page(wiki_path, "entities/python")
        result = delete_wiki_page.invoke({"wiki_path": str(wiki_path), "slug": "entities/python"})
        assert "Deleted" in result
        assert not (wiki_path / "entities/python.md").is_file()


class TestUpdateIndex:
    def test_adds_entry(self, wiki_path: Path) -> None:
        result = update_index.invoke({
            "wiki_path": str(wiki_path),
            "slug": "python",
            "page_type": "entity",
            "summary": "Programming language",
            "sources": ["manual"],
        })
        assert "Index updated" in result
        idx = read_index(wiki_path)
        entries = idx.categories.get("entities", [])
        assert any(e.slug == "python" for e in entries)


class TestAppendLog:
    def test_appends_entry(self, wiki_path: Path) -> None:
        result = tool_append_log.invoke({
            "wiki_path": str(wiki_path),
            "op": "ingest",
            "title": "sample.md",
            "details": "Created: [[Python]]",
        })
        assert "Log entry appended" in result
        content = (wiki_path / "log.md").read_text()
        assert "ingest" in content
        assert "sample.md" in content


class TestFlagContradiction:
    def test_returns_contradiction_details(self, wiki_path: Path) -> None:
        result = flag_contradiction.invoke({
            "wiki_path": str(wiki_path),
            "page_slug": "entities/python",
            "existing_claim": "Python is interpreted",
            "new_claim": "Python can be compiled",
            "proposed_resolution": "Clarify both cases",
        })
        assert "CONTRADICTION FLAGGED" in result
        assert "entities/python" in result
        assert "HITL" in result


# --- Query tools ---


class TestFindRelevantPages:
    def test_finds_direct_matches(self, wiki_path: Path) -> None:
        _create_test_index(wiki_path)
        _create_test_page(wiki_path, "entities/python")
        result = find_relevant_pages.invoke({"wiki_path": str(wiki_path), "query": "python"})
        assert "python" in result
        assert "direct match" in result

    def test_traverses_links(self, wiki_path: Path) -> None:
        _create_test_index(wiki_path)
        # Page at root slug (matches index slug "python") with link to ml-ai
        fm = Frontmatter(
            slug="python", type="entity", title="Python",
            sources=[], updated=date(2025, 1, 1),
        )
        write_page(wiki_path, "python", serialize_frontmatter(fm) + "# Python\n\nSee also [[ml-ai]].")
        # Target page exists
        fm2 = Frontmatter(
            slug="ml-ai", type="concept", title="ML-AI",
            sources=[], updated=date(2025, 1, 1),
        )
        write_page(wiki_path, "ml-ai", serialize_frontmatter(fm2) + "# ML-AI\n")

        result = find_relevant_pages.invoke({"wiki_path": str(wiki_path), "query": "python"})
        assert "python" in result
        assert "ml-ai" in result
        assert "via links" in result

    def test_no_results(self, wiki_path: Path) -> None:
        result = find_relevant_pages.invoke({"wiki_path": str(wiki_path), "query": "rust"})
        assert "No pages found" in result


# --- Lint tools ---


class TestReadAllPages:
    def test_returns_all_pages(self, wiki_path: Path) -> None:
        _create_test_page(wiki_path, "entities/python", "# Python\n\nA language.")
        _create_test_page(wiki_path, "concepts/ml", "# ML\n\nMachine learning.")
        result = read_all_pages.invoke({"wiki_path": str(wiki_path)})
        assert "python" in result
        assert "ml" in result

    def test_empty_wiki(self, wiki_path: Path) -> None:
        result = read_all_pages.invoke({"wiki_path": str(wiki_path)})
        assert "No wiki pages" in result


class TestFindInboundLinks:
    def test_finds_linking_pages(self, wiki_path: Path) -> None:
        fm = Frontmatter(
            slug="entities/python", type="entity", title="Python",
            sources=[], updated=date(2025, 1, 1),
        )
        write_page(wiki_path, "entities/python", serialize_frontmatter(fm) + "# Python\n\nSee [[concepts/ml]].")
        fm2 = Frontmatter(
            slug="concepts/ml", type="concept", title="ML",
            sources=[], updated=date(2025, 1, 1),
        )
        write_page(wiki_path, "concepts/ml", serialize_frontmatter(fm2) + "# ML\n")

        result = find_inbound_links.invoke({"wiki_path": str(wiki_path), "slug": "concepts/ml"})
        assert "entities/python" in result

    def test_no_inbound_links(self, wiki_path: Path) -> None:
        _create_test_page(wiki_path, "entities/python")
        result = find_inbound_links.invoke({"wiki_path": str(wiki_path), "slug": "entities/python"})
        assert "No pages link" in result
        assert "orphan" in result


class TestExtractConcepts:
    def test_extracts_headings_and_links(self) -> None:
        content = "# Title\n\n## Section\n\nSee [[other-page]] and [[another|alias]]."
        result = extract_concepts.invoke({"content": content})
        assert "# Title" in result
        assert "## Section" in result
        assert "[[other-page]]" in result
        assert "alias" in result

    def test_empty_content(self) -> None:
        result = extract_concepts.invoke({"content": "Just some text."})
        assert "No concepts found" in result


class TestWriteLintReport:
    def test_creates_report_file(self, wiki_path: Path) -> None:
        result = write_lint_report.invoke({
            "wiki_path": str(wiki_path),
            "report": "# Lint Report\n\nAll good.",
        })
        assert "Lint report written" in result
        today = date.today().isoformat()
        report_path = wiki_path / f"lint-report-{today}.md"
        assert report_path.is_file()
        assert "All good." in report_path.read_text()
