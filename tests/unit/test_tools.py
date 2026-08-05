"""Unit tests for the LangChain tools used by the production agents."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agentic_rag.io.index import write_index
from agentic_rag.io.log import append_log
from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.io.wiki_io import write_page
from agentic_rag.schemas.wiki import Frontmatter, Index, IndexEntry, LogEntry
from agentic_rag.tools.shared import get_index_summary, init_shared_tools
from agentic_rag.tools.ingest_tools import (
    read_source,
    create_page,
    update_page,
    delete_wiki_page,
    append_log as tool_append_log,
    flag_contradiction,
)
from agentic_rag.tools.lint_tools import write_lint_report


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
    """Populate index.md with a sample entry (derived-view style)."""
    index = Index(categories={
        "entities": [
            IndexEntry(
                slug="python",
                summary="High-level programming language",
                type="entity",
                sources=["manual"],
                updated=date(2025, 1, 1),
            )
        ]
    })
    write_index(wiki_path, index)


def _create_test_log(wiki_path: Path) -> None:
    """Add a log entry."""
    entry = LogEntry(
        timestamp=date(2025, 1, 15).isoformat(),  # type: ignore[arg-type]
        op="ingest",
        title="manual",
        details="Created: [[Python]]",
    )
    append_log(wiki_path, entry)


# --- Shared helpers ---


class TestGetIndexSummary:
    def test_reads_existing_index(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _create_test_index(wiki_path)
        result = get_index_summary(wiki_path)
        assert "Python" in result
        assert "High-level programming language" in result

    def test_empty_index(self, wiki_path: Path) -> None:
        result = get_index_summary(wiki_path)
        # Empty index has headers but no entries — still valid content.
        assert "## Entities" in result or "Index empty" in result

    def test_missing_file(self, tmp_path: Path) -> None:
        result = get_index_summary(tmp_path / "nonexistent")
        assert "Index not found" in result


# --- Navigation error handling (nav.wiki_read_page returns errors, never raises) ---


class TestNavWikiReadPageErrors:
    """nav.wiki_read_page must RETURN error strings (never raise) so the agent can recover."""

    def test_missing_page_returns_error_not_raise(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        from agentic_rag.tools.nav import wiki_read_page as nav_read
        result = nav_read.invoke({"slug": "nonexistent"})
        assert "Error: Wiki page not found: nonexistent" in result
        assert "wiki_scan" in result

    def test_wrong_directory_suggests_correct_slug(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _create_test_page(wiki_path, "entities/spec-driven-subagent-harness")
        from agentic_rag.tools.nav import wiki_read_page as nav_read
        result = nav_read.invoke({"slug": "concepts/spec-driven-subagent-harness"})
        assert "Error: Wiki page not found: concepts/spec-driven-subagent-harness" in result
        assert "entities/spec-driven-subagent-harness" in result

    def test_section_path_missing_page_returns_error(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        from agentic_rag.tools.nav import wiki_read_page as nav_read
        result = nav_read.invoke({"slug": "nonexistent", "section": "History"})
        assert result.startswith("Error: Wiki page not found: nonexistent")

    def test_read_source_missing_file_returns_error(self, wiki_path: Path, tmp_path: Path) -> None:
        init_shared_tools(wiki_path)
        result = read_source.invoke({"source_path": str(tmp_path / "missing.pdf")})
        assert result.startswith("Error: could not read source")

    def test_delete_wiki_page_missing_returns_error(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        result = delete_wiki_page.invoke({"slug": "nonexistent"})
        assert result == "Error: Wiki page not found: nonexistent"


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
        init_shared_tools(wiki_path)
        result = create_page.invoke({
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
        init_shared_tools(wiki_path)
        _create_test_page(wiki_path, "entities/python")
        result = create_page.invoke({
            "slug": "entities/python",
            "page_type": "entity",
            "title": "Python",
            "content": "duplicate",
        })
        assert "Error" in result
        assert "already exists" in result


class TestUpdatePage:
    def test_updates_existing_page(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _create_test_page(wiki_path, "entities/python")
        result = update_page.invoke({
            "slug": "entities/python",
            "content": "# Python\n\nUpdated content.",
            "sources": ["updated.pdf"],
        })
        assert "Updated" in result
        content = (wiki_path / "entities/python.md").read_text()
        assert "Updated content." in content

    def test_errors_if_missing(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        result = update_page.invoke({
            "slug": "nonexistent",
            "content": "nope",
        })
        assert "Error" in result
        assert "does not exist" in result


class TestDeleteWikiPage:
    def test_deletes_existing(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _create_test_page(wiki_path, "entities/python")
        result = delete_wiki_page.invoke({"slug": "entities/python"})
        assert "Deleted" in result
        assert not (wiki_path / "entities/python.md").is_file()


class TestAppendLog:
    def test_appends_entry(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        result = tool_append_log.invoke({
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
        init_shared_tools(wiki_path)
        result = flag_contradiction.invoke({
            "page_slug": "entities/python",
            "existing_claim": "Python is interpreted",
            "new_claim": "Python can be compiled",
            "proposed_resolution": "Clarify both cases",
        })
        assert "CONTRADICTION FLAGGED" in result
        assert "entities/python" in result
        assert "HITL" in result


# --- Lint tools ---


class TestWriteLintReport:
    def test_creates_report_file(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        result = write_lint_report.invoke({
            "report": "# Lint Report\n\nAll good.",
        })
        assert "Lint report written" in result
        today = date.today().isoformat()
        report_path = wiki_path / f"lint-report-{today}.md"
        assert report_path.is_file()
        assert "All good." in report_path.read_text()
