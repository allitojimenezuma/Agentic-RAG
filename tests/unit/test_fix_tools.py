"""Unit tests for fix tools: add_frontmatter, fix_link, append_related_section, edit_wiki_page."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.io.wiki_io import write_page
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools import fix_tools
from agentic_rag.tools.fix_tools import (
    add_frontmatter,
    append_related_section,
    edit_wiki_page,
    fix_link,
)
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.model import load_wiki


def _write_raw(wiki_path: Path, slug: str, content: str) -> None:
    """Write a page without frontmatter."""
    write_page(wiki_path, slug, content)


def _write_with_frontmatter(wiki_path: Path, slug: str, content: str) -> None:
    fm = Frontmatter(
        slug=slug, type="entity", title="Python",
        sources=["manual"], updated=date(2025, 1, 1),
    )
    write_page(wiki_path, slug, serialize_frontmatter(fm) + content)


class TestRemovedTools:
    def test_shell_and_index_entry_tools_removed(self) -> None:
        """execute_command / run_command / remove_index_entry no longer exist."""
        for name in ("execute_command", "run_command", "remove_index_entry"):
            assert not hasattr(fix_tools, name), f"{name} should have been removed"


class TestAddFrontmatter:
    def test_adds_valid_frontmatter(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        _write_raw(tmp_path, "entities/python", "# Python\n\nA language.")

        result = add_frontmatter.invoke({
            "slug": "entities/python",
            "title": "Python",
            "page_type": "entity",
        })
        assert "Added" in result

        raw = (tmp_path / "entities/python.md").read_text(encoding="utf-8")
        assert raw.startswith("---")
        # Re-load via the Wiki model: frontmatter must parse.
        wiki = load_wiki(tmp_path)
        page = wiki.by_slug["entities/python"]
        assert page.fm.slug == "entities/python"
        assert page.fm.type == "entity"
        assert page.fm.title == "Python"
        assert "A language." in raw

    def test_errors_when_frontmatter_present(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        _write_with_frontmatter(tmp_path, "entities/python", "# Python\n\nA language.")

        result = add_frontmatter.invoke({
            "slug": "entities/python",
            "title": "Python",
            "page_type": "entity",
        })
        assert "Error" in result
        assert "already has frontmatter" in result

    def test_errors_on_missing_page(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        result = add_frontmatter.invoke({
            "slug": "entities/nope",
            "title": "Nope",
            "page_type": "entity",
        })
        assert "Page not found" in result


class TestFixLink:
    def test_replaces_plain_and_alias_forms(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        # Target page must exist — fix_link refuses to create a NEW dangling link.
        _write_with_frontmatter(tmp_path, "concepts/new", "# New\n\nContent.")
        _write_with_frontmatter(
            tmp_path, "entities/python",
            "# Python\n\nSee [[old]] and [[old|alias text]] and [[old]] again.",
        )

        result = fix_link.invoke({
            "slug": "entities/python",
            "old_target": "old",
            "new_target": "concepts/new",
        })
        assert "Replaced 3" in result

        raw = (tmp_path / "entities/python.md").read_text(encoding="utf-8")
        assert "[[concepts/new]]" in raw
        assert "[[concepts/new|alias text]]" in raw
        assert "[[old]]" not in raw

    def test_returns_zero_when_no_links(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        _write_raw(tmp_path, "entities/python", "# Python\n\nNo links here.")
        _write_with_frontmatter(tmp_path, "concepts/new", "# New\n\nContent.")

        result = fix_link.invoke({
            "slug": "entities/python",
            "old_target": "missing",
            "new_target": "concepts/new",
        })
        assert "No links to 'missing' found" in result

    def test_errors_on_missing_page(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        result = fix_link.invoke({
            "slug": "entities/nope",
            "old_target": "a",
            "new_target": "b",
        })
        assert "Page not found" in result


class TestAppendRelatedSection:
    def test_appends_when_absent(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        _write_raw(tmp_path, "entities/python", "# Python\n\nA language.")

        result = append_related_section.invoke({
            "slug": "entities/python",
            "links": ["entities/a", "concepts/b"],
        })
        assert "Appended 2" in result

        raw = (tmp_path / "entities/python.md").read_text(encoding="utf-8")
        assert "## Related" in raw
        assert "- [[entities/a]]" in raw
        assert "- [[concepts/b]]" in raw

    def test_extends_existing_section(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        _write_raw(
            tmp_path, "entities/python",
            "# Python\n\n## Related\n\n- [[entities/existing]]\n\n## Other\n\nText.",
        )

        result = append_related_section.invoke({
            "slug": "entities/python",
            "links": ["concepts/new"],
        })
        assert "Appended 1" in result

        raw = (tmp_path / "entities/python.md").read_text(encoding="utf-8")
        assert raw.count("## Related") == 1
        assert "- [[entities/existing]]" in raw
        assert "- [[concepts/new]]" in raw
        # New links stay inside the Related section (before the next heading).
        related_block = raw.split("## Related", 1)[1].split("## Other", 1)[0]
        assert "- [[concepts/new]]" in related_block

    def test_errors_on_missing_page(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        result = append_related_section.invoke({
            "slug": "entities/nope",
            "links": ["entities/a"],
        })
        assert "Page not found" in result


class TestEditWikiPage:
    def test_still_replaces_text(self, tmp_path: Path) -> None:
        init_shared_tools(str(tmp_path))
        _write_raw(tmp_path, "entities/python", "# Python\n\nTypo here.")

        result = edit_wiki_page.invoke({
            "slug": "entities/python",
            "old_text": "Typo",
            "new_text": "Fixed",
        })
        assert "Replaced 1 occurrence" in result
        raw = (tmp_path / "entities/python.md").read_text(encoding="utf-8")
        assert "Fixed here." in raw
