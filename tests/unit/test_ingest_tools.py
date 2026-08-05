"""Unit tests for ingest tools' frontmatter handling (DEFECT 1: double frontmatter).

create_page/update_page write their own frontmatter; if the passed content
already embeds a YAML block, the tool must strip it so files never end up
with two frontmatter blocks.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.io.wiki_io import read_page_with_frontmatter, write_page
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools.ingest_tools import (
    _strip_embedded_frontmatter,
    create_page,
    update_page,
)
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.model import load_wiki

DEFAULT_WIKI_PATH = Path("./wiki")

# Exact shape from the production double-frontmatter log: content starts with a
# full --- frontmatter block, then the real body.
EMBEDDED_FM_CONTENT = (
    "---\n"
    "slug: pi\n"
    "type: entity\n"
    "title: Pi\n"
    "sources: []\n"
    "---\n"
    "# Pi\n\nPi is a mathematical constant."
)


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


class TestStripEmbeddedFrontmatter:
    def test_no_frontmatter_returns_content_unchanged(self) -> None:
        content = "# Pi\n\nPlain body."
        assert _strip_embedded_frontmatter(content) == content

    def test_leading_newlines_still_detected(self) -> None:
        content = "\n\n---\nslug: pi\n---\n# Pi"
        assert _strip_embedded_frontmatter(content) == "# Pi"

    def test_no_closing_delimiter_returns_content_unchanged(self) -> None:
        content = "---\nslug: pi\ntype: entity\n# Pi\n\nNo closing delimiter."
        assert _strip_embedded_frontmatter(content) == content

    def test_missing_frontmatter_returns_content_unchanged(self) -> None:
        content = "---\n# Pi\n\nBody."  # opening only, no closing ---
        assert _strip_embedded_frontmatter(content) == content


class TestCreatePageFrontmatterStrip:
    def test_strips_embedded_frontmatter_single_fm_result(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "entities/pi",
                "page_type": "entity",
                "title": "Pi",
                "content": EMBEDDED_FM_CONTENT,
                "sources": ["manual"],
            })
            assert "Created" in result
            raw = (wiki_path / "entities/pi.md").read_text()
            assert raw.count("---") == 2  # exactly ONE frontmatter block
            assert "slug: entities/pi" in raw  # the tool's own fm
            assert "slug: pi" not in raw  # embedded fm gone
            fm, body = read_page_with_frontmatter(wiki_path, "entities/pi")
            assert fm.type == "entity"
            assert "Pi is a mathematical constant." in body
            wiki = load_wiki(wiki_path)
            assert wiki.by_slug["entities/pi"].fm.type == "entity"
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_clean_body_only_content_unchanged(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "concepts/ai",
                "page_type": "concept",
                "title": "AI",
                "content": "# AI\n\nMachine intelligence.",
                "sources": ["test.pdf"],
            })
            assert "Created" in result
            raw = (wiki_path / "concepts/ai.md").read_text()
            assert raw.count("---") == 2  # single fm, body untouched
            assert "# AI" in raw
            assert "Machine intelligence." in raw
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_no_closing_delimiter_content_unchanged(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            content = "---\nslug: pi\ntype: entity\n# Pi\n\nBroken block, no closing ---."
            result = create_page.invoke({
                "slug": "entities/pi",
                "page_type": "entity",
                "title": "Pi",
                "content": content,
            })
            assert "Created" in result
            raw = (wiki_path / "entities/pi.md").read_text()
            # no strip: the embedded opening delimiter survives in the body verbatim
            fm, body = read_page_with_frontmatter(wiki_path, "entities/pi")
            assert body.startswith("---\nslug: pi")
            assert "Broken block, no closing" in body
            assert raw.count("slug: pi") == 1
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_restores_default_wiki_path(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        try:
            create_page.invoke({
                "slug": "entities/pi",
                "page_type": "entity",
                "title": "Pi",
                "content": "# Pi",
            })
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)


class TestCreatePageValidation:
    """create_page guards against the slug/title/type mistakes seen in the
    2026-08-05 ingest run (broken links, wrong type, non-ASCII slugs)."""

    def test_normalizes_non_ascii_slug(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "entities/málaga",
                "page_type": "entity",
                "title": "Málaga",
                "content": "# Málaga\n\nA city in southern Spain.",
            })
            assert "Created page: entities/malaga" in result
            assert (wiki_path / "entities/malaga.md").is_file()
            assert not (wiki_path / "entities/málaga.md").exists()
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_rejects_unknown_page_type(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "entities/alvaro",
                "page_type": "person",
                "title": "Álvaro Jiménez",
                "content": "# Álvaro Jiménez",
            })
            assert "Error" in result
            assert "Unknown page_type" in result
            assert "entity" in result  # suggests the correct types
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_rejects_type_directory_mismatch(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "entities/alvaro-jimenez-martinez",
                "page_type": "concept",
                "title": "Álvaro Jiménez Martínez",
                "content": "# Álvaro Jiménez Martínez",
            })
            assert "Error" in result
            assert "must use page_type 'entity'" in result
            assert not (wiki_path / "entities/alvaro-jimenez-martinez.md").exists()
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_rejects_title_that_does_not_slugify_to_slug(self, wiki_path: Path) -> None:
        """The exact root cause of the broken-link lint issue: title with a
        parenthetical slugifies to 'vision-language-models-vlm', not the slug."""
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "concepts/vision-language-models",
                "page_type": "concept",
                "title": "Vision-Language Models (VLM)",
                "content": "# Vision-Language Models",
            })
            assert "Error" in result
            assert "vision-language-models" in result
            assert not (wiki_path / "concepts/vision-language-models.md").exists()
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_warns_when_title_is_raw_slug(self, wiki_path: Path) -> None:
        """M1 regression: a raw slug is not a display title (soft warning only)."""
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "entities/alvaro-jimenez-martinez",
                "page_type": "entity",
                "title": "alvaro-jimenez-martinez",
                "content": "# Álvaro Jiménez Martínez\n\nBackend engineer.",
            })
            assert "Created page: entities/alvaro-jimenez-martinez" in result
            assert "Note: title" in result
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_auto_prefixes_directory_and_reports_normalized_slug(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            result = create_page.invoke({
                "slug": "Málaga",  # no directory, non-ASCII
                "page_type": "entity",
                "title": "Málaga",
                "content": "# Málaga\n\nA city.",
            })
            assert "Created page: entities/malaga" in result
            assert (wiki_path / "entities/malaga.md").is_file()
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)


class TestUpdatePageFrontmatterStrip:
    def test_strips_embedded_frontmatter_preserves_existing_fm(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            _create_test_page(wiki_path, "entities/python")
            result = update_page.invoke({
                "slug": "entities/python",
                "content": "---\nslug: python\ntype: entity\ntitle: Python\n---\n# Python\n\nUpdated content.",
                "sources": ["updated.pdf"],
            })
            assert "Updated" in result
            raw = (wiki_path / "entities/python.md").read_text()
            assert raw.count("---") == 2  # exactly ONE frontmatter block
            fm, body = read_page_with_frontmatter(wiki_path, "entities/python")
            assert fm.title == "Python"  # existing fm preserved (merge logic intact)
            assert fm.sources == ["updated.pdf"]  # explicit sources applied
            assert "Updated content." in body
            assert "slug: python" not in body  # embedded fm stripped
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)

    def test_clean_body_only_content_unchanged(self, wiki_path: Path) -> None:
        try:
            init_shared_tools(wiki_path)
            _create_test_page(wiki_path, "entities/python")
            result = update_page.invoke({
                "slug": "entities/python",
                "content": "# Python\n\nClean update.",
            })
            assert "Updated" in result
            raw = (wiki_path / "entities/python.md").read_text()
            assert raw.count("---") == 2
            assert "Clean update." in raw
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)
