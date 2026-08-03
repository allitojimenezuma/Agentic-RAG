"""Unit tests for the wiki_scan bulk-metadata tool in agentic_rag.tools.nav."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.io.wiki_io import write_page
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools.nav import wiki_scan
from agentic_rag.tools.shared import init_shared_tools

DEFAULT_WIKI_PATH = Path("./wiki")


def _write(
    wiki_path: Path,
    slug: str,
    content: str,
    *,
    frontmatter: bool = True,
    page_type: str | None = None,
    title: str | None = None,
) -> None:
    """Write a wiki page, optionally with valid frontmatter."""
    full = content
    if frontmatter:
        fm = Frontmatter(
            slug=slug,
            type=page_type
            or (slug.split("/")[0].rstrip("s") if "/" in slug else "entity"),
            title=title or slug.rsplit("/", 1)[-1].replace("-", " ").title(),
            sources=["manual"],
            updated=date(2025, 1, 1),
            tags=[],
        )
        full = serialize_frontmatter(fm) + content
    write_page(wiki_path, slug, full)


def _populate(wiki_path: Path) -> None:
    """Three content pages (one frontmatter-less) + one lint-report page."""
    _write(
        wiki_path,
        "entities/mlx",
        "# MLX\n\nMLX is a framework.\n\n## Related\n\n- [[Tool Calling]]",
        page_type="entity",
        title="MLX",
    )
    _write(
        wiki_path,
        "concepts/tool-calling",
        "# Tool Calling\n\nLets LLMs invoke tools.\n\n## Related\n\n- [[MLX]]",
        page_type="concept",
        title="Tool Calling",
    )
    # Frontmatter-less page (synthesized fm from path + first heading)
    _write(
        wiki_path,
        "concepts/no-fm",
        "# No FM\n\nSome   text  with  spaces.",
        frontmatter=False,
    )
    # Empty content edge case
    _write(wiki_path, "empty-page", "# Empty\n")
    # Derived artifact: must be excluded from lines AND from inbound counts
    _write(
        wiki_path,
        "lint-report-2025-01-02",
        "# Lint Report\n\nClean.\n\n- [[MLX]]",
    )


class TestWikiScan:
    def test_one_line_per_content_page(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _populate(wiki_path)
        result = wiki_scan.invoke({})
        lines = result.splitlines()
        assert len(lines) == 4
        assert all(line.startswith("- ") for line in lines)

    def test_excludes_lint_report_pages(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _populate(wiki_path)
        result = wiki_scan.invoke({})
        assert "lint-report" not in result

    def test_deterministic_slug_ordering(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _populate(wiki_path)
        lines = wiki_scan.invoke({}).splitlines()
        slugs = [line.split(" (", 1)[0][2:] for line in lines]
        assert slugs == sorted(slugs)
        assert slugs == [
            "concepts/no-fm",
            "concepts/tool-calling",
            "empty-page",
            "entities/mlx",
        ]

    def test_line_format_with_counts_and_date(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _populate(wiki_path)
        lines = wiki_scan.invoke({}).splitlines()
        mlx = next(line for line in lines if line.startswith("- entities/mlx"))
        assert mlx == (
            '- entities/mlx (entity) - MLX — "MLX is a framework." — '
            "out: 1 | in: 1 | updated: 2025-01-01"
        )
        # mlx in-count is 1 (tool-calling only) — the lint-report page's
        # [[MLX]] link must NOT count toward inbound.
        tc = next(line for line in lines if line.startswith("- concepts/tool-calling"))
        assert "out: 1 | in: 1" in tc

    def test_frontmatter_less_page_synthesized(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _populate(wiki_path)
        line = next(
            line
            for line in wiki_scan.invoke({}).splitlines()
            if line.startswith("- concepts/no-fm")
        )
        # Type/title synthesized from path + first heading; whitespace collapsed.
        assert line.startswith("- concepts/no-fm (concept) - No FM")
        assert '"Some text with spaces."' in line
        assert "out: 0 | in: 0" in line
        assert "updated: " in line

    def test_no_content_preview(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _populate(wiki_path)
        line = next(
            line
            for line in wiki_scan.invoke({}).splitlines()
            if line.startswith("- empty-page")
        )
        assert '"(no content)"' in line

    def test_preview_truncated_with_ellipsis(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        _populate(wiki_path)
        result = wiki_scan.invoke({"max_chars": 10})
        assert '— "MLX is a f…" —' in result
        # Default 200 keeps full short previews
        default = wiki_scan.invoke({})
        assert '"MLX is a framework."' in default
        assert "…" not in default

    def test_empty_wiki(self, wiki_path: Path) -> None:
        init_shared_tools(wiki_path)
        assert wiki_scan.invoke({}) == "No wiki pages found."


class TestWikiScanRestore:
    def test_restores_default_wiki_path(self, wiki_path: Path) -> None:
        """After a scan against a temp wiki, the global is restored (mirrors test_match.py)."""
        try:
            init_shared_tools(wiki_path)
            _populate(wiki_path)
            assert len(wiki_scan.invoke({}).splitlines()) == 4
        finally:
            init_shared_tools(DEFAULT_WIKI_PATH)
