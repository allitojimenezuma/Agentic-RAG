"""Unit tests for the pinned read-only wiki_command dispatcher.

Covers the grammar parser (join via && / newlines), every sub-command
(scan/search/read/links/match/health/help), read-only-by-construction behavior,
error surfacing, navigated-slug recording for citations (cite-or-die), and the
output cap. No network, no real LLM — pure deterministic engine output.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.io.wiki_io import write_page
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools import grounding
from agentic_rag.tools.nav import run_wiki_commands, wiki_command
from agentic_rag.tools.shared import init_shared_tools


def _page(wiki: Path, slug: str, title: str, page_type: str, body: str) -> None:
    """Write a wiki page with valid frontmatter."""
    fm = Frontmatter(
        slug=slug,
        type=page_type,
        title=title,
        sources=["sample.md"],
        updated=date(2025, 1, 1),
        tags=[],
    )
    write_page(wiki, slug, serialize_frontmatter(fm) + body)


def _populate(wiki: Path) -> None:
    """Two mutually-linked pages + one orphan + one lint-report (excluded)."""
    _page(
        wiki,
        "entities/mlx",
        "MLX",
        "entity",
        "# MLX\n\nMLX is a machine learning framework by Apple.\n\n## Related\n\n- [[Tool Calling]]",
    )
    _page(
        wiki,
        "concepts/tool-calling",
        "Tool Calling",
        "concept",
        "# Tool Calling\n\nLets LLMs invoke tools.\n\n## Related\n\n- [[MLX]]",
    )
    _page(
        wiki,
        "concepts/orphan",
        "Orphan Concept",
        "concept",
        "# Orphan Concept\n\nNot linked from anywhere.",
    )
    (wiki / "lint-report-2025-01-01.md").write_text("# Lint Report\n")


class TestSplitting:
    def test_multiple_commands_joined_by_and(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands('search "MLX" && health')
        assert "Found" in out
        assert "Pages audited" in out

    def test_multiple_commands_joined_by_newline(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands('search "MLX"\nhealth')
        assert "Found" in out
        assert "Pages audited" in out

    def test_blank_and_whitespace_lines_ignored(self, tmp_path) -> None:
        out = run_wiki_commands("   \n\nhelp")
        assert "scan" in out

    def test_empty_command_string(self, tmp_path) -> None:
        out = run_wiki_commands("")
        assert out == "No commands given."


class TestScan:
    def test_scan_lists_pages_excluding_lint_reports(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("scan")
        assert "entities/mlx" in out
        assert "concepts/orphan" in out
        assert "lint-report-2025-01-01" not in out
        assert "out: 1" in out  # link counts

    def test_scan_max_chars_shortens_preview(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        full = run_wiki_commands("scan")
        short = run_wiki_commands("scan --max-chars 8")
        assert len(short) < len(full)


class TestSearch:
    def test_search_does_not_record_navigated_slugs(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        cap = grounding.new_nav_capture()
        out = wiki_command.invoke(
            'search "Apple machine learning"',
            config={"configurable": {"nav_capture": cap}},
        )
        assert "entities/mlx" in out
        # Search must NOT mark hits as navigated: only `read` does. cite-or-die
        # therefore only grounds pages the agent actually opened, not every
        # keyword match.
        assert cap.navigated == set()

    def test_search_no_hits(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        assert "No relevant pages found" in run_wiki_commands('search "zzzzzz"')


class TestRead:
    def test_read_full_page_returns_raw_markdown(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("read entities/mlx")
        assert "slug: entities/mlx" in out  # includes frontmatter
        assert "MLX is a machine learning framework" in out

    def test_read_section(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands('read entities/mlx --section "Related"')
        assert "Tool Calling" in out
        assert "MLX is a machine learning" not in out

    def test_read_by_basename_slug_resolves(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("read mlx")
        assert "slug: entities/mlx" in out

    def test_read_records_navigated_slug(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        cap = grounding.new_nav_capture()
        wiki_command.invoke("read mlx", config={"configurable": {"nav_capture": cap}})
        assert "entities/mlx" in cap.navigated

    def test_read_missing_page_suggests_similar(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("read entities/nope")
        assert "Wiki page not found" in out


class TestLinks:
    def test_links_summary_marks_orphans(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("links")
        assert "### entities/mlx" in out
        assert "ORPHAN" in out
        assert "concepts/orphan" in out

    def test_links_single_slug(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("links --slug entities/mlx")
        assert "### entities/mlx" in out
        assert "### concepts/tool-calling" not in out


class TestMatch:
    def test_match_decision(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands('match "MLX" --type entity')
        assert out.startswith("exact: entities/mlx")
        out = run_wiki_commands('match "Brand New" --type entity')
        assert out.startswith("none:")


class TestHealth:
    def test_health_audits_deterministically(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("health")
        assert "Pages audited: 3" in out
        assert "[high] orphan: concepts/orphan" in out


class TestHelp:
    def test_help_prints_grammar(self, tmp_path) -> None:
        out = run_wiki_commands("help")
        for token in ("scan", "search", "read", "links", "match", "health"):
            assert token in out


class TestErrors:
    def test_unknown_command_is_an_error_line(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        out = run_wiki_commands("frobnicate")
        assert out.startswith("Error: unknown command")

    def test_bad_flag_is_reported(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        assert "unknown flag" in run_wiki_commands("scan --bogus 3")

    def test_stop_flag_value_flag_injection_unknown_flag(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        # A flag-like value on a flag is rejected, never executed.
        out = run_wiki_commands('search "x" --k --type entity')
        assert "requires a non-flag value" in out

    def test_extra_positionals_are_usage_errors(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        assert "usage" in run_wiki_commands('search "x" 1')

    def test_unbalanced_quotes_surface_error(self, tmp_path) -> None:
        out = run_wiki_commands('search "unterminated')
        assert "could not parse" in out

    def test_error_in_one_command_does_not_stop_the_batch(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("bogus && health")
        assert "Error: unknown command" in out
        assert "Pages audited" in out

    def test_output_truncated_when_huge(self, tmp_path) -> None:
        init_shared_tools(tmp_path)
        _populate(tmp_path)
        out = run_wiki_commands("read entities/mlx && " * 80 + "health")
        assert out.endswith("… (output truncated)")


class TestReadOnlyByConstruction:
    def test_commands_never_mutate_the_wiki(self, tmp_path) -> None:
        wiki = tmp_path
        init_shared_tools(wiki)
        _populate(wiki)
        before = {
            p.relative_to(wiki): p.read_text(encoding="utf-8")
            for p in wiki.rglob("*.md")
        }
        # Hammer every sub-command (including hostile strings).
        run_wiki_commands(
            "scan && links && health && search \"x\" && read entities/mlx "
            "&& match \"MLX\" --type entity && "
            "read ../../etc/passwd && search \"rm -rf wiki\" && links --slug ../../x"
        )
        after = {
            p.relative_to(wiki): p.read_text(encoding="utf-8")
            for p in wiki.rglob("*.md")
        }
        assert before == after


def test_tool_schema_has_single_command_arg() -> None:
    """wiki_command exposes exactly one argparse-free str param `command`."""
    schema = wiki_command.args_schema.model_json_schema()
    assert schema["required"] == ["command"]
    assert set(schema["properties"]) == {"command"}