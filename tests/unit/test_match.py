"""Unit tests for the deterministic match_page matcher and its @tool wrapper."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.io.wiki_io import write_page
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.match import MatchResult, match_page, match_page_tool
from agentic_rag.wiki.model import Wiki, load_wiki

DEFAULT_WIKI_PATH = Path("./wiki")


def _write_page(
    wiki_path: Path, slug: str, title: str, page_type: str, body: str
) -> None:
    """Write a wiki page with valid frontmatter (mirrors test_tools.py)."""
    fm = Frontmatter(
        slug=slug,
        type=page_type,
        title=title,
        sources=["manual"],
        updated=date(2025, 1, 1),
    )
    write_page(wiki_path, slug, serialize_frontmatter(fm) + body)


@pytest.fixture
def wiki(tmp_path: Path) -> Wiki:
    """Small wiki with entity/concept pages but NO sources/ directory."""
    _write_page(
        tmp_path,
        "entities/mlx",
        "MLX",
        "entity",
        "# MLX\n\nApple's machine learning framework for Apple silicon.",
    )
    _write_page(
        tmp_path,
        "entities/python",
        "Python",
        "entity",
        "# Python\n\nA general-purpose programming language.",
    )
    _write_page(
        tmp_path,
        "concepts/rag",
        "Retrieval-Augmented Generation",
        "concept",
        "# RAG\n\nRetrieval augmented generation combines retrieval with generation.",
    )
    return load_wiki(tmp_path)


@pytest.fixture
def wiki_conflict(tmp_path: Path) -> Wiki:
    """Two entity pages that both match the query 'Apple machine learning'."""
    _write_page(
        tmp_path,
        "entities/mlx",
        "MLX",
        "entity",
        "# MLX\n\nApple's machine learning framework.",
    )
    _write_page(
        tmp_path,
        "entities/apple-silicon",
        "Apple Silicon",
        "entity",
        "# Apple Silicon\n\nApple's machine learning hardware.",
    )
    return load_wiki(tmp_path)


def test_exact_root_slug_match(wiki: Wiki, tmp_path: Path) -> None:
    """A candidate that is itself a by_slug key (root-level page) is exact."""
    _write_page(tmp_path, "mlx", "MLX Overview", "overview", "# MLX\n\nOverview.")
    wiki = load_wiki(tmp_path)
    result = match_page(wiki, "MLX", "entity")
    assert result == MatchResult(
        decision="exact", slugs=["mlx"], detail="exact slug match"
    )


def test_exact_nested_slug_short_form(wiki: Wiki) -> None:
    """'MLX' slugifies to 'mlx', resolving to the nested slug entities/mlx."""
    result = match_page(wiki, "MLX", "entity")
    assert result == MatchResult(
        decision="exact", slugs=["entities/mlx"], detail="exact slug match"
    )


def test_similar_single_direct_hit(wiki: Wiki) -> None:
    """One direct BM25 hit -> similar, even though the name is not an exact slug."""
    result = match_page(wiki, "Apple machine learning framework", "entity")
    assert result == MatchResult(
        decision="similar",
        slugs=["entities/mlx"],
        detail="BM25 match — update existing",
    )


def test_conflict_multiple_direct_hits(wiki_conflict: Wiki) -> None:
    """Two direct BM25 hits -> conflict with the top two slugs."""
    result = match_page(wiki_conflict, "Apple machine learning", "entity")
    assert result.decision == "conflict"
    assert set(result.slugs) == {"entities/mlx", "entities/apple-silicon"}
    assert result.detail == "multiple candidates — flag contradiction"


def test_none_no_direct_hits(wiki: Wiki) -> None:
    """No direct BM25 hit -> none."""
    result = match_page(wiki, "Nonexistent Thing", "entity")
    assert result == MatchResult(
        decision="none", slugs=[], detail="no existing page — create new"
    )


def test_untyped_fallback_when_type_dir_missing(wiki: Wiki) -> None:
    """page_type 'source' maps to no sources/ dir -> search runs untyped."""
    result = match_page(wiki, "Python programming language", "source")
    assert result == MatchResult(
        decision="similar",
        slugs=["entities/python"],
        detail="BM25 match — update existing",
    )


def test_types_filter_applied_when_type_dir_exists(tmp_path: Path) -> None:
    """With a sources/ dir present, the types filter excludes entity pages."""
    _write_page(
        tmp_path,
        "sources/mlx-notes",
        "MLX Notes",
        "source",
        "# MLX Notes\n\nSource notes on Apple machine learning.",
    )
    _write_page(
        tmp_path,
        "entities/mlx",
        "MLX",
        "entity",
        "# MLX\n\nApple's machine learning framework.",
    )
    wiki = load_wiki(tmp_path)
    result = match_page(wiki, "Apple machine learning", "source")
    assert result == MatchResult(
        decision="similar",
        slugs=["sources/mlx-notes"],
        detail="BM25 match — update existing",
    )


def test_match_page_tool_format(wiki: Wiki, tmp_path: Path) -> None:
    """Tool returns '<decision>: <slugs joined ', '> — <detail>'."""
    init_shared_tools(tmp_path)
    try:
        assert (
            match_page_tool.invoke({"name": "MLX", "page_type": "entity"})
            == "exact: entities/mlx — exact slug match"
        )
        assert (
            match_page_tool.invoke(
                {"name": "Apple machine learning framework", "page_type": "entity"}
            )
            == "similar: entities/mlx — BM25 match — update existing"
        )
        out = match_page_tool.invoke(
            {"name": "Nonexistent Thing", "page_type": "entity"}
        )
        assert out.startswith("none:")
        assert out.endswith("— no existing page — create new")
    finally:
        init_shared_tools(DEFAULT_WIKI_PATH)


def test_match_page_tool_conflict_format(
    wiki_conflict: Wiki, tmp_path: Path
) -> None:
    """Tool renders both candidate slugs for a conflict."""
    init_shared_tools(tmp_path)
    try:
        result = match_page(load_wiki(tmp_path), "Apple machine learning", "entity")
        assert result.decision == "conflict"
        out = match_page_tool.invoke(
            {"name": "Apple machine learning", "page_type": "entity"}
        )
        assert out == (
            f"conflict: {', '.join(result.slugs)}"
            " — multiple candidates — flag contradiction"
        )
    finally:
        init_shared_tools(DEFAULT_WIKI_PATH)
