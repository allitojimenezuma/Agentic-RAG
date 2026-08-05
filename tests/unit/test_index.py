"""Tests for io/index.py — index.md read/write/update."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agentic_rag.io.index import read_index, write_index
from agentic_rag.schemas.wiki import Index, IndexEntry


@pytest.fixture
def wiki(tmp_path: Path) -> Path:
    """Create a temp wiki with a sample index.md."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    wiki.joinpath("index.md").write_text(
        "# Wiki Index\n\n"
        "## Entities\n"
        "- [[Python]] - High-level programming language | Source: manual | Updated: 2025-01-01\n"
        "- [[MLX]] - ML framework for Apple Silicon | Sources: cv.pdf | Updated: 2026-07-20\n\n"
        "## Concepts\n"
        "- [[Machine Learning]] - AI subset | Sources: 1 | Updated: 2025-01-01\n\n"
        "## Sources\n"
        "- [Cv](cv.pdf) - Ingested: 2026-07-20\n\n"
        "## Comparisons\n"
    )
    return wiki


class TestReadIndex:
    def test_parses_entities(self, wiki: Path) -> None:
        index = read_index(wiki)
        assert "entities" in index.categories
        entities = index.categories["entities"]
        assert len(entities) == 2
        assert entities[0].slug == "python"
        assert entities[0].summary == "High-level programming language"
        assert entities[0].updated == date(2025, 1, 1)

    def test_parses_concepts(self, wiki: Path) -> None:
        index = read_index(wiki)
        assert "concepts" in index.categories
        assert len(index.categories["concepts"]) == 1
        assert index.categories["concepts"][0].slug == "machine-learning"

    def test_parses_sources(self, wiki: Path) -> None:
        index = read_index(wiki)
        assert "sources" in index.categories
        assert len(index.categories["sources"]) == 1
        assert index.categories["sources"][0].slug == "cv"

    def test_empty_index(self, tmp_path: Path) -> None:
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        wiki.joinpath("index.md").write_text("# Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n\n## Comparisons\n")
        index = read_index(wiki)
        for entries in index.categories.values():
            assert entries == []

    def test_missing_index_file(self, tmp_path: Path) -> None:
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        index = read_index(wiki)
        assert index.categories == {}


class TestWriteIndex:
    def test_write_and_read_back(self, wiki: Path) -> None:
        index = Index(
            categories={
                "entities": [
                    IndexEntry(
                        slug="python",
                        summary="Programming language",
                        type="entity",
                        sources=["manual"],
                        updated=date(2025, 1, 1),
                    )
                ],
                "concepts": [],
                "sources": [],
                "comparisons": [],
            }
        )
        write_index(wiki, index)

        read_back = read_index(wiki)
        assert len(read_back.categories["entities"]) == 1
        assert read_back.categories["entities"][0].slug == "python"
