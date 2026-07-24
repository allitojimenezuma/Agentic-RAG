"""Shared test fixtures for agentic-rag tests."""

import pytest
from pathlib import Path


@pytest.fixture
def wiki_path(tmp_path: Path) -> Path:
    """Create a temporary wiki directory with basic structure."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "entities").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "comparisons").mkdir()
    (wiki / "index.md").write_text("# Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n\n## Comparisons\n")
    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    """Create a temporary raw sources directory."""
    raw = tmp_path / "raw"
    raw.mkdir()
    return raw


@pytest.fixture
def agents_md_path(tmp_path: Path) -> Path:
    """Create a temporary AGENTS.md file."""
    path = tmp_path / "AGENTS.md"
    path.write_text("# Wiki Schema\n\nPage types: entity, concept, source, comparison, overview.\n")
    return path
