"""Tests for schemas/agents_md.py — AGENTS.md loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.schemas.agents_md import load_agents_md


class TestLoadAgentsMd:
    def test_loads_existing_file(self, agents_md_path: Path) -> None:
        """When AGENTS.md exists, returns its content."""
        result = load_agents_md(agents_md_path)
        assert "Wiki Schema" in result
        assert "entity" in result.lower()

    def test_returns_default_when_missing(self, tmp_path: Path) -> None:
        """When AGENTS.md is missing, returns a sensible default."""
        missing_path = tmp_path / "nonexistent" / "AGENTS.md"
        result = load_agents_md(missing_path)
        # Default should contain key schema elements
        assert "Wiki Schema" in result
        assert "entity" in result.lower()
        assert "concept" in result.lower()
        assert "source" in result.lower()
        assert "NEVER" in result  # Hard rules section

    def test_default_has_page_types(self, tmp_path: Path) -> None:
        result = load_agents_md(tmp_path / "missing.md")
        for ptype in ("entity", "concept", "source", "comparison", "overview"):
            assert ptype in result.lower()

    def test_default_has_naming_convention(self, tmp_path: Path) -> None:
        result = load_agents_md(tmp_path / "missing.md")
        assert "entities/" in result
        assert "concepts/" in result
        assert "sources/" in result

    def test_default_has_cross_reference_format(self, tmp_path: Path) -> None:
        result = load_agents_md(tmp_path / "missing.md")
        assert "[[" in result  # Obsidian-style links

    def test_default_has_hard_rules(self, tmp_path: Path) -> None:
        result = load_agents_md(tmp_path / "missing.md")
        assert "NEVER write outside" in result
        assert "NEVER modify `raw/`" in result
