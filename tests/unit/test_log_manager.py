"""Tests for io/log_manager.py — append-only log.md operations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agentic_rag.io.log_manager import append_log, tail_log
from agentic_rag.schemas.wiki import LogEntry


@pytest.fixture
def wiki(tmp_path: Path) -> Path:
    """Create a temp wiki with empty log.md."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    wiki.joinpath("log.md").write_text("# Wiki Log\n")
    return wiki


class TestAppendLog:
    def test_creates_correct_prefix(self, wiki: Path) -> None:
        entry = LogEntry(
            timestamp=datetime(2025, 6, 15, 14, 30),
            op="ingest",
            title="My Source",
            details="Created: [[Page A]], [[Page B]]",
        )
        append_log(wiki, entry)

        content = (wiki / "log.md").read_text()
        assert "## [2025-06-15 14:30] ingest | My Source" in content
        assert "- Created: [[Page A]], [[Page B]]" in content

    def test_append_multiple(self, wiki: Path) -> None:
        for i in range(3):
            entry = LogEntry(
                timestamp=datetime(2025, 1, 1, 0, i),
                op="create",
                title=f"Page {i}",
                details=f"Detail {i}",
            )
            append_log(wiki, entry)

        content = (wiki / "log.md").read_text()
        assert content.count("## [") == 3

    def test_empty_details(self, wiki: Path) -> None:
        entry = LogEntry(
            timestamp=datetime(2025, 1, 1, 0, 0),
            op="query",
            title="What is ML?",
            details="",
        )
        append_log(wiki, entry)

        content = (wiki / "log.md").read_text()
        assert "## [2025-01-01 00:00] query | What is ML?" in content


class TestTailLog:
    def test_returns_last_n(self, wiki: Path) -> None:
        for i in range(5):
            entry = LogEntry(
                timestamp=datetime(2025, 1, 1, 0, i),
                op="create",
                title=f"Page {i}",
                details=f"Detail {i}",
            )
            append_log(wiki, entry)

        entries = tail_log(wiki, n=3)
        assert len(entries) == 3
        assert entries[0].title == "Page 2"
        assert entries[-1].title == "Page 4"

    def test_empty_log(self, wiki: Path) -> None:
        entries = tail_log(wiki, n=5)
        assert entries == []

    def test_tail_more_than_available(self, wiki: Path) -> None:
        entry = LogEntry(
            timestamp=datetime(2025, 1, 1, 0, 0),
            op="create",
            title="Only Page",
            details="Detail",
        )
        append_log(wiki, entry)

        entries = tail_log(wiki, n=10)
        assert len(entries) == 1
        assert entries[0].title == "Only Page"

    def test_preserves_metadata(self, wiki: Path) -> None:
        entry = LogEntry(
            timestamp=datetime(2025, 6, 15, 14, 30),
            op="ingest",
            title="Test Source",
            details="Line 1\nLine 2",
        )
        append_log(wiki, entry)

        entries = tail_log(wiki, n=1)
        assert entries[0].timestamp == datetime(2025, 6, 15, 14, 30)
        assert entries[0].op == "ingest"
        assert entries[0].title == "Test Source"
        assert "Line 1" in entries[0].details
        assert "Line 2" in entries[0].details
