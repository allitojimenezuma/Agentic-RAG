"""Unit tests for frontend/history_store: durable JSONL transcript store.

Pure stdlib; exercises append/load round-trip, per-agent isolation, mtime-desc
list_threads, corrupt-line tolerance, delete, and unique new_thread_id.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from frontend.history_store import DEFAULT_ROOT, HistoryStore


@pytest.fixture
def store(tmp_path: Path) -> HistoryStore:
    return HistoryStore(tmp_path)


class TestAppendLoad:
    def test_round_trip_preserves_order_and_shape(self, store):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Again"},
        ]
        for msg in messages:
            store.append("query", "tid-1", msg["role"], msg["content"])
        assert store.load("query", "tid-1") == messages

    def test_append_creates_parent_dirs(self, store):
        store.append("ingest", "tid-1", "user", "x")
        path = store.root / "ingest" / "tid-1.jsonl"
        assert path.is_file()
        assert path.read_text(encoding="utf-8").strip() == '{"role": "user", "content": "x"}'

    def test_load_missing_file_returns_empty(self, store):
        assert store.load("query", "nope") == []

    def test_load_missing_agent_dir_returns_empty(self, store):
        assert store.load("unknown-agent", "tid") == []


class TestIsolation:
    def test_same_thread_id_different_agents(self, store):
        store.append("query", "shared", "user", "q1")
        store.append("ingest", "shared", "user", "q2")
        assert store.load("query", "shared") == [{"role": "user", "content": "q1"}]
        assert store.load("ingest", "shared") == [{"role": "user", "content": "q2"}]


class TestListThreads:
    def test_sorted_by_mtime_desc(self, store):
        # Control mtimes explicitly: filesystem timestamps are not reliable at
        # creation-time granularity.
        for tid, t in [("old", 100.0), ("mid", 200.0), ("new", 300.0)]:
            store.append("query", tid, "user", tid)
            os.utime(store.root / "query" / f"{tid}.jsonl", (t, t))
        assert store.list_threads("query") == ["new", "mid", "old"]

    def test_no_files_returns_empty(self, store):
        assert store.list_threads("query") == []
        assert store.list_threads("unknown-agent") == []

    def test_ignores_non_jsonl_files(self, store):
        (store.root / "query").mkdir(parents=True)
        (store.root / "query" / "notes.txt").write_text("hi", encoding="utf-8")
        assert store.list_threads("query") == []


class TestCorruptLines:
    def test_skips_corrupt_lines_and_keeps_good(self, store):
        store.append("query", "tid", "user", "good")
        path = store.root / "query" / "tid.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write("not json\n")
            fh.write('{"role": "user"\n')  # truncated JSON
            fh.write("[1, 2, 3]\n")  # valid JSON but not a dict
            fh.write("\n")
        assert store.load("query", "tid") == [{"role": "user", "content": "good"}]

    def test_fully_corrupt_file_returns_empty(self, store):
        path = store.root / "query" / "tid.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("garbage\n{{{\n", encoding="utf-8")
        assert store.load("query", "tid") == []


class TestDelete:
    def test_delete_removes_file(self, store):
        store.append("query", "tid", "user", "x")
        assert (store.root / "query" / "tid.jsonl").is_file()
        store.delete("query", "tid")
        assert not (store.root / "query" / "tid.jsonl").exists()
        assert store.load("query", "tid") == []

    def test_delete_missing_file_is_noop(self, store):
        store.delete("query", "ghost")  # must not raise


class TestNewThreadId:
    def test_unique_and_valid_uuid(self, store):
        a = store.new_thread_id()
        b = store.new_thread_id()
        assert isinstance(a, str)
        assert a != b
        uuid.UUID(a)  # raises if malformed

    def test_default_root_is_repo_frontend_history(self):
        # DEFAULT_ROOT is pinned to <repo>/frontend/history.
        assert DEFAULT_ROOT.name == "history"
        assert DEFAULT_ROOT.parent.name == "frontend"
        assert (DEFAULT_ROOT.parent.parent / "frontend" / "history") == DEFAULT_ROOT
