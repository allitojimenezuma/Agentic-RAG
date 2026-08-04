"""Durable JSONL transcript store for per-agent chat history.

Pure Python (stdlib only): no Streamlit, no langchain imports — unit-testable.

Transcripts live under ``<repo>/frontend/history/<agent>/<thread_id>.jsonl``,
one JSON object per line: ``{"role": "user"|"assistant", "content": str}`` —
exactly the shape of the messages in ``st.session_state``. Append-only files
survive app restarts; corrupt lines are skipped on load. All reads are
tolerant: missing files and malformed records never raise.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Pinned: the store root is <repo>/frontend/history.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "frontend" / "history"


class HistoryStore:
    """Append/load/list/delete JSONL transcripts for one agent + thread."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, agent: str, thread_id: str) -> Path:
        return self.root / agent / f"{thread_id}.jsonl"

    def append(self, agent: str, thread_id: str, role: str, content: str) -> None:
        """Append one JSON line {"role","content"} to root/<agent>/<thread_id>.jsonl.

        Creates the agent directory (and parents) as needed.
        """
        path = self._path(agent, thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def load(self, agent: str, thread_id: str) -> list[dict]:
        """Return the transcript as [{"role","content"}, ...] in written order.

        Missing file -> []; blank lines skipped; corrupt / non-dict JSON lines
        skipped; read errors degrade to []. Never raises.
        """
        messages: list[dict] = []
        path = self._path(agent, thread_id)
        if not path.exists():
            return messages
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(record, dict):
                        messages.append(record)
        except OSError:
            return []
        return messages

    def list_threads(self, agent: str) -> list[str]:
        """Thread ids for an agent, sorted by file mtime descending (newest first)."""
        agent_dir = self.root / agent
        if not agent_dir.is_dir():
            return []
        try:
            files = [p for p in agent_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"]
        except OSError:
            return []

        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        files.sort(key=_mtime, reverse=True)
        return [p.stem for p in files]

    def delete(self, agent: str, thread_id: str) -> None:
        """Remove the transcript file if present; no-op otherwise. Never raises."""
        path = self._path(agent, thread_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem edge, tolerant by design
            logger.warning("history delete failed agent=%s thread=%s: %s", agent, thread_id, exc)

    def new_thread_id(self) -> str:
        """Fresh unique thread id."""
        return str(uuid.uuid4())
