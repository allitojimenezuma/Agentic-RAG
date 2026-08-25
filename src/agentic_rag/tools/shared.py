"""Shared helpers for all agents: wiki-path init + raw index summary.

The per-agent navigation/search tools live in ``tools/nav.py``; this module
only holds the module-level wiki path state and the index summary injected
into agent system prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set once when agent is built via init_shared_tools()
_WIKI_PATH: Path = Path("./wiki")
_RAW_PATH: Path = Path("./raw")


def init_shared_tools(
    wiki_path: str | Path, raw_path: str | Path | None = None
) -> None:
    """Initialize the wiki path (and raw sources path) for all shared tools.

    Called once at agent build time. ``raw_path`` defaults to ``./raw`` and is
    only meaningful for the ingest agent, whose ``read_source`` tool is confined
    to that directory.
    """
    global _WIKI_PATH, _RAW_PATH
    _WIKI_PATH = Path(wiki_path)
    if raw_path is not None:
        _RAW_PATH = Path(raw_path)
    logger.debug("Shared tools wiki_path=%s raw_path=%s", _WIKI_PATH, _RAW_PATH)


def get_wiki_path() -> Path:
    """Get the current wiki path. Used by tool modules that import it."""
    return _WIKI_PATH


def get_raw_path() -> Path:
    """Get the configured raw sources directory.

    ``read_source`` (ingest agent) is restricted to reading files under this
    directory so the agent cannot read arbitrary repository files.
    """
    return _RAW_PATH


def get_index_summary(wiki_path: Path | None = None) -> str:
    """Read the raw wiki ``index.md`` for injection into agent system prompts.

    Gives every agent a lightweight overview of all pages (slug, type, title,
    sources, date) without requiring an extra tool call. Returns ``"Index
    empty."`` on missing file or parse failure — never raises.
    """
    path = wiki_path or _WIKI_PATH
    index_path = path / "index.md"
    try:
        content = index_path.read_text(encoding="utf-8").strip()
        return content if content else "Index empty."
    except FileNotFoundError:
        return "Index not found."
    except Exception:
        logger.debug("Failed to read index from %s", index_path, exc_info=True)
        return "Index unavailable."
