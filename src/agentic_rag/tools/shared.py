"""Shared tools used by multiple agents: read_index, read_wiki_page, search_index."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.io.index_manager import find_in_index, read_index as _read_index
from agentic_rag.io.wiki_io import read_page as _read_page

logger = logging.getLogger(__name__)

# Set once when agent is built via init_shared_tools()
_WIKI_PATH: Path = Path("./wiki")


def init_shared_tools(wiki_path: str | Path) -> None:
    """Initialize the wiki path for all shared tools. Called once at agent build time."""
    global _WIKI_PATH
    _WIKI_PATH = Path(wiki_path)
    logger.debug("Shared tools wiki_path set to %s", _WIKI_PATH)


def get_wiki_path() -> Path:
    """Get the current wiki path. Used by tool modules that import it."""
    return _WIKI_PATH


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


@tool
def read_index() -> str:
    """Read the wiki index.md and return its full content. Shows all entities, concepts, sources, and comparisons with summaries."""
    logger.debug("Reading wiki index from %s", _WIKI_PATH)
    idx = _read_index(_WIKI_PATH)
    if not idx.categories:
        logger.debug("Index is empty")
        return "Index is empty."

    lines: list[str] = []
    for category, entries in idx.categories.items():
        if not entries:
            continue
        lines.append(f"## {category.replace('-', ' ').title()}")
        for entry in entries:
            updated = entry.updated.isoformat()
            sources = ", ".join(entry.sources) if entry.sources else "manual"
            lines.append(
                f"- {entry.slug} ({entry.type}) - {entry.summary} | Sources: {sources} | Updated: {updated}"
            )
        lines.append("")

    return "\n".join(lines) if lines else "Index is empty."


@tool
def read_wiki_page(slug: str) -> str:
    """Read a wiki page by slug. Returns the full markdown content including frontmatter. Use this to get detailed information about any entity, concept, or source."""
    logger.debug("Reading wiki page: %s", slug)
    return _read_page(_WIKI_PATH, slug)


@tool
def search_index(query: str) -> str:
    """Search the wiki index by keyword. Returns matching entries with their slugs, types, and summaries. Use to find relevant pages before reading them."""
    logger.debug("Searching index for: %s", query)
    results = find_in_index(_WIKI_PATH, query)
    if not results:
        return f"No results found for '{query}'."

    lines: list[str] = [f"Found {len(results)} result(s) for '{query}':"]
    for entry in results:
        updated = entry.updated.isoformat()
        lines.append(
            f"- {entry.slug} ({entry.type}) - {entry.summary} | Updated: {updated}"
        )
    return "\n".join(lines)
