"""Shared tools used by multiple agents: read_index, read_wiki_page, search_index."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.io.index_manager import find_in_index, read_index as _read_index
from agentic_rag.io.wiki_io import read_page as _read_page


@tool
def read_index(wiki_path: str) -> str:
    """Read the wiki index.md and return its full content. Shows all entities, concepts, sources, and comparisons with summaries."""
    path = Path(wiki_path)
    idx = _read_index(path)
    if not idx.categories:
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
def read_wiki_page(wiki_path: str, slug: str) -> str:
    """Read a wiki page by slug. Returns the full markdown content including frontmatter. Use this to get detailed information about any entity, concept, or source."""
    path = Path(wiki_path)
    return _read_page(path, slug)


@tool
def search_index(wiki_path: str, query: str) -> str:
    """Search the wiki index by keyword. Returns matching entries with their slugs, types, and summaries. Use to find relevant pages before reading them."""
    path = Path(wiki_path)
    results = find_in_index(path, query)
    if not results:
        return f"No results found for '{query}'."

    lines: list[str] = [f"Found {len(results)} result(s) for '{query}':"]
    for entry in results:
        updated = entry.updated.isoformat()
        lines.append(
            f"- {entry.slug} ({entry.type}) - {entry.summary} | Updated: {updated}"
        )
    return "\n".join(lines)
