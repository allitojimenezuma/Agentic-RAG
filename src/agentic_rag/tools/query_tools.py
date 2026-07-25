"""Tools for the query agent (read-only): find_relevant_pages + shared tools."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.io.index_manager import find_in_index
from agentic_rag.io.markdown_parser import extract_links
from agentic_rag.io.wiki_io import read_page
from agentic_rag.tools.shared import read_index, read_wiki_page, search_index  # noqa: F401

logger = logging.getLogger(__name__)


@tool
def find_relevant_pages(wiki_path: str, query: str) -> str:
    """Find wiki pages relevant to a query. Combines index search with link traversal. Returns a list of slugs to read."""
    logger.debug("Finding relevant pages for: %s", query)
    path = Path(wiki_path)
    # Step 1: search the index for matching entries
    index_results = find_in_index(path, query)
    slugs: list[str] = [e.slug for e in index_results]
    logger.debug("Index matched slugs: %s", slugs)

    # Step 2: for each matched page, follow [[links]] to related pages
    visited: set[str] = set(slugs)
    queue = list(slugs)
    while queue:
        current = queue.pop(0)
        try:
            content = read_page(path, current)
            links = extract_links(content)
            for link in links:
                target = link.target
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
        except FileNotFoundError:
            continue

    if not slugs:
        logger.debug("No pages found for query: %s", query)
        return f"No pages found for query '{query}'."

    result_slugs = sorted(visited)
    logger.debug("Found relevant pages: %s", result_slugs)
    lines = [f"Found {len(result_slugs)} relevant page(s) for '{query}':"]
    for s in result_slugs:
        marker = " (direct match)" if s in slugs else " (via links)"
        lines.append(f"- {s}{marker}")
    return "\n".join(lines)
