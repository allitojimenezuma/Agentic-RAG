"""Deterministic create/update/conflict matcher for wiki page names.

Replaces the old ``search_index(name)`` + ``read_wiki_page(slug)`` round-trip
guessing with one Python call over ``wiki.by_slug`` + BM25 ``search``. Pure
decision logic (0 LLM calls) plus a thin ``@tool`` wrapper.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel

from agentic_rag.io.markdown_parser import slugify
from agentic_rag.tools.shared import get_wiki_path
from agentic_rag.wiki.model import Wiki, load_wiki
from agentic_rag.wiki.search import search

logger = logging.getLogger(__name__)

# Inverse of ``wiki.model._DIR_TO_TYPE``: page type -> ``by_slug`` directory
# prefix. A page_type with no directory mapping (e.g. ``overview``, which
# lives at the wiki root) triggers the untyped-search fallback.
_TYPE_TO_DIR = {
    "entity": "entities",
    "concept": "concepts",
    "source": "sources",
    "comparison": "comparisons",
}


class MatchResult(BaseModel):
    """Outcome of ``match_page``: a decision plus the candidate slugs."""

    decision: Literal["exact", "similar", "none", "conflict"]
    slugs: list[str]
    detail: str


def match_page(wiki: Wiki, name: str, page_type: str) -> MatchResult:
    """Decide whether ``name`` matches an existing wiki page.

    Pure and deterministic — no score thresholds. Decision rule:

    1. ``candidate = slugify(name)``; if the candidate is a ``by_slug`` key,
       or is the unique basename of a nested slug (e.g. ``mlx`` ->
       ``entities/mlx``), the match is ``exact``.
    2. Otherwise run ``search(wiki, name, k=5, types=[page_type])`` and look
       only at direct hits (``matched_via != "expand-link"``): one hit ->
       ``similar``, two or more -> ``conflict``, none -> ``none``.
    3. When ``page_type`` maps to no ``by_slug`` directory (e.g. ``source``
       on a wiki without a ``sources/`` dir), step 2 runs WITHOUT the
       ``types`` filter (untyped fallback).
    """
    candidate = slugify(name)
    resolved = _resolve_exact_slug(wiki, candidate)
    if resolved is not None:
        return MatchResult(decision="exact", slugs=[resolved], detail="exact slug match")

    types: list[str] | None = [page_type] if _type_dir_exists(wiki, page_type) else None
    hits = search(wiki, name, k=5, types=types)
    direct = [h for h in hits if h.matched_via != "expand-link"]

    if len(direct) == 1:
        return MatchResult(
            decision="similar",
            slugs=[direct[0].slug],
            detail="BM25 match — update existing",
        )
    if len(direct) >= 2:
        return MatchResult(
            decision="conflict",
            slugs=[direct[0].slug, direct[1].slug],
            detail="multiple candidates — flag contradiction",
        )
    return MatchResult(
        decision="none", slugs=[], detail="no existing page — create new"
    )


def _resolve_exact_slug(wiki: Wiki, candidate: str) -> str | None:
    """Resolve ``candidate`` to a page slug, or None when not an exact match.

    Checks the candidate as a literal ``by_slug`` key first (root-level pages
    such as overviews), then as the unique basename of a nested slug (e.g.
    ``mlx`` -> ``entities/mlx``). An ambiguous basename (same short name under
    several directories) is NOT exact — it falls through to the search step,
    which surfaces it as a conflict.
    """
    if candidate in wiki.by_slug:
        return candidate
    matches = [slug for slug in wiki.by_slug if slug.rsplit("/", 1)[-1] == candidate]
    if len(matches) == 1:
        return matches[0]
    return None


def _type_dir_exists(wiki: Wiki, page_type: str) -> bool:
    """True when ``page_type`` maps to a directory present in ``wiki.by_slug``."""
    directory = _TYPE_TO_DIR.get(page_type)
    if directory is None:
        return False
    return any(slug.startswith(directory + "/") for slug in wiki.by_slug)


@tool
def match_page_tool(name: str, page_type: str) -> str:
    """Match a page name against the wiki to decide create vs update vs conflict. Returns '<decision>: <slugs> — <detail>' where decision is one of 'exact' (use update_page), 'similar' (use update_page), 'conflict' (use flag_contradiction), or 'none' (use create_page). Call this before create_page/update_page."""
    logger.debug("Matching page: %s (type=%s)", name, page_type)
    wiki = load_wiki(get_wiki_path())
    result = match_page(wiki, name, page_type)
    return f"{result.decision}: {', '.join(result.slugs)} — {result.detail}"
