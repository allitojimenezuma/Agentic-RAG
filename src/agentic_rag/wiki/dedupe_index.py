"""Regenerate ``wiki/index.md`` from the source-of-truth :class:`Wiki` model.

``index.md`` is a *derived view*: the filesystem + frontmatter are the source
of truth. This module rebuilds it deterministically from ``wiki.model.load_wiki``
using the exact entry format from ``io.index._format_entry``. Page
summaries come from the first section's body text (never a raw ``# H1``
heading, which was the previous hand-maintained corruption).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agentic_rag.io.index import _category_for_type, write_index
from agentic_rag.schemas.wiki import Index, IndexEntry
from agentic_rag.wiki.model import Page, Wiki, load_wiki

logger = logging.getLogger(__name__)

# Obsidian links: [[target]] or [[target|alias]].
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
# Markdown links/images: [text](url) / ![alt](url).
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
# Leading heading markers (# through ######) at line start.
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# Inline emphasis / code markers.
_EMPHASIS_RE = re.compile(r"[*_`]+")

_MAX_SUMMARY_CHARS = 160


def regenerate_index(wiki_path: Path) -> Path:
    """Rewrite ``wiki/index.md`` from the wiki's pages; returns its path.

    Loads the wiki via :func:`agentic_rag.wiki.model.load_wiki`, excludes
    ``lint-report-*.md`` pages, groups the rest by frontmatter ``type`` and
    writes the derived index atomically (temp + replace via ``write_index``).
    """
    wiki = load_wiki(wiki_path)
    categories = _build_categories(wiki)
    write_index(wiki_path, Index(categories=categories))
    index_path = wiki_path / "index.md"
    logger.info(
        "Regenerated %s (%d categories, %d entries)",
        index_path,
        len(categories),
        sum(len(e) for e in categories.values()),
    )
    return index_path


def _build_categories(wiki: Wiki) -> dict[str, list[IndexEntry]]:
    """Group non-lint-report pages by type into plural category lists."""
    categories: dict[str, list[IndexEntry]] = {}
    for page in wiki.pages:
        if page.rel_path.name.startswith("lint-report-"):
            continue
        category = _category_for_type(page.fm.type)
        categories.setdefault(category, []).append(
            IndexEntry(
                slug=page.slug,
                display_name=page.fm.title,
                updated=page.fm.updated,
                sources=list(page.fm.sources),
                type=page.fm.type,
                summary=_page_summary(page),
            )
        )
    return categories


def _page_summary(page: Page) -> str:
    """One-line plain-text summary for a page: first section text or title.

    Never a raw H1 heading — the intro body text is stripped of link syntax
    and other markdown, collapsed to a single line, and truncated to ~160
    chars at a word boundary. Falls back to the page title when no usable
    section text exists.
    """
    text = ""
    if page.sections:
        text = page.sections[0].text
    return _summarize(text) if text.strip() else page.fm.title


def _summarize(text: str) -> str:
    """Strip markdown/link syntax, collapse whitespace, truncate at a word
    boundary (append '…' when truncated)."""
    text = _WIKI_LINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _EMPHASIS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= _MAX_SUMMARY_CHARS:
        return text
    cut = text.rfind(" ", 0, _MAX_SUMMARY_CHARS)
    if cut <= 0:
        cut = _MAX_SUMMARY_CHARS
    return text[:cut].rstrip() + "…"
