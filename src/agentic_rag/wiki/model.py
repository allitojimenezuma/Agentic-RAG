"""Source-of-truth in-memory Wiki model.

Built from ``io.wiki_io.list_pages`` + parsed frontmatter + extracted
headings/links. The filesystem and frontmatter are the source of truth;
``index.md`` is a derived view (regenerated elsewhere, see T4).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from agentic_rag.io.markdown_parser import (
    extract_headings,
    extract_links,
    parse_frontmatter,
    slugify,
)
from agentic_rag.io.wiki_io import list_pages
from agentic_rag.schemas.wiki import Frontmatter, Heading, Link

logger = logging.getLogger(__name__)

# First path segment -> page type, used for frontmatter-less pages.
DIR_TO_TYPE = {
    "entities": "entity",
    "concepts": "concept",
    "sources": "source",
    "comparisons": "comparison",
}


class Section(BaseModel):
    """A body section under a single heading (heading line itself excluded)."""

    heading: str
    level: int
    text: str


class Page(BaseModel):
    """A curated wiki page: slug, path, frontmatter, sections and outbound links."""

    slug: str
    rel_path: Path
    fm: Frontmatter
    sections: list[Section]
    outbound_links: list[str]
    word_count: int


class Wiki(BaseModel):
    """In-memory source-of-truth model of a wiki directory."""

    pages: list[Page]
    by_slug: dict[str, Page]


def load_wiki(wiki_path: Path) -> Wiki:
    """Load every page under ``wiki_path`` into a :class:`Wiki` model.

    Uses ``io.wiki_io.list_pages`` (excludes ``index.md``/``log.md``). Pages
    without (valid) frontmatter get a synthesized ``Frontmatter`` with an
    inferred type/title and the file's mtime date — never raises on a
    frontmatter-less page.
    """
    page_paths = list_pages(wiki_path)
    # Slug set must be complete BEFORE resolving links (mirrors lint_tools).
    page_slugs = {str(p.relative_to(wiki_path)).removesuffix(".md") for p in page_paths}

    pages: list[Page] = []
    by_slug: dict[str, Page] = {}
    for page_path in page_paths:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        content = page_path.read_text(encoding="utf-8")
        body = _strip_frontmatter_block(content)

        headings = extract_headings(body)
        fm = _parse_or_synthesize_frontmatter(slug, content, body, page_path, headings)

        sections = _build_sections(body, headings)
        outbound_links = _resolve_outbound_links(extract_links(body), slug, page_slugs)
        word_count = len(body.split())

        page = Page(
            slug=slug,
            rel_path=page_path,
            fm=fm,
            sections=sections,
            outbound_links=outbound_links,
            word_count=word_count,
        )
        pages.append(page)
        by_slug[slug] = page

    if not pages:
        logger.debug("No pages found in %s", wiki_path)
    return Wiki(pages=pages, by_slug=by_slug)


def _strip_frontmatter_block(content: str) -> str:
    """Return the body with the leading ``---`` frontmatter block removed.

    Mirrors ``io.wiki_io.read_page_with_frontmatter``: split on ``---`` and
    keep everything after the closing delimiter (lstrip newline). Pages
    without frontmatter pass through untouched.
    """
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[2].lstrip("\n")
    return content


def _parse_or_synthesize_frontmatter(
    slug: str,
    content: str,
    body: str,
    page_path: Path,
    headings: list[Heading],
) -> Frontmatter:
    """Parse frontmatter; synthesize a fallback instead of raising."""
    if content.startswith("---"):
        try:
            return parse_frontmatter(content)
        except Exception:  # noqa: BLE001 - malformed frontmatter must not break loading
            logger.debug("Malformed frontmatter in %s; synthesizing fallback", slug)
    return _synthesize_frontmatter(slug, body, page_path, headings)


def _synthesize_frontmatter(
    slug: str,
    body: str,
    page_path: Path,
    headings: list[Heading],
) -> Frontmatter:
    """Build a Frontmatter from path/body when parsing fails or is absent."""
    first_segment = slug.split("/", 1)[0] if "/" in slug else ""
    page_type = DIR_TO_TYPE.get(first_segment, "unknown")

    title = slug.rsplit("/", 1)[-1]
    for heading in headings:
        if heading.level == 1:
            title = heading.text
            break

    try:
        mtime = date.fromtimestamp(page_path.stat().st_mtime)
    except OSError:
        mtime = date.today()

    return Frontmatter(
        slug=slug,
        type=page_type,
        title=title,
        updated=mtime,
        sources=[],
        tags=[],
    )


def _build_sections(body: str, headings: list[Heading]) -> list[Section]:
    """Split the body into sections at each heading (preamble prepended to
    the first section; a lone preamble becomes a single synthetic section).

    Headings are matched to source lines in order by their ``#``-prefix and
    extracted text, so section text = body lines from that heading until the
    next heading (exclusive), with the heading line itself excluded.
    """
    lines = body.split("\n")
    if not headings:
        text = body.strip()
        return [Section(heading="", level=0, text=text)] if text else []

    sections: list[Section] = []
    preamble: list[str] = []
    current: Section | None = None
    heading_index = 0

    for line in lines:
        if heading_index < len(headings):
            heading = headings[heading_index]
            stripped = line.strip()
            prefix = "#" * heading.level + " "
            if stripped.startswith(prefix) and stripped[len(prefix) :].strip() == heading.text:
                if current is not None:
                    sections.append(current)
                current = Section(heading=heading.text, level=heading.level, text="")
                heading_index += 1
                continue
        if current is None:
            preamble.append(line)
        else:
            current.text += line + "\n"

    if current is not None:
        sections.append(current)

    preamble_text = "\n".join(preamble).strip()
    if preamble_text:
        if sections:
            sections[0].text = (
                preamble_text + "\n\n" + sections[0].text if sections[0].text else preamble_text
            )
        else:
            return [Section(heading="", level=0, text=preamble_text)]

    for section in sections:
        section.text = section.text.strip()
    return sections


def _resolve_outbound_links(
    links: list[Link],
    slug: str,
    page_slugs: set[str],
) -> list[str]:
    """Resolve ``[[target]]`` links to page slugs, dropping unresolvable
    targets and self-links. Mirrors ``lint_tools._resolve_link`` resolution:
    exact slug match, then slugified short-name match, then unicode-preserving
    lowercase/hyphen short-name match (e.g. ``[[Málaga]]`` -> ``entities/málaga``).
    """
    resolved: list[str] = []
    for link in links:
        target = link.target
        resolved_slug = _resolve_link(target, page_slugs)
        if resolved_slug is not None and resolved_slug != slug:
            if resolved_slug not in resolved:
                resolved.append(resolved_slug)
    return resolved


def _resolve_link(target: str, page_slugs: set[str]) -> str | None:
    """Resolve a link target (display name) to a page slug, or None."""
    if target in page_slugs:
        return target

    s = slugify(target)
    for ps in page_slugs:
        short = ps.rsplit("/", 1)[-1]
        if short == s or ps.endswith("/" + s):
            return ps

    t = target.lower().replace(" ", "-")
    for ps in page_slugs:
        short = ps.rsplit("/", 1)[-1]
        if short == t or ps.endswith("/" + t):
            return ps

    return None
