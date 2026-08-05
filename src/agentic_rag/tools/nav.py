"""Navigation tools over the Wiki model + BM25 search.

Consolidates navigation into four deterministic, token-efficient tools:
``wiki_search`` (ranked pages + bounded link expansion), ``wiki_read_page``
(section-scoped reading), ``wiki_summary`` (compact page catalog) and
``wiki_link_graph`` (deterministic inbound/outbound summary, moved here from
``lint_tools.wiki_link_summary``). The ``_WIKI_PATH`` global from
``tools.shared`` is reused (no second global).
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.io.markdown_parser import extract_links, parse_frontmatter, slugify
from agentic_rag.io.wiki_io import list_pages, read_page as _read_page
from agentic_rag.tools.shared import get_wiki_path
from agentic_rag.wiki.dedupe_index import regenerate_index as _regenerate_index
from agentic_rag.wiki.model import DIR_TO_TYPE, Page, Wiki, load_wiki
from agentic_rag.wiki.search import search

logger = logging.getLogger(__name__)


@tool
def regenerate_index() -> str:
    """Regenerate the wiki index.md from the pages on disk. Call this after creating or updating pages (replaces update_index)."""
    logger.debug("Regenerating wiki index")
    _regenerate_index(get_wiki_path())
    return "Index regenerated."


@tool
def wiki_search(
    query: str, k: int = 8, types: str | None = None, tags: str | None = None
) -> str:
    """Search the wiki for pages relevant to a query. Returns ranked page slugs with BM25 scores and the sections that matched, plus a bounded set of linked pages. Use this to find relevant pages before reading them. Optionally filter by comma-separated page types (entity, concept, source, comparison) or tags."""
    logger.debug("Searching wiki for: %s (k=%d, types=%s, tags=%s)", query, k, types, tags)
    type_list = _split_csv(types)
    tag_list = _split_csv(tags)
    wiki = load_wiki(get_wiki_path())
    hits = search(wiki, query, k=k, types=type_list, tags=tag_list)
    if not hits:
        return f"No relevant pages found for '{query}'."

    from agentic_rag.tools.grounding import record_navigated

    record_navigated(h.slug for h in hits)

    direct = [h for h in hits if h.matched_via != "expand-link"]
    linked = [h for h in hits if h.matched_via == "expand-link"]
    lines: list[str] = []
    for i, h in enumerate(direct):
        prefix = f"Found {len(hits)} relevant: " if i == 0 else "- "
        lines.append(
            f"{prefix}{h.slug} (score={h.score:.2f}, sections: {'; '.join(h.sections)})"
        )
    lines.extend(f"+ linked: {h.slug}" for h in linked)
    return "\n".join(lines)


@tool
def wiki_read_page(slug: str, section: str | None = None) -> str:
    """Read a wiki page by slug. Without a section, returns the full raw markdown including frontmatter. With a section name, returns only that section's text (heading line excluded). Use this to get detailed information about any entity, concept, or source."""
    logger.debug("Reading wiki page: %s (section=%s)", slug, section)
    from agentic_rag.tools.grounding import record_navigated

    try:
        if section is None:
            content = _read_page(get_wiki_path(), slug)
            record_navigated([_resolved_slug(get_wiki_path(), slug)])
            return content

        wiki = load_wiki(get_wiki_path())
        page = _find_page(wiki, slug)
        if page is None:
            return _page_not_found_error(slug)
        record_navigated([page.slug])
        target = section.lower()
        for s in page.sections:
            if s.heading.lower() == target:
                return s.text
        headings = "; ".join(s.heading for s in page.sections if s.heading) or "none"
        return f"Section '{section}' not found on page '{slug}'. Available headings: {headings}"
    except FileNotFoundError:
        # Never crash the agent run on a bad slug — return a recoverable error
        # (with a suggestion when a page with the same basename exists).
        return _page_not_found_error(slug)


def _page_not_found_error(slug: str) -> str:
    """Helpful not-found message: suggest an existing slug with the same basename."""
    wiki = load_wiki(get_wiki_path())
    short = slug.rsplit("/", 1)[-1]
    matches = sorted(p.slug for p in wiki.pages if p.slug.rsplit("/", 1)[-1] == short)
    if matches:
        return (
            f"Error: Wiki page not found: {slug}. "
            f"Did you mean: {', '.join(matches)}?"
        )
    return (
        f"Error: Wiki page not found: {slug}. "
        "Check the slug — use wiki_scan() or wiki_search() to list pages."
    )


@tool
def wiki_summary() -> str:
    """List all wiki pages, one compact line per page: slug (type) - title, with the last update date. Use this to get an overview of what the wiki covers."""
    logger.debug("Building wiki summary")
    wiki = load_wiki(get_wiki_path())
    if not wiki.pages:
        return "No wiki pages found."

    lines: list[str] = []
    for page in sorted(wiki.pages, key=lambda p: p.slug):
        updated = f" | updated: {page.fm.updated.isoformat()}" if page.fm.updated else ""
        lines.append(f"- {page.slug} ({page.fm.type}) - {page.fm.title}{updated}")
    return "\n".join(lines)


@tool
def wiki_scan(max_chars: int = 200) -> str:
    """Get a one-call overview of ALL content pages: slug, type, title, a preview of each page's first section, inbound/outbound link counts, and last update date. Use this INSTEAD of reading every page to survey the wiki — it replaces per-page reads for overview purposes. Deterministic and free (0 LLM calls)."""
    logger.debug("Scanning wiki (max_chars=%d)", max_chars)
    wiki = load_wiki(get_wiki_path())
    content_pages = [
        p for p in wiki.pages if not p.rel_path.name.startswith("lint-report-")
    ]
    if not content_pages:
        return "No wiki pages found."

    # Inbound: from content pages' RESOLVED outbound links only (mirrors lint/health.py).
    content_slugs = {p.slug for p in content_pages}
    inbound: dict[str, set[str]] = {}
    for page in content_pages:
        for target in page.outbound_links:
            if target in content_slugs:
                inbound.setdefault(target, set()).add(page.slug)

    lines: list[str] = []
    for page in sorted(content_pages, key=lambda p: p.slug):
        preview = _preview_text(page, max_chars)
        updated = page.fm.updated.isoformat() if page.fm.updated else "-"
        out_n = len(page.outbound_links)
        in_n = len(inbound.get(page.slug, ()))
        lines.append(
            f'- {page.slug} ({page.fm.type}) - {page.fm.title} — "{preview}" — '
            f"out: {out_n} | in: {in_n} | updated: {updated}"
        )
    logger.debug("Scanned %d content pages", len(content_pages))
    return "\n".join(lines)


def _preview_text(page: Page, max_chars: int) -> str:
    """First-section preview: whitespace collapsed to single spaces, truncated to
    ``max_chars`` with a ``…`` suffix when cut; ``(no content)`` if no section text."""
    if not page.sections or not page.sections[0].text:
        return "(no content)"
    text = " ".join(page.sections[0].text.split())
    if not text:
        return "(no content)"
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


@tool
def wiki_link_graph() -> str:
    """Get a summary of ALL pages with their inbound and outbound links in one call.
    Returns each page's slug, type, outbound links (what it links to), and inbound links (what links to it)."""
    wiki_path = get_wiki_path()
    logger.info("Building link summary for %s", wiki_path)
    pages = list_pages(wiki_path)
    if not pages:
        logger.debug("No pages found in %s", wiki_path)
        return "No wiki pages found."
    logger.debug("Found %d pages to analyze", len(pages))

    # Build slug set for link resolution (must be complete BEFORE resolving links)
    page_slugs = set()
    for p in pages:
        page_slugs.add(str(p.relative_to(wiki_path)).removesuffix(".md"))
    page_data = {}  # slug -> {outbound: set, type: str, title: str}

    def _resolve_link(target: str) -> str | None:
        """Resolve a link target (display name) to an actual page slug.

        Handles both ASCII slugs (slugify output) and Unicode filenames
        (e.g. málaga.md) by trying both normalized forms.
        """
        # Exact match
        if target in page_slugs:
            return target

        # Try slugified match (ASCII normalized)
        s = slugify(target)
        for ps in page_slugs:
            short = ps.rsplit("/", 1)[-1] if "/" in ps else ps
            if short == s or ps.endswith("/" + s):
                return ps

        # Try Unicode-preserving match: lowercase + replace spaces with hyphens
        # but keep unicode chars (e.g. "Málaga" -> "málaga")
        t = target.lower().replace(" ", "-")
        for ps in page_slugs:
            short = ps.rsplit("/", 1)[-1] if "/" in ps else ps
            if short == t or ps.endswith("/" + t):
                return ps

        return None

    for page_path in pages:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        content = page_path.read_text(encoding="utf-8")

        # Extract links (works with or without frontmatter)
        links = extract_links(content)
        outbound = set()
        for link in links:
            resolved = _resolve_link(link.target)
            if resolved and resolved != slug:
                outbound.add(resolved)

        # Try to parse frontmatter for type/title; fallback to directory + first heading
        page_type = "unknown"
        title = slug.rsplit("/", 1)[-1] if "/" in slug else slug
        if content.startswith("---"):
            try:
                fm = parse_frontmatter(content)
                page_type = fm.type or page_type
                title = fm.title or title
            except Exception:
                pass
        else:
            # Infer type from directory: entities/foo -> entity, concepts/foo -> concept
            if "/" in slug:
                dir_name = slug.split("/")[0]
                page_type = DIR_TO_TYPE.get(dir_name, dir_name.rstrip("s"))
            # Infer title from first heading
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

        page_data[slug] = {
            "outbound": outbound,
            "type": page_type,
            "title": title,
        }

    # Compute inbound links
    inbound = {slug: set() for slug in page_slugs}
    for slug, data in page_data.items():
        for target in data["outbound"]:
            if target in inbound:
                inbound[target].add(slug)

    # Format output
    lines: list[str] = []
    for slug in sorted(page_slugs):
        data = page_data[slug]
        in_links = sorted(inbound.get(slug, set()))
        out_links = sorted(data["outbound"])
        lines.append(f"### {slug} ({data['type']})")
        lines.append(f"  Outbound ({len(out_links)}): {', '.join(out_links) if out_links else 'none'}")
        lines.append(f"  Inbound  ({len(in_links)}): {', '.join(in_links) if in_links else 'none — ORPHAN?'}")
        lines.append("")

    # Summary stats
    orphans = [s for s in page_slugs if not inbound.get(s)]
    lines.append("--- SUMMARY ---")
    lines.append(f"Total pages: {len(page_slugs)}")
    lines.append(f"Total links: {sum(len(d['outbound']) for d in page_data.values())}")
    lines.append(f"Orphans (no inbound links): {len(orphans)}")
    if orphans:
        lines.append(f"  {', '.join(sorted(orphans))}")

    logger.info(
        "Link summary: %d pages, %d links, %d orphans",
        len(page_slugs),
        sum(len(d['outbound']) for d in page_data.values()),
        len(orphans),
    )
    return "\n".join(lines)


def _split_csv(value: str | None) -> list[str] | None:
    """Split a comma-separated tool arg into a stripped list (None if empty)."""
    if not value:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def _resolved_slug(wiki_path: Path, slug: str) -> str:
    """Resolve a slug to its canonical page slug (delegates to ``wiki_io._resolve_page_path``)."""
    try:
        from agentic_rag.io.wiki_io import _resolve_page_path

        resolved = _resolve_page_path(wiki_path, slug)
        return str(resolved.relative_to(wiki_path)).removesuffix(".md")
    except FileNotFoundError:
        return slug  # unreachable: read_page already raised if the page is missing


def _find_page(wiki: Wiki, slug: str) -> Page | None:
    """Resolve a slug against the model like ``wiki_io._resolve_page_path``:
    exact relative-path slug first, then recursive basename match."""
    return wiki.by_slug.get(_resolved_slug(get_wiki_path(), slug))
