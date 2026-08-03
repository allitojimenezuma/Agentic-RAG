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
from agentic_rag.wiki.model import Page, Wiki, load_wiki
from agentic_rag.wiki.search import search

logger = logging.getLogger(__name__)

_DIR_TO_TYPE = {
    "entities": "entity",
    "concepts": "concept",
    "sources": "source",
    "comparisons": "comparison",
}


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

    if section is None:
        content = _read_page(get_wiki_path(), slug)
        record_navigated([_resolved_slug(get_wiki_path(), slug)])
        return content

    wiki = load_wiki(get_wiki_path())
    page = _find_page(wiki, slug)
    if page is None:
        raise FileNotFoundError(f"Wiki page not found: {slug}")
    record_navigated([page.slug])
    target = section.lower()
    for s in page.sections:
        if s.heading.lower() == target:
            return s.text
    headings = "; ".join(s.heading for s in page.sections if s.heading) or "none"
    return f"Section '{section}' not found on page '{slug}'. Available headings: {headings}"


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
                page_type = _DIR_TO_TYPE.get(dir_name, dir_name.rstrip("s"))
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
    """Resolve a slug to its canonical page slug (mirror of wiki_io._resolve_page_path)."""
    if (wiki_path / f"{slug}.md").is_file():
        return slug
    for md_file in wiki_path.rglob(f"{slug}.md"):
        if md_file.is_file():
            return str(md_file.relative_to(wiki_path)).removesuffix(".md")
    return slug  # unreachable: read_page already raised if the page is missing


def _find_page(wiki: Wiki, slug: str) -> Page | None:
    """Resolve a slug against the model like ``wiki_io._resolve_page_path``:
    exact relative-path slug first, then recursive basename match."""
    if slug in wiki.by_slug:
        return wiki.by_slug[slug]
    for page in wiki.pages:
        if page.slug.rsplit("/", 1)[-1] == slug:
            return page
    return None
