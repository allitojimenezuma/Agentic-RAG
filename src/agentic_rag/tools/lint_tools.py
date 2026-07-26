"""Tools for the lint agent: link summary, read pages, write report."""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.tools import tool

from agentic_rag.io.markdown_parser import extract_links, parse_frontmatter, slugify
from agentic_rag.io.wiki_io import list_pages
from agentic_rag.tools.shared import get_wiki_path

logger = logging.getLogger(__name__)


@tool
def wiki_link_summary() -> str:
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
                _DIR_TO_TYPE = {"entities": "entity", "concepts": "concept", "sources": "source", "comparisons": "comparison"}
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
    lines.append(f"--- SUMMARY ---")
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


@tool
def read_all_pages() -> str:
    """Read all wiki pages. Returns slug, type, title, updated, and outbound links for each page."""
    wiki_path = get_wiki_path()
    logger.debug("Reading all wiki pages from %s", wiki_path)
    pages = list_pages(wiki_path)
    if not pages:
        return "No wiki pages found."

    lines: list[str] = []
    for page_path in pages:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        content = page_path.read_text(encoding="utf-8")
        page_type = "unknown"
        title = slug.rsplit("/", 1)[-1] if "/" in slug else slug
        updated = "unknown"
        if content.startswith("---"):
            try:
                fm = parse_frontmatter(content)
                page_type = fm.type or page_type
                title = fm.title or title
                updated = str(fm.updated)
            except Exception:
                pass
        else:
            if "/" in slug:
                _DIR_TO_TYPE = {"entities": "entity", "concepts": "concept", "sources": "source", "comparisons": "comparison"}
                page_type = _DIR_TO_TYPE.get(slug.split("/")[0], "unknown")
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        links = extract_links(content)
        outbound = [l.target for l in links]
        lines.append(
            f"=== {slug} === type={page_type} title={title} updated={updated}"
            + (f" links={outbound}" if outbound else " links=[]")
        )

    return "\n".join(lines)



@tool
def write_lint_report(report: str) -> str:
    """Write a lint report to wiki/lint-report-YYYY-MM-DD.md with today's date."""
    today = date.today().isoformat()
    report_path = get_wiki_path() / f"lint-report-{today}.md"
    logger.debug("Writing lint report to %s", report_path)
    report_path.write_text(report, encoding="utf-8")
    return f"Lint report written to: lint-report-{today}.md"
