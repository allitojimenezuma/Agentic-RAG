"""Tools for the lint agent: link summary, read pages, find links, extract concepts, write report."""

from __future__ import annotations

import logging
import re
from datetime import date

from langchain_core.tools import tool

from agentic_rag.io.markdown_parser import extract_headings, extract_links, parse_frontmatter
from agentic_rag.io.wiki_io import list_pages, read_page
from agentic_rag.tools.shared import get_wiki_path

logger = logging.getLogger(__name__)


@tool
def wiki_link_summary() -> str:
    """Get a summary of ALL pages with their inbound and outbound links in one call.
    Returns each page's slug, type, outbound links (what it links to), and inbound links (what links to it).
    Use this instead of calling find_inbound_links per page — much more efficient."""
    wiki_path = get_wiki_path()
    pages = list_pages(wiki_path)
    if not pages:
        return "No wiki pages found."

    # Build slug set for link resolution
    page_slugs = set()
    page_data = {}  # slug -> {outbound: set, type: str, title: str}

    def _resolve_link(target: str) -> str | None:
        """Resolve a link target (display name) to an actual page slug."""
        # Exact match
        if target in page_slugs:
            return target
        # Slugified match: "3D Gaussian Splatting" -> "3d-gaussian-splatting"
        s = target.lower().replace(" ", "-")
        for ps in page_slugs:
            short = ps.rsplit("/", 1)[-1] if "/" in ps else ps
            if short == s or ps.endswith("/" + s):
                return ps
        return None

    for page_path in pages:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        page_slugs.add(slug)
        try:
            content = page_path.read_text(encoding="utf-8")
            fm = parse_frontmatter(content)
            links = extract_links(content)
            outbound = set()
            for link in links:
                resolved = _resolve_link(link.target)
                if resolved and resolved != slug:
                    outbound.add(resolved)
            page_data[slug] = {
                "outbound": outbound,
                "type": fm.type or "unknown",
                "title": fm.title or slug,
            }
        except Exception:
            page_data[slug] = {"outbound": set(), "type": "unknown", "title": slug}

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
    lines.append(f"Orphans (no inbound links): {len(orphans)}")
    if orphans:
        lines.append(f"  {', '.join(sorted(orphans))}")

    return "\n".join(lines)


@tool
def read_all_pages(full: bool = False) -> str:
    """Read all wiki pages. Returns slug and metadata for each page.

    Args:
        full: If False (default), returns only frontmatter (slug, type, title, updated)
              plus outbound links — cheap, use for health checks.
              If True, returns full content — expensive, use only when needed.
    """
    wiki_path = get_wiki_path()
    logger.debug("Reading all wiki pages (full=%s) from %s", full, wiki_path)
    pages = list_pages(wiki_path)
    if not pages:
        return "No wiki pages found."

    lines: list[str] = []
    for page_path in pages:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        content = page_path.read_text(encoding="utf-8")

        if full:
            lines.append(f"=== {slug} ===\n{content}\n")
        else:
            # Lightweight: frontmatter + outbound links only
            try:
                fm = parse_frontmatter(content)
                links = extract_links(content)
                outbound = [l.target for l in links]
                lines.append(
                    f"=== {slug} === type={fm.type} title={fm.title} updated={fm.updated}"
                    + (f" links={outbound}" if outbound else " links=[]")
                )
            except Exception:
                lines.append(f"=== {slug} === (parse error)")

    return "\n".join(lines)


@tool
def find_inbound_links(slug: str) -> str:
    """Find all pages that link to a given slug via [[slug]] or [[slug|alias]] syntax. Use to detect orphan pages."""
    logger.debug("Finding inbound links to: %s", slug)
    pattern = re.compile(r"\[\[" + re.escape(slug) + r"(?:\|[^\]]+)?\]\]")
    pages = list_pages(get_wiki_path())
    linking_pages: list[str] = []

    for page_path in pages:
        content = page_path.read_text(encoding="utf-8")
        if pattern.search(content):
            page_slug = str(page_path.relative_to(get_wiki_path())).removesuffix(".md")
            linking_pages.append(page_slug)

    logger.debug("Found pages linking to '%s': %s", slug, linking_pages)
    if not linking_pages:
        return f"No pages link to '{slug}'. This page may be an orphan."
    return f"Found {len(linking_pages)} page(s) linking to '{slug}':\n" + "\n".join(
        f"- {p}" for p in linking_pages
    )


@tool
def extract_concepts(content: str) -> str:
    """Extract concept names from page content. Returns headings and [[link]] targets found in the content."""
    logger.debug("Extracting concepts from content (%d chars)", len(content))
    headings = extract_headings(content)
    links = extract_links(content)
    logger.debug("Found %d headings, %d links", len(headings), len(links))

    lines: list[str] = []
    if headings:
        lines.append("Headings:")
        for h in headings:
            lines.append(f"  {'#' * h.level} {h.text}")
    if links:
        lines.append("Links:")
        for link in links:
            alias_part = f" (alias: {link.alias})" if link.alias else ""
            lines.append(f"  [[{link.target}]]{alias_part}")

    return "\n".join(lines) if lines else "No concepts found."


@tool
def write_lint_report(report: str) -> str:
    """Write a lint report to wiki/lint-report-YYYY-MM-DD.md with today's date."""
    today = date.today().isoformat()
    report_path = get_wiki_path() / f"lint-report-{today}.md"
    logger.debug("Writing lint report to %s", report_path)
    report_path.write_text(report, encoding="utf-8")
    return f"Lint report written to: lint-report-{today}.md"
