"""Tools for the lint agent: link summary, read pages, find links, extract concepts, write report."""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.tools import tool

from agentic_rag.io.markdown_parser import extract_headings, extract_links, parse_frontmatter
from agentic_rag.io.wiki_io import list_pages
from agentic_rag.lint.health import _render_report_markdown, health_check
from agentic_rag.schemas.lint import LintReport
from agentic_rag.tools.nav import wiki_link_graph
from agentic_rag.tools.shared import get_wiki_path
from agentic_rag.wiki.model import load_wiki

logger = logging.getLogger(__name__)


@tool
def wiki_link_summary() -> str:
    """Get a summary of ALL pages with their inbound and outbound links in one call.
    Returns each page's slug, type, outbound links (what it links to), and inbound links (what links to it)."""
    # Delegates to the consolidated nav tool (moved implementation, same output).
    return wiki_link_graph.invoke({})


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
def find_inbound_links(slug: str) -> str:
    """Find all pages that link to a given slug via [[slug]] or [[slug|alias]] syntax. Use to detect orphan pages."""
    logger.debug("Finding inbound links to: %s", slug)
    wiki = load_wiki(get_wiki_path())

    # Resolve the query to candidate page slugs (exact slug or short name).
    targets = [
        p.slug for p in wiki.pages
        if p.slug == slug or p.slug.rsplit("/", 1)[-1] == slug
    ]
    if not targets:
        return f"No pages link to '{slug}'. This page may be an orphan."

    # Inbound computed from content pages' RESOLVED outbound links (model);
    # lint-report pages are derived artifacts and never count as sources.
    linking_pages = sorted({
        page.slug
        for page in wiki.pages
        if not page.rel_path.name.startswith("lint-report-")
        for target in page.outbound_links
        if target in targets
    })

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
def write_lint_report(report: LintReport | str) -> str:
    """Write a lint report to wiki/lint-report-YYYY-MM-DD.md with today's date.

    Accepts either a structured ``LintReport`` (rendered deterministically via
    ``_render_report_markdown``) or a raw markdown string (back-compat for the
    scripted test harness)."""
    today = date.today().isoformat()
    report_path = get_wiki_path() / f"lint-report-{today}.md"
    content = report if isinstance(report, str) else _render_report_markdown(report)
    logger.debug("Writing lint report to %s", report_path)
    report_path.write_text(content, encoding="utf-8")
    return f"Lint report written to: lint-report-{today}.md"


@tool
def run_health_check() -> str:
    """Run a deterministic structural health check of the wiki: orphan, missing-index, broken-link, missing-frontmatter, missing-related, empty, and stale pages. Zero LLM calls — instant and free. Call this FIRST in any lint run; it is the ground truth for structural findings."""
    report = health_check(get_wiki_path())
    lines = [f"Pages audited: {report.pages_audited} | Issues: {len(report.issues)}"]
    lines.extend(
        f"[{issue.severity}] {issue.kind}: {issue.slug} — {issue.detail}"
        for issue in report.issues
    )
    return "\n".join(lines)
