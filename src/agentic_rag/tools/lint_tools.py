"""Tools for the lint agent: read_all_pages, find_inbound_links, extract_concepts, write_lint_report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.io.markdown_parser import extract_headings, extract_links
from agentic_rag.io.wiki_io import list_pages, read_page


@tool
def read_all_pages(wiki_path: str) -> str:
    """Read ALL wiki pages. Returns a dict of slug→content. Use sparingly - this is expensive for large wikis."""
    path = Path(wiki_path)
    pages = list_pages(path)
    if not pages:
        return "No wiki pages found."

    lines: list[str] = []
    for page_path in pages:
        slug = str(page_path.relative_to(path)).removesuffix(".md")
        content = page_path.read_text(encoding="utf-8")
        lines.append(f"=== {slug} ===\n{content}\n")
    return "\n".join(lines)


@tool
def find_inbound_links(wiki_path: str, slug: str) -> str:
    """Find all pages that link to a given slug via [[slug]] or [[slug|alias]] syntax. Use to detect orphan pages."""
    import re

    path = Path(wiki_path)
    pattern = re.compile(r"\[\[" + re.escape(slug) + r"(?:\|[^\]]+)?\]\]")
    pages = list_pages(path)
    linking_pages: list[str] = []

    for page_path in pages:
        content = page_path.read_text(encoding="utf-8")
        if pattern.search(content):
            page_slug = str(page_path.relative_to(path)).removesuffix(".md")
            linking_pages.append(page_slug)

    if not linking_pages:
        return f"No pages link to '{slug}'. This page may be an orphan."
    return f"Found {len(linking_pages)} page(s) linking to '{slug}':\n" + "\n".join(
        f"- {p}" for p in linking_pages
    )


@tool
def extract_concepts(content: str) -> str:
    """Extract concept names from page content. Returns headings and [[link]] targets found in the content."""
    headings = extract_headings(content)
    links = extract_links(content)

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
def write_lint_report(wiki_path: str, report: str) -> str:
    """Write a lint report to wiki/lint-report-YYYY-MM-DD.md with today's date."""
    path = Path(wiki_path)
    today = date.today().isoformat()
    report_path = path / f"lint-report-{today}.md"
    report_path.write_text(report, encoding="utf-8")
    return f"Lint report written to: lint-report-{today}.md"
