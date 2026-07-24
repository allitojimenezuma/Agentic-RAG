"""Tools for the ingest agent: read_source, create/update/delete page, index/log, flag_contradiction."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.io.index_manager import upsert_entry as _upsert_entry
from agentic_rag.io.log_manager import append_log as _append_log
from agentic_rag.io.source_loader import SourceLoader
from agentic_rag.io.wiki_io import (
    delete_page as _delete_page,
    page_exists,
    read_page_with_frontmatter as _read_page_with_frontmatter,
    write_page as _write_page,
)
from agentic_rag.schemas.wiki import Frontmatter, IndexEntry, LogEntry


@tool
def read_source(source_path: str) -> str:
    """Read and convert a source file to markdown using MarkItDown. Supports pdf, docx, pptx, xlsx, html, csv, json, xml, ipynb, images, epub, and more."""
    # Minimal settings for the loader (no LLM image describe)
    loader = SourceLoader(settings=type("S", (), {"markitdown_llm_describe_images": False})())
    return loader.load(source_path)


@tool
def create_page(
    wiki_path: str,
    slug: str,
    page_type: str,
    title: str,
    content: str,
    sources: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Create a new wiki page. Fails if the page already exists. Use update_page for existing pages."""
    path = Path(wiki_path)
    if page_exists(path, slug):
        return f"Error: Page '{slug}' already exists. Use update_page to modify it."

    fm = Frontmatter(
        slug=slug,
        type=page_type,
        title=title,
        sources=sources or [],
        updated=date.today(),
        tags=tags or [],
    )
    _write_page(path, slug, content, frontmatter=fm)
    return f"Created page: {slug} (type={page_type}, title={title})"


@tool
def update_page(
    wiki_path: str,
    slug: str,
    content: str,
    sources: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update an existing wiki page. Preserves frontmatter fields unless explicitly changed. Fails if the page does not exist."""
    path = Path(wiki_path)
    if not page_exists(path, slug):
        return f"Error: Page '{slug}' does not exist. Use create_page first."

    fm, _body = _read_page_with_frontmatter(path, slug)
    fm.updated = date.today()
    if sources is not None:
        fm.sources = sources
    if tags is not None:
        fm.tags = tags
    _write_page(path, slug, content, frontmatter=fm)
    return f"Updated page: {slug}"


@tool
def delete_wiki_page(wiki_path: str, slug: str) -> str:
    """Delete a wiki page. This action requires human approval (HITL). Will be paused for confirmation."""
    path = Path(wiki_path)
    _delete_page(path, slug)
    return f"Deleted page: {slug}"


@tool
def update_index(
    wiki_path: str,
    slug: str,
    page_type: str,
    summary: str,
    sources: list[str] | None = None,
) -> str:
    """Update the wiki index with a new or modified entry. Call this after creating or updating a page."""
    path = Path(wiki_path)
    entry = IndexEntry(
        slug=slug,
        summary=summary,
        type=page_type,
        sources=sources or [],
        updated=date.today(),
    )
    _upsert_entry(path, entry)
    return f"Index updated for: {slug}"


@tool
def append_log(wiki_path: str, op: str, title: str, details: str = "") -> str:
    """Append an entry to the wiki log.md. Use op='ingest' for source ingestion, 'query' for queries, 'lint' for health checks."""
    path = Path(wiki_path)
    entry = LogEntry(
        timestamp=datetime.now(),
        op=op,
        title=title,
        details=details,
    )
    _append_log(path, entry)
    return f"Log entry appended: [{op}] {title}"


@tool
def flag_contradiction(
    wiki_path: str,
    page_slug: str,
    existing_claim: str,
    new_claim: str,
    proposed_resolution: str,
) -> str:
    """Flag a contradiction between existing wiki content and new source material. This action requires human approval (HITL). Will be paused for decision."""
    return (
        f"CONTRADICTION FLAGGED (requires HITL):\n"
        f"  Page: {page_slug}\n"
        f"  Existing claim: {existing_claim}\n"
        f"  New claim: {new_claim}\n"
        f"  Proposed resolution: {proposed_resolution}\n"
        f"Awaiting human decision: approve, edit, or reject."
    )
