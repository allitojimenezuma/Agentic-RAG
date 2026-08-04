"""Tools for the ingest agent: read_source, create/update/delete page, index/log, flag_contradiction."""

from __future__ import annotations

import logging
from datetime import date, datetime

from langchain_core.tools import tool

from agentic_rag.io.index_manager import upsert_entry as _upsert_entry
from agentic_rag.io.log_manager import append_log as _append_log
from agentic_rag.io.source_loader import SourceLoader
from agentic_rag.io.wiki_io import (
    _resolve_page_path,
    delete_page as _delete_page,
    page_exists,
    read_page_with_frontmatter as _read_page_with_frontmatter,
    write_page as _write_page,
)
from agentic_rag.schemas.wiki import Frontmatter, IndexEntry, LogEntry
from agentic_rag.tools.shared import get_wiki_path

logger = logging.getLogger(__name__)


def _strip_embedded_frontmatter(content: str) -> str:
    """If content begins with a YAML frontmatter block (--- ... ---), strip it — create_page/update_page always write their own frontmatter, so an embedded block would produce a double-frontmatter page."""
    stripped = content.lstrip("\n")
    if not stripped.startswith("---"):
        return content
    lines = stripped.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return content


@tool
def read_source(source_path: str) -> str:
    """Read and convert a source file to markdown using MarkItDown. Supports pdf, docx, pptx, xlsx, html, csv, json, xml, ipynb, images, epub, and more."""
    logger.debug("Reading source file: %s", source_path)
    loader = SourceLoader(settings=type("S", (), {"markitdown_llm_describe_images": False})())
    try:
        result = loader.load(source_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        # Never crash the agent run on a bad source path — return a recoverable error.
        return f"Error: could not read source '{source_path}': {e}"
    logger.debug("Source converted, first 200 chars: %s", result[:200])
    return result


@tool
def create_page(
    slug: str,
    page_type: str,
    title: str,
    content: str,
    sources: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Create a new wiki page. Fails if the page already exists. Use update_page for existing pages."""
    logger.debug("Creating page: %s (type=%s)", slug, page_type)
    if page_exists(get_wiki_path(), slug):
        logger.debug("Page already exists: %s", slug)
        return f"Error: Page '{slug}' already exists. Use update_page to modify it."

    # Ensure parent directory exists (e.g. entities/ for slug 'entities/foo')
    target = get_wiki_path() / f"{slug}.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    fm = Frontmatter(
        slug=slug,
        type=page_type,
        title=title,
        sources=sources or [],
        updated=date.today(),
        tags=tags or [],
    )
    content = _strip_embedded_frontmatter(content)
    _write_page(get_wiki_path(), slug, content, frontmatter=fm)
    logger.debug("Page written: %s", slug)
    return f"Created page: {slug} (type={page_type}, title={title})"


@tool
def update_page(
    slug: str,
    content: str,
    sources: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Update an existing wiki page. Preserves frontmatter fields unless explicitly changed. Fails if the page does not exist."""
    logger.debug("Updating page: %s", slug)
    # Resolve slug to actual path (handles 'mlx' → 'entities/mlx')
    try:
        resolved = _resolve_page_path(get_wiki_path(), slug)
    except FileNotFoundError:
        return f"Error: Page '{slug}' does not exist. Use create_page first."
    # Use resolved slug for write
    resolved_slug = str(resolved.relative_to(get_wiki_path())).removesuffix(".md")

    try:
        fm, _body = _read_page_with_frontmatter(get_wiki_path(), slug)
    except (ValueError, KeyError):
        fm = Frontmatter(
            slug=resolved_slug,
            type="concept",
            title=resolved_slug.rsplit("/", 1)[-1],
            sources=[],
            updated=date.today(),
            tags=[],
        )
    fm.updated = date.today()
    if sources is not None:
        fm.sources = sources
    if tags is not None:
        fm.tags = tags
    content = _strip_embedded_frontmatter(content)
    _write_page(get_wiki_path(), resolved_slug, content, frontmatter=fm)
    return f"Updated page: {resolved_slug}"


@tool
def delete_wiki_page(slug: str) -> str:
    """Delete a wiki page. This action requires human approval (HITL). Will be paused for confirmation."""
    logger.debug("Deleting page: %s", slug)
    if not page_exists(get_wiki_path(), slug):
        return f"Error: Wiki page not found: {slug}"
    _delete_page(get_wiki_path(), slug)
    return f"Deleted page: {slug}"


@tool
def update_index(
    slug: str,
    page_type: str,
    summary: str,
    sources: list[str] | None = None,
) -> str:
    """Update the wiki index with a new or modified entry. Call this after creating or updating a page."""
    logger.debug("Updating index entry: %s", slug)
    entry = IndexEntry(
        slug=slug,
        summary=summary,
        type=page_type,
        sources=sources or [],
        updated=date.today(),
    )
    _upsert_entry(get_wiki_path(), entry)
    return f"Index updated for: {slug}"


@tool
def append_log(op: str, title: str, details: str = "") -> str:
    """Append an entry to the wiki log.md. Use op='ingest' for source ingestion, 'query' for queries, 'lint' for health checks."""
    logger.debug("Appending log entry: %s | %s", op, title)
    entry = LogEntry(
        timestamp=datetime.now(),
        op=op,
        title=title,
        details=details,
    )
    _append_log(get_wiki_path(), entry)
    return f"Log entry appended: [{op}] {title}"


@tool
def flag_contradiction(
    page_slug: str,
    existing_claim: str,
    new_claim: str,
    proposed_resolution: str,
) -> str:
    """Flag a contradiction between existing wiki content and new source material. This action requires human approval (HITL). Will be paused for decision."""
    logger.debug(
        "Flagging contradiction on page: %s — existing: %s, new: %s",
        page_slug,
        existing_claim[:100],
        new_claim[:100],
    )
    return (
        f"CONTRADICTION FLAGGED (requires HITL):\n"
        f"  Page: {page_slug}\n"
        f"  Existing claim: {existing_claim}\n"
        f"  New claim: {new_claim}\n"
        f"  Proposed resolution: {proposed_resolution}\n"
        f"The human decision has been captured by the approval flow. Continue the ingestion: "
        f"approve → proceed with your proposed_resolution; reject → leave the existing page "
        f"unchanged; edit → apply the edited resolution. Do NOT ask for approval again — "
        f"finish with regenerate_index + append_log."
    )
