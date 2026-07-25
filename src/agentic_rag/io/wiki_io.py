"""Filesystem ops on wiki/ directory: read, write, delete, list pages."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from agentic_rag.io.markdown_parser import parse_frontmatter, serialize_frontmatter
from agentic_rag.schemas.wiki import Frontmatter

_EXCLUDED_FILES = {"index.md", "log.md"}


def _validate_slug(slug: str) -> None:
    """Reject slugs that could escape the wiki directory.

    Slugs like ``entities/python`` use ``/`` for subdirectory structure and are
    allowed.  Only path-traversal components (``..``) and absolute paths are
    rejected.
    """
    if not slug:
        raise ValueError("Slug must not be empty")
    if ".." in slug:
        raise ValueError(f"Slug must not contain '..': {slug}")
    if slug.startswith("/") or slug.startswith("\\"):
        raise ValueError(f"Slug must not be an absolute path: {slug}")


def list_pages(wiki_path: Path) -> list[Path]:
    """List all wiki page .md files recursively, excluding index.md and log.md."""
    pages: list[Path] = []
    if not wiki_path.is_dir():
        return pages
    for md_file in sorted(wiki_path.rglob("*.md")):
        if md_file.name not in _EXCLUDED_FILES:
            pages.append(md_file)
    logger.debug("Listed %d pages in %s", len(pages), wiki_path)
    return pages


def _resolve_page_path(wiki_path: Path, slug: str) -> Path:
    """Resolve a slug to a page file path.

    If the slug contains '/', treat it as a relative path (e.g. 'entities/mlx').
    Otherwise, search recursively for the first matching .md file.
    """
    # Direct path (slug already has directory)
    direct = wiki_path / f"{slug}.md"
    if direct.is_file():
        logger.debug("Resolved slug '%s' -> %s (direct)", slug, direct)
        return direct
    # Recursive search for simple slugs like 'mlx'
    for md_file in wiki_path.rglob(f"{slug}.md"):
        if md_file.is_file():
            logger.debug("Resolved slug '%s' -> %s (recursive)", slug, md_file)
            return md_file
    raise FileNotFoundError(f"Wiki page not found: {slug}")


def read_page(wiki_path: Path, slug: str) -> str:
    """Read raw markdown content of a wiki page by slug.

    The slug can be nested, e.g. 'entities/python' resolves to wiki_path/entities/python.md.
    Simple slugs like 'mlx' are searched recursively.
    """
    _validate_slug(slug)
    page_path = _resolve_page_path(wiki_path, slug)
    logger.debug("Reading page file: %s", page_path)
    return page_path.read_text(encoding="utf-8")


def read_page_with_frontmatter(wiki_path: Path, slug: str) -> tuple[Frontmatter, str]:
    """Read a wiki page and parse its frontmatter.

    Returns (Frontmatter, remaining_content).
    """
    _validate_slug(slug)
    raw = read_page(wiki_path, slug)
    fm = parse_frontmatter(raw)
    # Strip frontmatter from content
    parts = raw.split("---", 2)
    if len(parts) >= 3:
        body = parts[2].lstrip("\n")
    else:
        body = raw
    return fm, body


def write_page(
    wiki_path: Path,
    slug: str,
    content: str,
    frontmatter: Optional[Frontmatter] = None,
) -> Path:
    """Write a wiki page atomically (temp + rename). Creates parent dirs as needed.

    Returns the path to the written file.
    """
    _validate_slug(slug)
    page_path = wiki_path / f"{slug}.md"
    logger.info("Writing page file: %s", page_path)
    page_path.parent.mkdir(parents=True, exist_ok=True)

    # Build full content with optional frontmatter
    full_content = ""
    if frontmatter is not None:
        full_content = serialize_frontmatter(frontmatter)
    full_content += content

    # Atomic write: write to temp file in same directory, then rename
    fd, tmp_path = tempfile.mkstemp(dir=page_path.parent, suffix=".md.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(full_content)
        Path(tmp_path).replace(page_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    return page_path


def delete_page(wiki_path: Path, slug: str) -> None:
    """Delete a wiki page file by slug."""
    _validate_slug(slug)
    try:
        page_path = _resolve_page_path(wiki_path, slug)
        logger.info("Deleting page file: %s", page_path)
        page_path.unlink()
    except FileNotFoundError:
        pass


def page_exists(wiki_path: Path, slug: str) -> bool:
    """Check if a wiki page exists by slug."""
    _validate_slug(slug)
    try:
        _resolve_page_path(wiki_path, slug)
        return True
    except FileNotFoundError:
        return False
