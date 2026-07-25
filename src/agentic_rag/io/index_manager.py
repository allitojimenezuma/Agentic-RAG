"""Index manager: read, update, parse wiki/index.md."""

from __future__ import annotations

import logging
import re
import tempfile
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

from agentic_rag.schemas.wiki import Index, IndexEntry

_SECTION_TO_TYPE = {
    "entities": "entity",
    "concepts": "concept",
    "sources": "source",
    "comparisons": "comparison",
    "overviews": "overview",
}


# Section header pattern: "## Entities", "## Concepts", etc.
_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
# Entry pattern: "- [[Name]] - summary | Source: X | Updated: YYYY-MM-DD"
# or: "- [[Name]] - summary | Sources: N | Updated: YYYY-MM-DD"
# or: "- [Title](path) - Ingested: YYYY-MM-DD"
_ENTRY_RE = re.compile(
    r"^-\s+\[\[([^\]]+)\]\]\s+-\s+(.+?)\s+\|\s+"
    r"(?:Source(?:s)?:\s*(.+?)\s+\|)?\s*"
    r"Updated:\s*(\d{4}-\d{2}-\d{2})$",
    re.MULTILINE,
)
_SOURCE_ENTRY_RE = re.compile(
    r"^-\s+\[([^\]]+)\]\(([^)]+)\)\s+-\s+Ingested:\s*(\d{4}-\d{2}-\d{2})$",
    re.MULTILINE,
)


def _parse_source_field(raw: str | None) -> list[str]:
    """Parse the source/sources field from an index entry line."""
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    # Handle "cv.pdf" or "1" (count) or "cv.pdf, doc.pdf"
    # If it's just a number, it's a count — we can't reconstruct filenames
    if raw.isdigit():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def read_index(wiki_path: Path) -> Index:
    """Parse wiki/index.md into an Index model."""
    index_path = wiki_path / "index.md"
    if not index_path.is_file():
        logger.debug("No index.md found at %s", index_path)
        return Index(categories={})

    content = index_path.read_text(encoding="utf-8")
    categories: dict[str, list[IndexEntry]] = {}

    # Split content by section headers
    sections = _SECTION_RE.split(content)
    # sections[0] is the title line, then alternating: header_name, header_content
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip().lower()
        section_content = sections[i + 1] if i + 1 < len(sections) else ""

        entries: list[IndexEntry] = []

        # Parse [[link]] entries
        for m in _ENTRY_RE.finditer(section_content):
            title = m.group(1).strip()
            summary = m.group(2).strip()
            source_raw = m.group(3)
            updated_str = m.group(4).strip()
            slug = title.lower().replace(" ", "-")
            sources = _parse_source_field(source_raw)
            page_type = _SECTION_TO_TYPE.get(section_name, section_name.rstrip("s"))
            entries.append(
                IndexEntry(
                    slug=slug,
                    summary=summary,
                    type=page_type,
                    sources=sources,
                    updated=date.fromisoformat(updated_str),
                    display_name=title,
                )
            )

        # Parse [Title](path) entries (sources section)
        for m in _SOURCE_ENTRY_RE.finditer(section_content):
            title = m.group(1).strip()
            path_str = m.group(2).strip()
            updated_str = m.group(3).strip()
            slug = Path(path_str).stem
            entries.append(
                IndexEntry(
                    slug=slug,
                    summary=title,
                    type="source",
                    sources=[title],
                    updated=date.fromisoformat(updated_str),
                    display_name=title,
                )
            )

        categories[section_name] = entries

    return Index(categories=categories)


def _format_entry(entry: IndexEntry) -> str:
    """Format an IndexEntry as an index.md line."""
    updated_str = entry.updated.isoformat()
    if entry.type == "source":
        sources_str = ", ".join(entry.sources) if entry.sources else entry.slug
        return f"- [{entry.summary}](sources/{entry.slug}.md) - Ingested: {updated_str}"
    else:
        # entity, concept, comparison, overview
        sources_str = ", ".join(entry.sources) if entry.sources else "manual"
        # Preserve original casing if available, otherwise derive from slug
        display_name = entry.display_name or entry.slug.replace("-", " ").title()
        return (
            f"- [[{display_name}]] - {entry.summary} "
            f"| Sources: {sources_str} | Updated: {updated_str}"
        )


def write_index(wiki_path: Path, index: Index) -> None:
    """Write the Index model to wiki/index.md atomically."""
    total = sum(len(e) for e in index.categories.values())
    lines = ["# Wiki Index", ""]

    for category_name, entries in index.categories.items():
        # Capitalize for display: "entities" -> "Entities"
        display_name = category_name.replace("-", " ").title()
        lines.append(f"## {display_name}")
        for entry in entries:
            lines.append(_format_entry(entry))
        lines.append("")

    content = "\n".join(lines) + "\n"

    index_path = wiki_path / "index.md"
    fd, tmp_path = tempfile.mkstemp(dir=wiki_path, suffix=".md.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp_path).replace(index_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


_PLURAL_MAP = {
    "entity": "entities",
    "concept": "concepts",
    "source": "sources",
    "comparison": "comparisons",
    "overview": "overviews",
}


def _category_for_type(page_type: str) -> str:
    """Map a page type to its plural category name."""
    return _PLURAL_MAP.get(page_type, page_type + "s")


def upsert_entry(wiki_path: Path, entry: IndexEntry) -> None:
    """Add or update an entry in the correct category of the index."""
    index = read_index(wiki_path)
    category = _category_for_type(entry.type)
    if category not in index.categories:
        index.categories[category] = []

    # Find existing entry by slug
    entries = index.categories[category]
    for i, existing in enumerate(entries):
        if existing.slug == entry.slug:
            entries[i] = entry
            write_index(wiki_path, index)
            return

    # Not found — append
    entries.append(entry)
    write_index(wiki_path, index)


def remove_entry(wiki_path: Path, slug: str) -> None:
    """Remove an entry by slug from all categories in the index."""
    index = read_index(wiki_path)
    for category_name, entries in index.categories.items():
        index.categories[category_name] = [e for e in entries if e.slug != slug]
    write_index(wiki_path, index)


def find_in_index(wiki_path: Path, query: str) -> list[IndexEntry]:
    """Search the index for entries matching a query (substring/keyword on summary or slug)."""
    index = read_index(wiki_path)
    query_lower = query.lower()
    results: list[IndexEntry] = []
    for entries in index.categories.values():
        for entry in entries:
            if query_lower in entry.summary.lower() or query_lower in entry.slug.lower():
                results.append(entry)
    return results
