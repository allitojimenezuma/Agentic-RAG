"""IO layer: source loading, wiki file ops, markdown parsing, index/log management."""

from agentic_rag.io.index_manager import (
    find_in_index,
    read_index,
    remove_entry,
    upsert_entry,
    write_index,
)
from agentic_rag.io.log_manager import append_log, tail_log
from agentic_rag.io.markdown_parser import (
    extract_headings,
    extract_links,
    parse_frontmatter,
    serialize_frontmatter,
    slugify,
)
from agentic_rag.io.source_loader import SourceLoader
from agentic_rag.io.wiki_io import (
    delete_page,
    list_pages,
    page_exists,
    read_page,
    read_page_with_frontmatter,
    write_page,
)

__all__ = [
    "SourceLoader",
    "append_log",
    "delete_page",
    "extract_headings",
    "extract_links",
    "find_in_index",
    "list_pages",
    "page_exists",
    "parse_frontmatter",
    "read_index",
    "read_page",
    "read_page_with_frontmatter",
    "remove_entry",
    "serialize_frontmatter",
    "slugify",
    "tail_log",
    "upsert_entry",
    "write_index",
    "write_page",
]
