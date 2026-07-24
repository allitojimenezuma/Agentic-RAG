"""Path resolution helpers for agentic-rag."""

from __future__ import annotations

from pathlib import Path


def resolve_wiki_path(base: Path, slug: str) -> Path:
    """Resolve a wiki page slug to its file path under the wiki directory."""
    return base / f"{slug}.md"


def slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    return (
        name.lower()
        .strip()
        .replace(" ", "-")
        .replace("_", "-")
        .replace(".", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("'", "")
        .replace('"', "")
    )
