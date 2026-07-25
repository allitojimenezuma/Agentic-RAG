"""Markdown parsing utilities using markdown-it-py.

Extracts links, headings, frontmatter, and handles slugification.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import yaml
from markdown_it import MarkdownIt

from agentic_rag.schemas.wiki import Frontmatter, Heading, Link

_md = MarkdownIt()

# Regex for Obsidian-style [[target]] and [[target|alias]]
_LINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]")


def extract_links(content: str) -> list[Link]:
    """Find all [[Target]] and [[Target|alias]] links in markdown content."""
    return [
        Link(target=m.group(1).strip(), alias=m.group(2).strip() if m.group(2) else None)
        for m in _LINK_RE.finditer(content)
    ]


def extract_headings(content: str) -> list[Heading]:
    """Extract headings from markdown content using markdown-it-py."""
    tokens = _md.parse(content)
    headings: list[Heading] = []
    level: int | None = None
    for token in tokens:
        if token.type == "heading_open":
            level = int(token.tag[1])  # "h1" -> 1, "h2" -> 2, etc.
        elif token.type == "inline" and token.children:
            # The inline token after heading_open contains the heading text
            text = "".join(child.content for child in token.children if child.type == "text")
            if level is not None:
                headings.append(Heading(level=level, text=text.strip()))
                level = None  # type: ignore[assignment]
    return headings


def parse_frontmatter(content: str) -> Frontmatter:
    """Parse YAML frontmatter between --- delimiters at the start of content.

    Missing optional fields (updated, tags, sources) get sensible defaults.
    """
    from datetime import date

    if not content.startswith("---"):
        raise ValueError("Content does not start with YAML frontmatter delimiter '---'")

    # Find the closing ---
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("Malformed frontmatter: missing closing '---' delimiter")

    yaml_str = parts[1].strip()
    data = yaml.safe_load(yaml_str)
    if not isinstance(data, dict):
        raise ValueError("Frontmatter is not a YAML mapping")

    # Fill defaults for missing fields
    data.setdefault("updated", date.today())
    data.setdefault("sources", [])
    data.setdefault("tags", [])

    return Frontmatter(**data)


def serialize_frontmatter(fm: Frontmatter) -> str:
    """Serialize a Frontmatter model to YAML string with --- delimiters."""
    data = fm.model_dump()
    # Convert date objects to strings for YAML serialization
    for key in ("updated",):
        if key in data and hasattr(data[key], "isoformat"):
            data[key] = data[key].isoformat()
    yaml_str = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return f"---\n{yaml_str}---\n"


def slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug.

    Examples:
        "3D Gaussian Splatting" -> "3d-gaussian-splatting"
        "Álvaro Jiménez" -> "alvaro-jimenez"
    """
    # Normalize unicode to ASCII
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace non-alphanumeric with hyphens
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    # Collapse multiple hyphens and strip leading/trailing
    name = re.sub(r"-+", "-", name).strip("-")
    return name
