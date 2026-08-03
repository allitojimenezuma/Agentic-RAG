"""Tools for the fix agent: safe in-wiki page edits (no shell access)."""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.tools import tool

from agentic_rag.io.wiki_io import read_page, write_page
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools.shared import get_wiki_path

logger = logging.getLogger(__name__)


@tool
def add_frontmatter(slug: str, title: str, page_type: str) -> str:
    """Add YAML frontmatter to a wiki page that currently lacks it.

    Args:
        slug: Page slug (e.g., 'entities/python', 'concepts/ai')
        title: Human-readable display title
        page_type: Page type (entity, concept, source, comparison, overview)
    """
    wiki_path = get_wiki_path()
    try:
        body = read_page(wiki_path, slug)
    except FileNotFoundError:
        return f"Page not found: {slug}"

    if body.startswith("---"):
        return f"Error: {slug} already has frontmatter"

    fm = Frontmatter(
        slug=slug,
        type=page_type,
        title=title,
        sources=[],
        updated=date.today(),
        tags=[],
    )
    write_page(wiki_path, slug, body, frontmatter=fm)
    logger.info("Added frontmatter to %s (type=%s)", slug, page_type)
    return f"Added frontmatter to {slug}."


@tool
def fix_link(slug: str, old_target: str, new_target: str) -> str:
    """Fix broken wiki links by replacing a link target.

    Replaces one occurrence of each form: [[old_target]] -> [[new_target]] and
    [[old_target|alias]] -> [[new_target|alias]]. Returns the number of links
    replaced (0, 1, or 2).

    Args:
        slug: Page slug containing the broken links
        old_target: Current link target to replace
        new_target: Replacement link target
    """
    wiki_path = get_wiki_path()
    try:
        content = read_page(wiki_path, slug)
    except FileNotFoundError:
        return f"Page not found: {slug}"

    count = 0
    plain_old = f"[[{old_target}]]"
    if plain_old in content:
        content = content.replace(plain_old, f"[[{new_target}]]", 1)
        count += 1

    alias_prefix = f"[[{old_target}|"
    idx = content.find(alias_prefix)
    if idx != -1:
        end = content.find("]]", idx)
        if end != -1:
            alias = content[idx + len(alias_prefix):end]
            content = content[:idx] + f"[[{new_target}|{alias}]]" + content[end + 2:]
            count += 1

    if count == 0:
        return f"No links to '{old_target}' found in {slug}"

    write_page(wiki_path, slug, content)
    logger.info("Fixed %d link(s) in %s: %s -> %s", count, slug, old_target, new_target)
    return f"Replaced {count} link(s) in {slug}."


@tool
def append_related_section(slug: str, links: list[str]) -> str:
    """Add a '## Related' section with cross-links, or append new links to an existing one.

    Args:
        slug: Page slug to update
        links: Page slugs (or display names) to add as - [[link]] entries
    """
    wiki_path = get_wiki_path()
    try:
        content = read_page(wiki_path, slug)
    except FileNotFoundError:
        return f"Page not found: {slug}"

    bullet_lines = [f"- [[{link}]]" for link in links]
    lines = content.split("\n")
    rel_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == "## Related"), None
    )
    if rel_idx is None:
        lines.append("## Related")
        lines.append("")
        lines.extend(bullet_lines)
    else:
        end = len(lines)
        for i in range(rel_idx + 1, len(lines)):
            if lines[i].startswith("## "):
                end = i
                break
        lines[end:end] = bullet_lines

    write_page(wiki_path, slug, "\n".join(lines))
    logger.info("Appended %d related link(s) to %s", len(links), slug)
    return f"Appended {len(links)} related link(s) to {slug}."


@tool
def edit_wiki_page(slug: str, old_text: str, new_text: str) -> str:
    """Replace text in a wiki page. Use for fixing content, frontmatter, links.

    Args:
        slug: Page slug (e.g., 'entities/python', 'concepts/ai')
        old_text: Exact text to find (must match exactly, including whitespace)
        new_text: Replacement text
    """
    wiki_path = get_wiki_path()
    page_path = wiki_path / f"{slug}.md"
    if not page_path.exists():
        return f"Page not found: {slug}"

    content = page_path.read_text(encoding="utf-8")
    if old_text not in content:
        return f"Text not found in {slug}: {old_text!r}"

    count = content.count(old_text)
    new_content = content.replace(old_text, new_text, 1)
    page_path.write_text(new_content, encoding="utf-8")
    logger.info("Edited %s: replaced 1 occurrence of %r", slug, old_text)
    return f"Replaced 1 occurrence in {slug}.md ({count} remaining)"
