"""Tools for the fix agent: safe in-wiki page edits (no shell access).

All four tools resolve slugs the same way as the rest of the codebase
(exact path, then recursive basename lookup), so ``mlx`` and
``entities/mlx`` behave identically. Edits are guarded so the fix agent can
never leave a page in a worse state than it found it: ``edit_wiki_page``
refuses no-ops and edits that would corrupt the YAML frontmatter,
``fix_link`` refuses to create a NEW dangling link, and
``append_related_section`` is idempotent (no duplicate bullets).
"""

from __future__ import annotations

import logging
import re
from datetime import date

from langchain_core.tools import tool

from agentic_rag.io.markdown_parser import parse_frontmatter, slugify
from agentic_rag.io.wiki_io import (
    _resolve_page_path,
    list_pages,
    read_page,
    write_page,
)
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools.shared import get_wiki_path

logger = logging.getLogger(__name__)

# [[target]] or [[target|alias]] — mirrors io.markdown_parser._LINK_RE.
_LINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]+?))?\]\]")


def _canonical_slug(wiki_path, slug: str) -> str:
    """Resolve ``slug`` to its canonical wiki-relative slug (''-free path)."""
    resolved = _resolve_page_path(wiki_path, slug)
    return str(resolved.relative_to(wiki_path)).removesuffix(".md")


def _resolve_target_exists(wiki_path, target: str) -> bool:
    """True if ``target`` resolves to an existing page slug (exact, slugified,
    or unicode-preserving short-name match — same rules as wiki.model)."""
    page_slugs = {str(p.relative_to(wiki_path)).removesuffix(".md") for p in list_pages(wiki_path)}
    if target in page_slugs:
        return True
    s = slugify(target)
    for ps in page_slugs:
        if ps.rsplit("/", 1)[-1] == s or ps.endswith("/" + s):
            return True
    t = target.lower().replace(" ", "-")
    for ps in page_slugs:
        if ps.rsplit("/", 1)[-1] == t or ps.endswith("/" + t):
            return True
    return False


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
        canonical = _canonical_slug(wiki_path, slug)
        body = read_page(wiki_path, canonical)
    except FileNotFoundError:
        return f"Page not found: {slug}"

    if body.startswith("---"):
        return f"Error: {slug} already has frontmatter"

    fm = Frontmatter(
        slug=canonical,
        type=page_type,
        title=title,
        sources=[],
        updated=date.today(),
        tags=[],
    )
    write_page(wiki_path, canonical, body, frontmatter=fm)
    logger.info("Added frontmatter to %s (type=%s)", canonical, page_type)
    return f"Added frontmatter to {canonical}."


@tool
def fix_link(slug: str, old_target: str, new_target: str) -> str:
    """Fix every occurrence of a broken [[old_target]] link on a page.

    Replaces both plain (``[[old_target]]``) and aliased
    (``[[old_target|alias]]``) forms, preserving the alias. Refuses to run when
    ``new_target`` does not resolve to an existing wiki page, so fixing a link
    can never create a new dangling link.

    Args:
        slug: Page slug containing the broken links
        old_target: Current link target to replace
        new_target: Replacement link target
    """
    wiki_path = get_wiki_path()
    try:
        canonical = _canonical_slug(wiki_path, slug)
        content = read_page(wiki_path, canonical)
    except FileNotFoundError:
        return f"Page not found: {slug}"

    if old_target.strip() == new_target.strip():
        return f"old_target and new_target are identical: {old_target}"

    replaced = 0

    def _repl(match: re.Match) -> str:
        nonlocal replaced
        if match.group(1) != old_target:
            return match.group(0)
        replaced += 1
        return f"[[{new_target}|{match.group(2)}]]" if match.group(2) else f"[[{new_target}]]"

    new_content = _LINK_RE.sub(_repl, content)
    count = replaced
    if count == 0:
        return f"No links to '{old_target}' found in {canonical}"

    # Only validate the target when there is actually something to fix — a page
    # with no occurrences reports zero replacements without extra checks.
    if not _resolve_target_exists(wiki_path, new_target):
        return (
            f"Error: new_target {new_target!r} does not resolve to an existing "
            f"page. Use wiki_command('scan') to list valid slugs."
        )

    write_page(wiki_path, canonical, new_content)
    logger.info("Fixed %d link(s) in %s: %s -> %s", count, canonical, old_target, new_target)
    return f"Replaced {count} link(s) in {canonical}."


@tool
def append_related_section(slug: str, links: list[str]) -> str:
    """Add a '## Related' section with cross-links, or append missing ones.

    Idempotent: links already present as bullets are skipped, and duplicates
    in ``links`` are collapsed, so repeated calls never grow duplicate bullets.

    Args:
        slug: Page slug to update
        links: Page slugs (or display names) to add as - [[link]] entries
    """
    wiki_path = get_wiki_path()
    try:
        canonical = _canonical_slug(wiki_path, slug)
        content = read_page(wiki_path, canonical)
    except FileNotFoundError:
        return f"Page not found: {slug}"

    # Existing Related bullets (targets only, alias stripped) for dedup.
    existing: set[str] = set()
    lines = content.split("\n")
    rel_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == "## Related"), None
    )
    if rel_idx is not None:
        end = len(lines)
        for i in range(rel_idx + 1, len(lines)):
            if lines[i].startswith("## "):
                end = i
                break
        for line in lines[rel_idx + 1 : end]:
            m = _LINK_RE.match(line.strip().lstrip("- "))
            if m:
                existing.add(m.group(1).strip())

    wanted = []
    for link in links:
        target = link.strip()
        if not target or target in existing or target in wanted:
            continue
        wanted.append(target)

    if not wanted:
        return f"No new links to append to {canonical} (all already related)."

    if rel_idx is None:
        lines.append("## Related")
        lines.append("")
        lines.extend(f"- [[{link}]]" for link in wanted)
    else:
        end = len(lines)
        for i in range(rel_idx + 1, len(lines)):
            if lines[i].startswith("## "):
                end = i
                break
        lines[end:end] = [f"- [[{link}]]" for link in wanted]

    write_page(wiki_path, canonical, "\n".join(lines))
    logger.info("Appended %d related link(s) to %s", len(wanted), canonical)
    return f"Appended {len(wanted)} related link(s) to {canonical}."


@tool
def edit_wiki_page(slug: str, old_text: str, new_text: str) -> str:
    """Replace the first occurrence of old_text in a wiki page.

    Guards: refuses a no-op edit (identical texts), and refuses an edit that
    would corrupt the page's YAML frontmatter block (when the page has one, the
    edited result must still parse as valid frontmatter, so legitimate
    frontmatter edits are allowed but schema-breaking ones are rejected).

    Args:
        slug: Page slug (e.g., 'entities/python', 'concepts/ai')
        old_text: Exact text to find (must match exactly, including whitespace)
        new_text: Replacement text
    """
    wiki_path = get_wiki_path()
    try:
        canonical = _canonical_slug(wiki_path, slug)
        content = read_page(wiki_path, canonical)
    except FileNotFoundError:
        return f"Page not found: {slug}"

    if old_text == new_text:
        return f"old_text and new_text are identical — nothing to change."

    if old_text not in content:
        return f"Text not found in {canonical}: {old_text!r}"

    had_frontmatter = content.startswith("---")
    total_occurrences = content.count(old_text)
    new_content = content.replace(old_text, new_text, 1)

    if had_frontmatter:
        try:
            parse_frontmatter(new_content)
        except Exception as exc:
            return (
                f"Error: this edit would corrupt the page's YAML frontmatter "
                f"({exc}). Use add_frontmatter or fix_link for schema-level changes."
            )

    write_page(wiki_path, canonical, new_content)
    logger.info("Edited %s: replaced 1 occurrence of %r", canonical, old_text)
    return f"Replaced 1 occurrence in {canonical}.md ({total_occurrences - 1} remaining)"