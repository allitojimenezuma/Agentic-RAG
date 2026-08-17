"""Navigation for agents: one pinned, read-only command dispatcher.

The agents explore the wiki through a SINGLE tool, ``wiki_command``, instead
of a zoo of similar navigation tools (old surface: ``wiki_search``,
``wiki_read_page``, ``wiki_summary``, ``wiki_scan``, ``wiki_link_graph``,
``match_page_tool``, ``run_health_check``). The command string follows a
pinned grammar; every sub-command dispatches to the deterministic engine
functions (``load_wiki``, ``search``, ``match_page``, ``health_check``). There
is no shell, no ``subprocess``, no redirection — the surface is read-only BY
CONSTRUCTION, and mutations happen only through the typed write tools
(``create_page``/``update_page``/... in other modules).

Compound commands (``&&`` or newlines) let the agent run several reads in one
tool call, saving turns and context.

Grammar (also printed by the ``help`` sub-command):

    scan [--max-chars N]
    search "<query>" [--k N] [--type TYPE] [--tags a,b]
    read <slug> [--section "Heading"]
    links [--slug S]
    match "<name>" --type TYPE
    health
    help
"""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.io.markdown_parser import extract_links, parse_frontmatter, slugify
from agentic_rag.io.wiki_io import list_pages, read_page as _read_page
from agentic_rag.tools.shared import get_wiki_path
from agentic_rag.wiki.dedupe_index import regenerate_index as _regenerate_index
from agentic_rag.wiki.health import health_check
from agentic_rag.wiki.match import match_page
from agentic_rag.wiki.model import DIR_TO_TYPE, Page, Wiki, load_wiki
from agentic_rag.wiki.search import search

logger = logging.getLogger(__name__)

# Hard cap on the dispatcher output so a broad command never floods the
# agent's context window.
_MAX_OUTPUT_CHARS = 12_000

_GRAMMAR = """Read-only wiki commands (join several with && or newlines):
- scan [--max-chars N]        overview of every content page
- search "<query>" [--k N] [--type TYPE] [--tags a,b]   BM25-ranked pages
- read <slug> [--section S]   full page markdown, or one section
- links [--slug S]            inbound/outbound link summary
- match "<name>" --type TYPE  create vs update vs conflict decision
- health                      deterministic structural audit (0 LLM calls)
- help                        this reference"""


@tool
def wiki_command(command: str) -> str:
    """Run read-only wiki commands. One string, multiple commands joined by '&&' or newlines. Sub-commands: scan, search "<query>", read <slug>, links, match "<name>" --type <type>, health, help. Call 'wiki_command("help")' for the full grammar. Read-only by construction — the wiki can only be changed through the write tools."""
    return run_wiki_commands(command)


def run_wiki_commands(command: str) -> str:
    """Parse and execute a pinned wiki-command string. Never raises.

    Each sub-command runs independently; a failing sub-command yields an
    ``Error: ...`` line inline and execution continues with the next one.
    """
    outputs: list[str] = []
    for segment in _split_commands(command):
        try:
            args = shlex.split(segment)
        except ValueError as exc:  # unbalanced quotes
            outputs.append(f"Error: could not parse {segment!r}: {exc}")
            continue
        if not args:
            continue
        outputs.append(_dispatch(args))

    text = "\n\n".join(outputs) if outputs else "No commands given."
    if len(text) > _MAX_OUTPUT_CHARS:
        text = text[:_MAX_OUTPUT_CHARS] + "\n… (output truncated)"
    return text


def _split_commands(text: str) -> list[str]:
    """Split a command string on ``&&`` and newlines; drop empties."""
    return [part.strip() for part in re.split(r"\n|&&", text) if part.strip()]


def _dispatch(argv: list[str]) -> str:
    """Route one parsed command line to its handler. Never raises."""
    name = argv[0]
    try:
        handler = {
            "scan": _cmd_scan,
            "search": _cmd_search,
            "read": _cmd_read,
            "links": _cmd_links,
            "match": _cmd_match,
            "health": _cmd_health,
            "help": _cmd_help,
        }[name]
        return handler(argv[1:])
    except KeyError:
        return f"Error: unknown command {name!r}. Run wiki_command('help') for the grammar."
    except Exception as exc:  # defensive — one command must never kill the batch
        logger.warning("wiki_command %r failed: %s", argv, exc, exc_info=True)
        return f"Error: command {name!r} failed: {exc}"


def _parse_kv(args: list[str], aliases: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Split ``--key value`` style flags out of a positional arg list.

    Returns (positionals, {canonical_key: value}). Unknown flags are errors.
    """
    positionals: list[str] = []
    flags: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--") and arg[2:] in aliases:
            key = aliases[arg[2:]]
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raise ValueError(f"flag {arg} requires a non-flag value")
            flags[key] = args[i + 1]
            i += 2
        elif arg.startswith("--"):
            raise ValueError(f"unknown flag {arg}")
        else:
            positionals.append(arg)
            i += 1
    return positionals, flags


# --- Sub-command handlers -----------------------------------------------------

def _cmd_scan(args: list[str]) -> str:
    """Per-page overview: slug, type, title, preview, link counts, date."""
    positionals, flags = _parse_kv(args, {"max-chars": "max_chars"})
    if positionals:
        raise ValueError(f"unexpected argument {positionals[0]!r}")
    max_chars = int(flags.get("max_chars", "200"))

    wiki = load_wiki(get_wiki_path())
    content_pages = [p for p in wiki.pages if not p.rel_path.name.startswith("lint-report-")]
    if not content_pages:
        return "No wiki pages found."

    content_slugs = {p.slug for p in content_pages}
    inbound: dict[str, set[str]] = {}
    for page in content_pages:
        for target in page.outbound_links:
            if target in content_slugs:
                inbound.setdefault(target, set()).add(page.slug)

    lines: list[str] = []
    for page in sorted(content_pages, key=lambda p: p.slug):
        preview = _preview_text(page, max_chars)
        updated = page.fm.updated.isoformat() if page.fm.updated else "-"
        lines.append(
            f'- {page.slug} ({page.fm.type}) - {page.fm.title} — "{preview}" — '
            f"out: {len(page.outbound_links)} | in: {len(inbound.get(page.slug, ()))} | updated: {updated}"
        )
    return "\n".join(lines)


def _cmd_search(args: list[str]) -> str:
    """BM25 search with bounded link expansion; records navigated slugs."""
    aliases = {"k": "k", "type": "types", "tags": "tags"}
    positionals, flags = _parse_kv(args, aliases)
    if len(positionals) != 1:
        raise ValueError("usage: search \"<query>\" [--k N] [--type TYPE] [--tags a,b]")
    query = positionals[0]

    from agentic_rag.tools.grounding import record_navigated

    try:
        k = int(flags.get("k", "8"))
    except ValueError:
        raise ValueError("usage: --k must be an integer") from None
    type_list = _split_csv(flags.get("types"))
    tag_list = _split_csv(flags.get("tags"))

    wiki = load_wiki(get_wiki_path())
    hits = search(wiki, query, k=k, types=type_list, tags=tag_list)
    if not hits:
        return f"No relevant pages found for '{query}'."

    record_navigated(h.slug for h in hits)
    direct = [h for h in hits if h.matched_via != "expand-link"]
    linked = [h for h in hits if h.matched_via == "expand-link"]
    lines: list[str] = []
    for i, h in enumerate(direct):
        prefix = f"Found {len(hits)} relevant: " if i == 0 else "- "
        lines.append(
            f"{prefix}{h.slug} (score={h.score:.2f}, sections: {'; '.join(h.sections)})"
        )
    lines.extend(f"+ linked: {h.slug}" for h in linked)
    return "\n".join(lines)


def _cmd_read(args: list[str]) -> str:
    """Read one page (full markdown or one section); records the navigated slug."""
    from agentic_rag.tools.grounding import record_navigated

    positionals, flags = _parse_kv(args, {"section": "section"})
    if not positionals:
        raise ValueError("usage: read <slug> [--section \"Heading\"]")
    slug = positionals[0]
    section = flags.get("section")

    wiki_path = get_wiki_path()
    try:
        if section is None:
            content = _read_page(wiki_path, slug)
            record_navigated([_resolved_slug(wiki_path, slug)])
            return content

        wiki = load_wiki(wiki_path)
        page = _find_page(wiki, slug)
        if page is None:
            return _page_not_found_error(slug)
        record_navigated([page.slug])
        target = section.lower()
        for s in page.sections:
            if s.heading.lower() == target:
                return s.text
        headings = "; ".join(s.heading for s in page.sections if s.heading) or "none"
        return f"Section '{section}' not found on page '{slug}'. Available headings: {headings}"
    except FileNotFoundError:
        return _page_not_found_error(slug)


def _cmd_links(args: list[str]) -> str:
    """Inbound/outbound link summary for the whole wiki or one page."""
    positionals, flags = _parse_kv(args, {"slug": "slug"})
    if positionals:
        raise ValueError(f"unexpected argument {positionals[0]!r}")
    only_slug = flags.get("slug")

    lines = _link_graph_lines()
    if only_slug is None:
        return "\n".join(lines)

    # One-page view: the block between that page's "### slug" marker and the
    # next "###" marker (or the summary separator).
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"### {only_slug} ")),
        None,
    )
    if start is None:
        return f"Error: no page with slug '{only_slug}'."
    block: list[str] = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("### ") or line.startswith("---"):
            break
        block.append(line)
    return "\n".join(block).strip() or f"Error: no page with slug '{only_slug}'."


def _cmd_match(args: list[str]) -> str:
    """Deterministic create/update/conflict decision for a page name."""
    positionals, flags = _parse_kv(args, {"type": "type"})
    if not positionals or not flags.get("type"):
        raise ValueError('usage: match "<name>" --type entity|concept|source|comparison')
    wiki = load_wiki(get_wiki_path())
    result = match_page(wiki, positionals[0], flags["type"])
    return f"{result.decision}: {', '.join(result.slugs)} — {result.detail}"


def _cmd_health(args: list[str]) -> str:
    """Deterministic structural audit — zero LLM calls."""
    positionals, flags = _parse_kv(args, {})
    if positionals:
        raise ValueError(f"unexpected argument {positionals[0]!r}")
    report = health_check(get_wiki_path())
    lines = [f"Pages audited: {report.pages_audited} | Issues: {len(report.issues)}"]
    lines.extend(
        f"[{issue.severity}] {issue.kind}: {issue.slug} — {issue.detail}"
        for issue in report.issues
    )
    if not report.issues:
        lines.append("No issues — the wiki is structurally clean.")
    return "\n".join(lines)


def _cmd_help(args: list[str]) -> str:
    return _GRAMMAR


# --- Helpers ------------------------------------------------------------------

def _preview_text(page: Page, max_chars: int) -> str:
    """First-section preview: whitespace collapsed, truncated with '…'."""
    if not page.sections or not page.sections[0].text:
        return "(no content)"
    text = " ".join(page.sections[0].text.split())
    if not text:
        return "(no content)"
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def _split_csv(value: str | None) -> list[str] | None:
    """Comma-separated tool arg -> stripped list (None if empty)."""
    if not value:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts or None


def _resolved_slug(wiki_path: Path, slug: str) -> str:
    """Resolve a slug to its canonical page slug."""
    try:
        from agentic_rag.io.wiki_io import _resolve_page_path

        resolved = _resolve_page_path(wiki_path, slug)
        return str(resolved.relative_to(wiki_path)).removesuffix(".md")
    except FileNotFoundError:
        return slug


def _find_page(wiki: Wiki, slug: str) -> Page | None:
    """Resolve a slug against the in-memory model (exact, then basename)."""
    return wiki.by_slug.get(_resolved_slug(get_wiki_path(), slug))


def _page_not_found_error(slug: str) -> str:
    """Helpful not-found message: suggest an existing slug with the same basename."""
    wiki = load_wiki(get_wiki_path())
    short = slug.rsplit("/", 1)[-1]
    matches = sorted(p.slug for p in wiki.pages if p.slug.rsplit("/", 1)[-1] == short)
    if matches:
        return (
            f"Error: Wiki page not found: {slug}. "
            f"Did you mean: {', '.join(matches)}?"
        )
    return (
        f"Error: Wiki page not found: {slug}. "
        "Check the slug — use wiki_command('scan') to list pages."
    )


def _link_graph_lines() -> list[str]:
    """Compute the full inbound/outbound link summary (shared with 'links')."""
    wiki_path = get_wiki_path()
    pages = list_pages(wiki_path)
    if not pages:
        return ["No wiki pages found."]

    page_slugs = {str(p.relative_to(wiki_path)).removesuffix(".md") for p in pages}
    page_data: dict[str, dict] = {}

    def _resolve_link(target: str) -> str | None:
        if target in page_slugs:
            return target
        s = slugify(target)
        for ps in page_slugs:
            short = ps.rsplit("/", 1)[-1]
            if short == s or ps.endswith("/" + s):
                return ps
        t = target.lower().replace(" ", "-")
        for ps in page_slugs:
            short = ps.rsplit("/", 1)[-1]
            if short == t or ps.endswith("/" + t):
                return ps
        return None

    for page_path in pages:
        slug = str(page_path.relative_to(wiki_path)).removesuffix(".md")
        content = page_path.read_text(encoding="utf-8")
        outbound = {
            resolved
            for link in extract_links(content)
            if (resolved := _resolve_link(link.target)) and resolved != slug
        }
        page_type = "unknown"
        title = slug.rsplit("/", 1)[-1]
        if content.startswith("---"):
            try:
                fm = parse_frontmatter(content)
                page_type = fm.type or page_type
                title = fm.title or title
            except Exception:
                pass
        else:
            if "/" in slug:
                page_type = DIR_TO_TYPE.get(slug.split("/", 1)[0], page_type)
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        page_data[slug] = {"outbound": outbound, "type": page_type, "title": title}

    inbound = {slug: set() for slug in page_slugs}
    for slug, data in page_data.items():
        for target in data["outbound"]:
            if target in inbound:
                inbound[target].add(slug)

    lines: list[str] = []
    for slug in sorted(page_slugs):
        data = page_data[slug]
        in_links = sorted(inbound.get(slug, set()))
        out_links = sorted(data["outbound"])
        lines.append(f"### {slug} ({data['type']})")
        lines.append(
            f"  Outbound ({len(out_links)}): {', '.join(out_links) if out_links else 'none'}"
        )
        lines.append(
            f"  Inbound  ({len(in_links)}): {', '.join(in_links) if in_links else 'none — ORPHAN?'}"
        )
        lines.append("")

    orphans = [s for s in page_slugs if not inbound.get(s)]
    lines.append("--- SUMMARY ---")
    lines.append(f"Total pages: {len(page_slugs)}")
    lines.append(f"Total links: {sum(len(d['outbound']) for d in page_data.values())}")
    lines.append(f"Orphans (no inbound links): {len(orphans)}")
    if orphans:
        lines.append(f"  {', '.join(sorted(orphans))}")
    return lines


@tool
def regenerate_index() -> str:
    """Regenerate the wiki index.md from the pages on disk. Call this after creating or updating pages (replaces update_index)."""
    logger.debug("Regenerating wiki index")
    _regenerate_index(get_wiki_path())
    return "Index regenerated."