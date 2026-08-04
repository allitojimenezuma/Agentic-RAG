"""Cite-or-die finalization + per-invocation navigation capture.

The nav tools (``wiki_search``, ``wiki_read_page``) record every slug they
return into the module-level ``NavCapture`` via ``record_navigated``. The
query agent has NO finalization tool: ``build_final_answer`` synthesizes the
``QueryAnswer`` from the model's own final message, extracting ``[[Page]]``
links as citations and validating them against the same navigated-set gate
(cite-or-die #15).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Literal

from agentic_rag.schemas.query import QueryAnswer, SourceCitation

logger = logging.getLogger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

# Set once per agent invocation via new_nav_capture().
_NAV_CAPTURE: NavCapture | None = None


class NavCapture:
    """Per-invocation mutable store of slugs navigated this turn."""

    navigated: set[str]

    def __init__(self) -> None:
        self.navigated = set()


def new_nav_capture() -> NavCapture:
    """Create a fresh NavCapture and register it as the module-global active capture."""
    global _NAV_CAPTURE
    capture = NavCapture()
    _NAV_CAPTURE = capture
    logger.debug("Created new NavCapture")
    return capture


def record_navigated(slugs: Iterable[str]) -> None:
    """Record navigated slugs into the active capture. NO-OP if none is active."""
    if _NAV_CAPTURE is None:
        return
    _NAV_CAPTURE.navigated.update(slugs)


def validate_citations(answer: QueryAnswer, navigated_slugs: set[str]) -> QueryAnswer:
    """Return a NEW QueryAnswer whose citations are filtered to navigated slugs.

    Pure function: never mutates the input and never raises. Dropped slugs are
    logged at WARNING.
    """
    kept: list[SourceCitation] = []
    for citation in answer.citations:
        if citation.slug in navigated_slugs:
            kept.append(citation)
        else:
            logger.warning("Dropping citation for non-navigated slug: %s", citation.slug)
    return answer.model_copy(update={"citations": kept})


def _resolve_navigated_target(target: str, navigated_slugs: set[str]) -> str | None:
    """Resolve a ``[[...]]`` target to a navigated slug (exact or basename)."""
    if target in navigated_slugs:
        return target
    t_lower = target.lower()
    for slug in navigated_slugs:
        if slug.rsplit("/", 1)[-1].lower() == t_lower:
            return slug
    return None


def render_answer_text(text: str) -> str:
    """Render a model answer for display: ``[[target|alias]]`` -> ``alias`` and
    ``[[target]]`` -> ``target`` (basename only). Non-navigated ``[[links]]`` are
    already dropped from citations by cite-or-die; this strips the bracket markup
    so the user sees clean text (e.g. ``[[glm-5.2]]`` -> ``glm-5.2``)."""
    return _WIKILINK_RE.sub(
        lambda m: (m.group(2) or m.group(1)).strip().rsplit("/", 1)[-1],
        text,
    )


def citations_from_links(
    text: str,
    navigated_slugs: set[str],
    titles: dict[str, str] | None = None,
) -> list[SourceCitation]:
    """Extract ``[[Page]]`` links from ``text`` as cite-or-die-validated citations.

    Handles ``[[slug]]`` and ``[[slug|alias]]``; the target is resolved against
    ``navigated_slugs`` (exact slug or case-insensitive basename match) and any
    link that was not navigated this turn is dropped (cite-or-die). Citation
    titles come from ``titles`` (slug -> frontmatter title) when available,
    else the link's alias/target text. Keeps first-seen order. Never raises.
    """
    kept: list[SourceCitation] = []
    seen: set[str] = set()
    for target, alias in _WIKILINK_RE.findall(text):
        slug = _resolve_navigated_target(target.strip(), navigated_slugs)
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        title = (titles or {}).get(slug) or (alias or target).strip() or slug
        kept.append(SourceCitation(slug=slug, title=title))
    return kept


def _navigated_titles(navigated_slugs: set[str]) -> dict[str, str]:
    """Best-effort slug -> frontmatter title map for navigated slugs. Never raises."""
    try:
        from agentic_rag.tools.shared import get_wiki_path
        from agentic_rag.wiki.model import load_wiki

        wiki = load_wiki(get_wiki_path())
    except Exception:
        logger.debug("Could not load wiki for citation titles", exc_info=True)
        return {}
    return {
        p.slug: p.fm.title
        for p in wiki.pages
        if p.slug in navigated_slugs and p.fm.title
    }


def build_final_answer(
    messages: Iterable[Any],
    navigated_slugs: set[str],
    free_text: str = "",
) -> QueryAnswer:
    """Synthesize the final QueryAnswer from the model's own output.

    There is no finalization tool: the answer is built from the last AI message
    content (falling back to ``free_text``), with ``[[Page]]`` links extracted
    as citations and validated against ``navigated_slugs`` (cite-or-die).
    Confidence is inferred: 'high' when navigated pages are cited, 'medium'
    when pages were navigated but none are cited, 'low' when nothing was
    navigated this turn.

    Never raises; always returns a valid QueryAnswer.
    """
    last_ai_content = ""
    for msg in messages:
        if getattr(msg, "type", "") == "ai":
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content:
                last_ai_content = content

    text = last_ai_content or free_text or ""
    citations = citations_from_links(
        text, navigated_slugs, _navigated_titles(navigated_slugs)
    )
    if navigated_slugs:
        confidence: Literal["high", "medium", "low"] = "high" if citations else "medium"
    else:
        confidence = "low"
    return QueryAnswer(
        answer=text,
        citations=citations,
        confidence=confidence,
        suggestion="",
    )
