"""Cite-or-die finalization + per-invocation navigation capture.

The nav surface (``wiki_command``: search/read/scan/links) records every slug it
returns into the per-run ``NavCapture`` (bound through the LangChain run config,
never a module global) via ``record_navigated``. The
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

class NavCapture:
    """Per-invocation mutable store of slugs navigated this turn."""

    navigated: set[str]

    def __init__(self) -> None:
        self.navigated = set()


def new_nav_capture() -> NavCapture:
    """Create a fresh NavCapture for one agent invocation.

    The capture is later bound to the run via
    ``config['configurable']['nav_capture']`` (see ``cli.py`` and
    ``frontend/query_driver.py``) so ``record_navigated`` can find it **without a
    shared module global**. No module-level mutable state is touched here, which
    keeps concurrent query turns fully isolated: each run owns its own capture
    object instead of contending on one process-wide global.
    """
    return NavCapture()


def record_navigated(slugs: Iterable[str]) -> None:
    """Record navigated slugs into the capture bound to the current run.

    The active capture is read from the LangChain run config
    (``config['configurable']['nav_capture']``), set per-invocation by the CLI
    and the Streamlit driver. This replaces the old module-level global so two
    concurrent query turns no longer share (and corrupt) one navigated set —
    cite-or-die grounding stays correct under concurrency.

    NO-OP when no capture is configured: the non-query agents (ingest/lint/fix),
    scripted test flows, and standalone ``wiki_command`` calls record nothing.
    Never raises.
    """
    capture = _active_capture()
    if capture is None:
        return
    capture.navigated.update(slugs)


def _active_capture() -> NavCapture | None:
    """Best-effort lookup of the per-run NavCapture from the LangChain config."""
    try:
        from langchain_core.runnables import ensure_config

        configurable = ensure_config().get("configurable") or {}
        return configurable.get("nav_capture")
    except Exception:
        logger.debug("No running config with a nav_capture; recording skipped")
        return None


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
