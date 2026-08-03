"""Cite-or-die finalization tool + per-invocation navigation capture.

The nav tools (``wiki_search``, ``wiki_read_page``) record every slug they
return into the module-level ``NavCapture`` via ``record_navigated``;
``submit_query_answer`` validates its citations against that capture, dropping
any citation whose slug was not navigated this turn (cite-or-die #15).
"""

from __future__ import annotations

import logging
from typing import Iterable, Literal

from langchain_core.tools import tool

from agentic_rag.schemas.query import QueryAnswer, SourceCitation

logger = logging.getLogger(__name__)

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


@tool
def submit_query_answer(
    answer: str,
    citations: list[SourceCitation],
    confidence: Literal["high", "medium", "low"],
    suggestion: str = "",
) -> str:
    """Submit the final grounded answer. Every citation slug must be a page you obtained from wiki_search/wiki_read_page this turn; citations for pages you did not navigate are dropped. If the wiki does not cover the question, set confidence to 'low' and provide a suggestion."""
    qa = QueryAnswer(
        answer=answer, citations=citations, confidence=confidence, suggestion=suggestion
    )
    navigated = _NAV_CAPTURE.navigated if _NAV_CAPTURE is not None else set()
    cleaned = validate_citations(qa, navigated)
    return cleaned.model_dump_json()
