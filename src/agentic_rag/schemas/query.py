"""Pydantic models for structured query output (QueryAnswer + SourceCitation)."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SourceCitation(BaseModel):
    """A single cited wiki page within a QueryAnswer."""

    slug: str
    title: str
    section: str | None = None


class QueryAnswer(BaseModel):
    """Structured answer from the query agent.

    Every citation slug must be in the turn's navigated set (cite-or-die #15);
    ``validate_citations`` enforces that at the boundary.
    """

    answer: str
    citations: list[SourceCitation]
    confidence: Literal["high", "medium", "low"]
    suggestion: str
