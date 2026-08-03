"""Structured lint report models — the deterministic health-check output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Issue(BaseModel):
    """A single structural lint finding on one wiki page."""

    slug: str
    kind: str  # orphan | missing-index | broken-link | missing-frontmatter | missing-related | empty | stale
    severity: Literal["critical", "high", "medium", "low"]
    detail: str
    action: str


class LintReport(BaseModel):
    """Deterministic structural audit of a wiki (0 LLM calls)."""

    pages_audited: int
    issues: list[Issue]
    counts: dict[str, int]
