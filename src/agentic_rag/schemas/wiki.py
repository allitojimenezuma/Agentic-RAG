"""Pydantic models for wiki pages, frontmatter, index, and log entries."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class Frontmatter(BaseModel):
    """YAML frontmatter for wiki pages."""

    slug: str
    type: str  # entity, concept, source, comparison, overview
    title: str
    sources: list[str] = Field(default_factory=list)
    updated: date
    tags: list[str] = Field(default_factory=list)


class IndexEntry(BaseModel):
    """An entry in the wiki index."""

    slug: str
    summary: str
    type: str  # entity, concept, source, comparison, overview
    sources: list[str] = Field(default_factory=list)
    updated: date
    display_name: str | None = None  # preserves original casing from parsed entry


class Index(BaseModel):
    """Parsed wiki index with categorized entries."""

    categories: dict[str, list[IndexEntry]] = Field(default_factory=dict)


class LogEntry(BaseModel):
    """An entry in the wiki log."""

    timestamp: datetime
    op: str  # ingest, query, lint, create, update, delete
    title: str
    details: str


class Heading(BaseModel):
    """A markdown heading."""

    level: int
    text: str


class Link(BaseModel):
    """An Obsidian-style [[link]]."""

    target: str
    alias: Optional[str] = None
