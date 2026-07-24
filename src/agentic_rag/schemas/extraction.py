"""Pydantic structured-output models for extraction (used by ingest agent)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """An extracted entity."""

    name: str
    type: str  # person, organization, software, hardware, etc.
    summary: str
    sources: list[str] = Field(default_factory=list)


class Concept(BaseModel):
    """An extracted concept."""

    name: str
    summary: str
    related_entities: list[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    """A detected contradiction between existing and new claims."""

    page_slug: str
    existing_claim: str
    new_claim: str
    proposed_resolution: str


class ExtractionResult(BaseModel):
    """Result of extracting entities, concepts, and contradictions from a source."""

    entities: list[Entity] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
