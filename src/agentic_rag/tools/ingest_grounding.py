"""Structured extraction finalization tool for the ingest agent.

Mirrors ``tools/grounding.py::submit_query_answer`` (the project standard for a
finalization-as-``@tool``): ``submit_extraction`` forces a structured, testable
extraction boundary before the ingest agent writes any pages. It is PURE — no
filesystem writes, no side effects, never raises. No ``NavCapture``-style store
is needed: extraction has no citation-binding.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from agentic_rag.schemas.extraction import (
    Concept,
    Contradiction,
    Entity,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


@tool
def submit_extraction(
    entities: list[Entity],
    concepts: list[Concept],
    contradictions: list[Contradiction],
) -> str:
    """Submit the structured extraction from the source you just read.

    Call this BEFORE any create_page/update_page. Returns the validated
    ExtractionResult as JSON. List every Entity and Concept found in the source,
    and every Contradiction where the source conflicts with an existing wiki page
    (existing_claim vs new_claim + a proposed_resolution). Empty lists are fine."""
    result = ExtractionResult(
        entities=entities, concepts=concepts, contradictions=contradictions
    )
    return result.model_dump_json()
