"""Unit tests for ingest grounding: the submit_extraction finalization tool."""

from __future__ import annotations

from agentic_rag.schemas import Concept, Contradiction, Entity, ExtractionResult
from agentic_rag.tools import ingest_grounding


def _entity(name: str = "MLX") -> Entity:
    return Entity(
        name=name,
        type="software",
        summary=f"{name} is a machine learning framework.",
        sources=["apple-mlx.md"],
    )


def _concept() -> Concept:
    return Concept(
        name="array programming",
        summary="Computations applied to whole arrays at once.",
        related_entities=["MLX"],
    )


def _contradiction() -> Contradiction:
    return Contradiction(
        page_slug="entities/mlx",
        existing_claim="MLX is CPU-only.",
        new_claim="MLX also runs on GPU.",
        proposed_resolution="Update page to mention GPU support.",
    )


class TestSubmitExtraction:
    def test_returns_parseable_extraction_result_json(self):
        entity = _entity()
        concept = _concept()
        contradiction = _contradiction()

        result = ingest_grounding.submit_extraction.invoke(
            {
                "entities": [entity],
                "concepts": [concept],
                "contradictions": [contradiction],
            }
        )

        parsed = ExtractionResult.model_validate_json(result)
        assert parsed.entities == [entity]
        assert parsed.concepts == [concept]
        assert parsed.contradictions == [contradiction]

    def test_empty_lists_round_trip(self):
        result = ingest_grounding.submit_extraction.invoke(
            {"entities": [], "concepts": [], "contradictions": []}
        )

        parsed = ExtractionResult.model_validate_json(result)
        assert parsed.entities == []
        assert parsed.concepts == []
        assert parsed.contradictions == []

    def test_never_raises_and_does_not_mutate_inputs(self):
        entity = _entity()
        original_entity = entity.model_copy(deep=True)

        result = ingest_grounding.submit_extraction.invoke(
            {"entities": [entity], "concepts": [], "contradictions": []}
        )

        assert ExtractionResult.model_validate_json(result).entities == [entity]
        assert entity == original_entity

    def test_no_filesystem_writes(self, monkeypatch):
        """Pure tool: any open() call during invocation is a violation."""

        def _fail_open(*_args, **_kwargs):
            raise AssertionError("submit_extraction must not touch the filesystem")

        monkeypatch.setattr("builtins.open", _fail_open)

        result = ingest_grounding.submit_extraction.invoke(
            {"entities": [_entity()], "concepts": [_concept()], "contradictions": []}
        )

        parsed = ExtractionResult.model_validate_json(result)
        assert parsed.entities[0].name == "MLX"
