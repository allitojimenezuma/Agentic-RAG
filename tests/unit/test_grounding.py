"""Unit tests for grounding: NavCapture, record_navigated, validate_citations, submit_query_answer."""

from __future__ import annotations

import pytest

from agentic_rag.schemas.query import QueryAnswer, SourceCitation
from agentic_rag.tools import grounding


@pytest.fixture(autouse=True)
def _reset_nav_capture():
    """Reset the module-global capture around each test."""
    grounding._NAV_CAPTURE = None
    yield
    grounding._NAV_CAPTURE = None


def _qa(citations: list[SourceCitation], confidence: str = "high", suggestion: str = "") -> QueryAnswer:
    return QueryAnswer(
        answer="MLX is Apple's machine learning framework.",
        citations=citations,
        confidence=confidence,
        suggestion=suggestion,
    )


class TestValidateCitations:
    def test_keeps_navigated_citations_preserving_order_and_fields(self):
        navigated = {"entities/mlx", "concepts/array-fire"}
        c1 = SourceCitation(slug="entities/mlx", title="MLX", section="Overview")
        c2 = SourceCitation(slug="concepts/array-fire", title="ArrayFire")
        answer = _qa([c1, c2])

        cleaned = grounding.validate_citations(answer, navigated)

        assert cleaned is not answer
        assert cleaned.answer == answer.answer
        assert cleaned.confidence == answer.confidence
        assert cleaned.suggestion == answer.suggestion
        assert cleaned.citations == [c1, c2]

    def test_drops_fabricated_citation_not_navigated(self):
        navigated = {"entities/mlx"}
        valid = SourceCitation(slug="entities/mlx", title="MLX")
        fabricated = SourceCitation(slug="entities/does-not-exist", title="Fabricated")
        answer = _qa([valid, fabricated])

        cleaned = grounding.validate_citations(answer, navigated)

        assert [c.slug for c in cleaned.citations] == ["entities/mlx"]

    def test_never_mutates_input(self):
        navigated = {"entities/mlx"}
        valid = SourceCitation(slug="entities/mlx", title="MLX")
        fabricated = SourceCitation(slug="entities/does-not-exist", title="Fabricated")
        original = _qa([valid, fabricated])
        original_citations = list(original.citations)

        grounding.validate_citations(original, navigated)

        assert original.citations == original_citations
        assert [c.slug for c in original.citations] == [
            "entities/mlx",
            "entities/does-not-exist",
        ]

    def test_empty_navigated_set_drops_all(self):
        answer = _qa([SourceCitation(slug="entities/mlx", title="MLX")])

        cleaned = grounding.validate_citations(answer, set())

        assert cleaned.citations == []

    def test_low_confidence_with_empty_suggestion_passes_through(self):
        answer = _qa(
            [SourceCitation(slug="entities/mlx", title="MLX")],
            confidence="low",
            suggestion="",
        )

        cleaned = grounding.validate_citations(answer, {"entities/mlx"})

        assert cleaned.confidence == "low"
        assert cleaned.suggestion == ""


class TestNavCapture:
    def test_record_navigated_noop_without_active_capture(self):
        grounding.record_navigated(["entities/mlx"])  # must not raise

    def test_record_navigated_adds_to_active_capture(self):
        capture = grounding.new_nav_capture()

        grounding.record_navigated(["entities/mlx", "entities/mlx"])
        grounding.record_navigated(["concepts/array-fire"])

        assert capture.navigated == {"entities/mlx", "concepts/array-fire"}
        assert grounding._NAV_CAPTURE is capture


class TestSubmitQueryAnswer:
    def test_drops_fabricated_and_returns_parseable_json(self):
        capture = grounding.new_nav_capture()
        capture.navigated.add("entities/mlx")
        valid = SourceCitation(slug="entities/mlx", title="MLX")
        fabricated = SourceCitation(slug="entities/fabricated", title="Fabricated")

        result = grounding.submit_query_answer.invoke(
            {
                "answer": "MLX is Apple's framework.",
                "citations": [valid, fabricated],
                "confidence": "high",
                "suggestion": "",
            }
        )

        parsed = QueryAnswer.model_validate_json(result)
        assert parsed.answer == "MLX is Apple's framework."
        assert [c.slug for c in parsed.citations] == ["entities/mlx"]
        assert parsed.citations[0].title == "MLX"
        assert parsed.confidence == "high"
        assert parsed.suggestion == ""

    def test_does_not_crash_when_all_citations_dropped(self):
        grounding.new_nav_capture()
        fabricated = SourceCitation(slug="entities/fabricated", title="Fabricated")

        result = grounding.submit_query_answer.invoke(
            {
                "answer": "Nothing grounded.",
                "citations": [fabricated],
                "confidence": "low",
                "suggestion": "No pages covered this.",
            }
        )

        parsed = QueryAnswer.model_validate_json(result)
        assert parsed.citations == []
        assert parsed.confidence == "low"
        assert parsed.suggestion == "No pages covered this."
