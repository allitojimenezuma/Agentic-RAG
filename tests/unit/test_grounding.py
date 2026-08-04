"""Unit tests for grounding: NavCapture, record_navigated, validate_citations, build_final_answer."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

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


def _ai(content: str) -> AIMessage:
    return AIMessage(content=content)


class TestRenderAnswerText:
    """Display cleanup: wikilink markup is stripped from the rendered answer."""

    def test_plain_text_unchanged(self):
        assert grounding.render_answer_text("GLM is cheap.") == "GLM is cheap."

    def test_slug_link_becomes_basename(self):
        assert grounding.render_answer_text("[[glm-5.2]] is ~10x pricier.") == (
            "glm-5.2 is ~10x pricier."
        )

    def test_aliased_link_uses_alias(self):
        assert grounding.render_answer_text("See [[entities/mlx|MLX]] for details.") == (
            "See MLX for details."
        )

    def test_full_path_link_uses_basename(self):
        assert grounding.render_answer_text("[[entities/deepseek-v4-pro]]") == (
            "deepseek-v4-pro"
        )

    def test_display_name_link_stays(self):
        assert grounding.render_answer_text("[[Spec-Driven Subagent Harness]]") == (
            "Spec-Driven Subagent Harness"
        )

    def test_multiple_links_and_empty(self):
        assert (
            grounding.render_answer_text(
                "A [[one]] and [[two|Two]] done."
            )
            == "A one and Two done."
        )
        assert grounding.render_answer_text("") == ""


class TestBuildFinalAnswer:
    """Auto-built final answer: there is no finalization tool."""

    @pytest.fixture(autouse=True)
    def _stub_titles(self, monkeypatch):
        """Keep title lookups offline: unit tests stub the wiki title map."""
        monkeypatch.setattr(grounding, "_navigated_titles", lambda slugs: {})

    def test_auto_build_from_last_ai_message_with_links(self):
        navigated = {"entities/mlx"}
        messages = [
            _ai("MLX is Apple's ML framework ([[entities/mlx]], [[entities/fabricated]]).")
        ]

        qa = grounding.build_final_answer(messages, navigated)

        assert qa.answer == "MLX is Apple's ML framework ([[entities/mlx]], [[entities/fabricated]])."
        assert [c.slug for c in qa.citations] == ["entities/mlx"]  # fabricated link dropped
        assert qa.confidence == "high"  # navigated pages cited
        assert qa.suggestion == ""

    def test_auto_finalize_display_name_and_alias_links(self):
        navigated = {"entities/mlx", "concepts/array-fire"}
        messages = [
            _ai("See [[MLX]] and [[concepts/array-fire|ArrayFire]] for details.")
        ]

        qa = grounding.build_final_answer(messages, navigated)

        assert [c.slug for c in qa.citations] == ["entities/mlx", "concepts/array-fire"]
        # Alias becomes the title when no frontmatter title is available.
        assert qa.citations[1].title == "ArrayFire"

    def test_auto_finalize_uses_titles_from_wiki_when_available(self):
        navigated = {"entities/mlx"}
        messages = [_ai("MLX ([[entities/mlx]]) is Apple's framework.")]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(grounding, "_navigated_titles", lambda slugs: {"entities/mlx": "MLX"})
            qa = grounding.build_final_answer(messages, navigated)

        assert qa.citations[0].title == "MLX"

    def test_auto_finalize_navigated_but_no_links_is_medium(self):
        navigated = {"entities/mlx"}
        messages = [_ai("MLX is Apple's machine learning framework.")]

        qa = grounding.build_final_answer(messages, navigated)

        assert qa.citations == []
        assert qa.confidence == "medium"  # consulted the wiki but cited nothing

    def test_auto_finalize_nothing_navigated_is_low(self):
        messages = [_ai("I don't know.")]

        qa = grounding.build_final_answer(messages, set())

        assert qa.citations == []
        assert qa.confidence == "low"
        assert qa.answer == "I don't know."

    def test_auto_finalize_falls_back_to_free_text(self):
        messages = []  # no AI message in state

        qa = grounding.build_final_answer(messages, set(), free_text="streamed text")

        assert qa.answer == "streamed text"
        assert qa.confidence == "low"

    def test_never_raises_on_empty_input(self):
        qa = grounding.build_final_answer([], set())

        assert qa.answer == ""
        assert qa.citations == []
        assert qa.confidence == "low"
