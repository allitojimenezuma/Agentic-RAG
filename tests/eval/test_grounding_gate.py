"""Eval gate: ``validate_citations`` drops hallucinated (non-navigated) citations.

Mirrors the QueryAnswer/SourceCitation fixture construction in
``tests/unit/test_grounding.py``. Pure, deterministic, no LLM.
"""

from __future__ import annotations

from agentic_rag.schemas.query import QueryAnswer, SourceCitation
from agentic_rag.tools.grounding import validate_citations


def _qa(citations: list[SourceCitation], confidence: str = "high", suggestion: str = "") -> QueryAnswer:
    """Build a QueryAnswer the same way tests/unit/test_grounding.py does."""
    return QueryAnswer(
        answer="MLX is Apple's machine learning framework.",
        citations=citations,
        confidence=confidence,
        suggestion=suggestion,
    )


def test_fabricated_citation_is_dropped_genuine_kept() -> None:
    """Only citations whose slug was navigated survive the gate."""
    navigated = {"entities/mlx"}
    genuine = SourceCitation(slug="entities/mlx", title="MLX", section="Overview")
    fabricated = SourceCitation(slug="entities/does-not-exist", title="Fabricated")
    answer = _qa([genuine, fabricated])

    cleaned = validate_citations(answer, navigated)

    assert cleaned is not answer
    assert [c.slug for c in cleaned.citations] == ["entities/mlx"]
    assert cleaned.citations[0] == genuine


def test_validate_citations_never_mutates_input() -> None:
    """The input QueryAnswer must be untouched (pure function contract)."""
    navigated = {"entities/mlx"}
    genuine = SourceCitation(slug="entities/mlx", title="MLX")
    fabricated = SourceCitation(slug="entities/does-not-exist", title="Fabricated")
    answer = _qa([genuine, fabricated])
    original_citations = list(answer.citations)

    validate_citations(answer, navigated)

    assert answer.citations == original_citations
    assert [c.slug for c in answer.citations] == [
        "entities/mlx",
        "entities/does-not-exist",
    ]


def test_validate_citations_never_raises() -> None:
    """Gate must not raise even when all citations are fabricated."""
    navigated = {"entities/mlx"}
    fabricated = SourceCitation(slug="entities/does-not-exist", title="Fabricated")

    cleaned = validate_citations(_qa([fabricated]), navigated)

    assert cleaned.citations == []
    assert cleaned.answer == "MLX is Apple's machine learning framework."
    assert cleaned.confidence == "high"
    assert cleaned.suggestion == ""
