"""Level 3 — answer-relevancy tier: deterministic proxy + calibrated judge (T7).

Three layers:

1. Deterministic relevancy proxy (ALWAYS runs, 0 LLM): asserts the text
   properties of scripted (directly-relevant) answers that relevancy
   depends on — non-empty, sane length bounds, and key-term containment
   (a query's central term must appear in the answer: "What is MLX?" ->
   the answer contains "MLX"). An off-topic answer must also be non-empty
   and within bounds. The proxy pins the deterministic contract; it cannot
   measure relevance itself.
2. Stub-model round-trip through ``eval_judge._judge`` (ALWAYS runs,
   0 LLM): a scripted model returning ``{"score": 0.8, ...}`` parses to a
   ``RelevancyScore(score=0.8)``; an out-of-bounds score (1.7) raises
   ``RuntimeError`` whose message includes the raw output — never a silent
   passing score.
3. Real-judge tier (``@requires_llm``): ``judge_relevancy`` on the real
   model asserts anchor separation — a direct answer must score >=
   GROUNDED_MIN (0.7), an off-topic dodge <= FABRICATED_MAX (0.4). The
   anchors are pinned by T1 and must never be loosened; a separation miss
   is a measured outcome of the model under test and is reported as-is.

No ``input()``, no network beyond the optional real-judge tier.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tests.fixtures.eval_judge import (
    FABRICATED_MAX,
    GROUNDED_MIN,
    RELEVANCY_SYSTEM,
    RelevancyScore,
    _judge,
    judge_relevancy,
)
from tests.fixtures.fake_llm import ScriptedChatModel
from tests.levels.conftest import requires_llm

# Deterministic proxy bounds (sane answer-length window).
MIN_LEN = 5
MAX_LEN = 2000

# A directly-relevant answer must contain the query's central key term.
KEY_TERM_CASES: list[tuple[str, str, str]] = [
    (
        "What is MLX?",
        "MLX is a machine learning framework developed by Apple.",
        "MLX",
    ),
    (
        "What is Azure?",
        "Azure is Microsoft's cloud computing platform.",
        "Azure",
    ),
]

OFF_TOPIC_DODGE = "I don't have information about that topic."


def _assert_relevancy_proxy(query: str, answer: str, key_term: str) -> None:
    """Deterministic relevancy proxy: non-empty, in-bounds, key term present."""
    assert answer, f"answer must be non-empty (query {query!r})"
    assert MIN_LEN <= len(answer) <= MAX_LEN, (
        f"answer length {len(answer)} outside [{MIN_LEN}, {MAX_LEN}] "
        f"(query {query!r})"
    )
    assert key_term in answer, (
        f"answer must contain key term {key_term!r} for query {query!r}; "
        f"got {answer!r}"
    )


# --- deterministic proxy (ALWAYS runs, 0 LLM) ----------------------------------


@pytest.mark.parametrize(
    "query,answer,key_term",
    KEY_TERM_CASES,
    ids=["mlx", "azure"],
)
def test_scripted_grounded_answer_passes_relevancy_proxy(
    query: str, answer: str, key_term: str
) -> None:
    """A directly-relevant answer is non-empty, in-bounds, and contains the key term."""
    _assert_relevancy_proxy(query, answer, key_term)


def test_off_topic_answer_is_nonempty_and_within_bounds() -> None:
    """Off-topic answers are still well-formed text (non-empty, sane length)."""
    assert OFF_TOPIC_DODGE
    assert MIN_LEN <= len(OFF_TOPIC_DODGE) <= MAX_LEN


# --- stub-model round-trip through _judge (ALWAYS runs, 0 LLM) -----------------


def test_relevancy_stub_round_trip_parses_score() -> None:
    """A scripted valid-JSON response parses to a RelevancyScore."""
    stub = ScriptedChatModel(
        responses=[
            AIMessage(content='{"score": 0.8, "rationale": "direct"}')
        ]
    )
    score = _judge(
        stub,
        RELEVANCY_SYSTEM,
        "Question: What is MLX?\n\nAnswer: MLX is Apple's ML framework.",
        RelevancyScore,
    )
    assert isinstance(score, RelevancyScore)
    assert score.score == 0.8
    assert score.rationale == "direct"


def test_relevancy_out_of_bounds_score_raises_runtime_error_never_silent() -> None:
    """A score outside [0, 1] fails validation -> RuntimeError with raw output."""
    raw = '{"score": 1.7, "rationale": "way out of range"}'
    stub = ScriptedChatModel(responses=[AIMessage(content=raw), AIMessage(content=raw)])
    with pytest.raises(RuntimeError) as exc_info:
        _judge(
            stub,
            RELEVANCY_SYSTEM,
            "Question: What is MLX?\n\nAnswer: MLX.",
            RelevancyScore,
        )
    message = str(exc_info.value)
    assert "Raw output" in message
    assert "1.7" in message


# --- real-judge tier: anchor separation (requires_llm) --------------------------


@requires_llm
def test_real_judge_direct_answer_meets_grounded_min() -> None:
    """A directly-on-topic answer must score >= GROUNDED_MIN (0.7)."""
    score = judge_relevancy("What is MLX?", "MLX is Apple's machine learning framework.")
    assert score.score >= GROUNDED_MIN, (
        f"direct answer scored {score.score:.2f} < pinned GROUNDED_MIN={GROUNDED_MIN} "
        f"— anchor separation missed by the model under test (measured outcome, "
        f"pins not loosened)"
    )


@requires_llm
def test_real_judge_off_topic_dodge_stays_below_fabricated_max() -> None:
    """A refusal-dodge must score <= FABRICATED_MAX (0.4)."""
    score = judge_relevancy("What is MLX?", OFF_TOPIC_DODGE)
    assert score.score <= FABRICATED_MAX, (
        f"off-topic dodge scored {score.score:.2f} > pinned FABRICATED_MAX={FABRICATED_MAX} "
        f"— anchor separation missed by the model under test (measured outcome, "
        f"pins not loosened)"
    )
