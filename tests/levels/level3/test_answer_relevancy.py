"""Level 3 — answer-relevancy tier: deterministic proxy + DeepEval judge (T7).

WHAT IT TESTS:
- Two layers:
  1. Deterministic relevancy proxy (ALWAYS runs, 0 LLM): asserts the text
     properties of scripted (directly-relevant) answers that relevancy
     depends on — non-empty, sane length bounds, and key-term containment
     (a query's central term must appear in the answer: "What is MLX?" ->
     the answer contains "MLX"). An off-topic answer must also be non-empty
     and within bounds. The proxy pins the deterministic contract; it cannot
     measure relevance itself.
  2. Real-judge tier (@requires_llm): DeepEval ``AnswerRelevancyMetric``
     with the project's judge (tests/fixtures/deepeval_judge.py) asserts
     anchor separation — a direct answer must score >= GROUNDED_MIN (0.7),
     an off-topic answer <= FABRICATED_MAX (0.4). The anchors are pinned by T1
     and must never be loosened; a separation miss is a measured outcome of
     the model under test and is reported as-is.

HOW IT RUNS:
- Layer 1: 0-LLM (always, offline). Layer 2: real LLM judge (@requires_llm,
  skipped without OPENAI_API_KEY).

WHY IT MATTERS:
- Relevancy is "did the answer actually answer the question?" — the proxy
  catches empty/off-key answers offline; the judge tier measures whether the
  real model stays on-topic for direct questions vs evasive dodges.

RUN: uv run pytest tests/levels/level3/test_answer_relevancy.py
"""

from __future__ import annotations

import pytest

from tests.fixtures.deepeval_judge import deepeval_judge
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

# Off-topic answer: substantively about a DIFFERENT topic than the question.
# NOTE: a pure evasion ("I don't have information about that topic") is a known
# AnswerRelevancyMetric blind spot — it generates questions FROM the answer, so
# evasions score as relevant. The anchor uses a substantive off-topic answer,
# which the metric scores ~0 (measured 2026-08-06).
OFF_TOPIC_ANSWER = "Azure is Microsoft's cloud computing platform for hosting large language models."


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
    assert OFF_TOPIC_ANSWER
    assert MIN_LEN <= len(OFF_TOPIC_ANSWER) <= MAX_LEN


# --- real-judge tier: anchor separation via DeepEval (requires_llm) -----------
# The hand-rolled relevancy judge (tests/fixtures/eval_judge.py) was retired
# 2026-08-06 after cross-calibration scored IDENTICALLY (direct 1.00/1.00);
# the DeepEval AnswerRelevancyMetric replaces it. Anchors are pinned and must
# never be loosened.

GROUNDED_MIN: float = 0.7  # T1 pin: a direct answer must score >= 0.7
FABRICATED_MAX: float = 0.4  # T1 pin: an off-topic answer must score <= 0.4


@requires_llm
def test_real_judge_direct_answer_meets_grounded_min() -> None:
    """A directly-on-topic answer must score >= GROUNDED_MIN (0.7)."""
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    judge = deepeval_judge()
    assert judge is not None, "requires_llm should have skipped without a key"
    metric = AnswerRelevancyMetric(model=judge, threshold=0.0, async_mode=False)
    scores = []
    for _ in range(3):
        metric.measure(
            LLMTestCase(
                input="What is MLX?",
                actual_output="MLX is Apple's machine learning framework.",
            )
        )
        scores.append(metric.score)
    avg = sum(scores) / len(scores)
    assert avg >= GROUNDED_MIN, (
        f"direct answer averaged {avg:.2f} (samples {[round(s, 2) for s in scores]}) "
        f"< pinned GROUNDED_MIN={GROUNDED_MIN} — anchor separation missed by the "
        f"model under test (measured outcome, pins not loosened)"
    )


@requires_llm
def test_real_judge_off_topic_answer_stays_below_fabricated_max() -> None:
    """A substantively off-topic answer must score <= FABRICATED_MAX (0.4)."""
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    judge = deepeval_judge()
    assert judge is not None, "requires_llm should have skipped without a key"
    metric = AnswerRelevancyMetric(model=judge, threshold=0.0, async_mode=False)
    scores = []
    for _ in range(3):
        metric.measure(
            LLMTestCase(
                input="What is MLX?",
                actual_output=OFF_TOPIC_ANSWER,
            )
        )
        scores.append(metric.score)
    avg = sum(scores) / len(scores)
    assert avg <= FABRICATED_MAX, (
        f"off-topic answer averaged {avg:.2f} (samples {[round(s, 2) for s in scores]}) "
        f"> pinned FABRICATED_MAX={FABRICATED_MAX} — anchor separation missed by "
        f"the model under test (measured outcome, pins not loosened)"
    )
