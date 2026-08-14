"""Level 3 — deterministic faithfulness proxies + DeepEval real-judge anchors (T7).

WHAT IT TESTS:
- Deterministic tier (0 LLM, zero network, fully headless — runs ALWAYS).
  The query agent's answer is synthesized by ``build_final_answer`` from the
  model's own final message: ``[[Page]]`` links become citations validated
  against the per-turn navigated set (cite-or-die). These tests pin the
  deterministic core of that contract without any judge model:
  1. cite-or-die: a fabricated ``[[link]]`` is dropped; only navigated pages
     are citable;
  2. link-citation consistency: every navigated ``[[link]]`` in the answer
     text becomes a citation (first-seen order); no uncited navigated links;
  3. confidence inference: navigated+cited -> "high", navigated but nothing
     cited -> "medium", nothing navigated -> "low";
  4. end-to-end NavCapture: a scripted query agent (wiki_search ->
     wiki_read_page -> final answer) records the navigated slug, and every
     citation built from the final answer resolves to a genuinely navigated
     page.
- Real-judge tier (@requires_llm): DeepEval ``FaithfulnessMetric`` with the
  project's judge (tests/fixtures/deepeval_judge.py) asserts anchor
  separation — a grounded answer must score >= GROUNDED_MIN (0.7), a
  fabricated answer <= FABRICATED_MAX (0.4). Pins never loosened; a miss is
  a measured model outcome reported as-is.

HOW IT RUNS:
- Deterministic tier: 0-LLM (always, offline). Real tier: skipped without
  OPENAI_API_KEY. The NavCapture is a module-global in
  ``agentic_rag.tools.grounding``; the autouse fixture below resets it around
  every test so no capture leaks between tests.

WHY IT MATTERS:
- These proxies are the offline half of faithfulness: they guarantee the
  answer-citation machinery itself is sound, so the real-model judge can only
  measure model behavior, not engine bugs.

RUN: uv run pytest tests/levels/level3/test_faithfulness.py
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_query_prompt
from agentic_rag.tools import grounding
from agentic_rag.tools.grounding import (
    build_final_answer,
    citations_from_links,
    new_nav_capture,
)
from agentic_rag.tools.nav import wiki_read_page, wiki_search, wiki_summary
from agentic_rag.tools.shared import init_shared_tools
from tests.fixtures.deepeval_judge import deepeval_judge
from tests.fixtures.fake_llm import ScriptedChatModel
from tests.levels.conftest import requires_llm


@pytest.fixture(autouse=True)
def _reset_nav_capture():
    """Reset the module-global capture around each test (mirrors the unit
    suite's ``tests/unit/test_grounding.py`` pattern)."""
    grounding._NAV_CAPTURE = None
    yield
    grounding._NAV_CAPTURE = None


# --- cite-or-die: fabricated links never become citations ----------------------


def test_cite_or_die_drops_fabricated_link() -> None:
    """A fabricated [[entities/fake]] link is dropped; only navigated pages are cited."""
    navigated = {"entities/mlx"}
    messages = [
        AIMessage(
            content="MLX is Apple's framework "
            "([[entities/mlx]], [[entities/fake]])."
        )
    ]

    qa = build_final_answer(messages, navigated)

    assert [c.slug for c in qa.citations] == ["entities/mlx"]
    assert all(c.slug != "entities/fake" for c in qa.citations)
    # Every citation must resolve to a navigated page (cite-or-die gate).
    assert all(c.slug in navigated for c in qa.citations)


def test_cite_or_die_empty_navigated_drops_everything() -> None:
    """Nothing navigated this turn -> no citations, regardless of links."""
    qa = build_final_answer(
        [AIMessage(content="See [[entities/mlx]] for details.")], set()
    )
    assert qa.citations == []
    assert qa.confidence == "low"


# --- link <-> citation consistency ---------------------------------------------


def test_every_navigated_link_becomes_a_citation_first_seen_order() -> None:
    """citations_from_links extracts ALL navigated links, deduped, in first-seen order."""
    text = (
        "See [[entities/mlx]] for the framework, "
        "[[concepts/3d-gaussian-splatting|3DGS]] for rendering, "
        "[[entities/mlx]] again, and [[entities/fabricated]]."
    )
    navigated = {"entities/mlx", "concepts/3d-gaussian-splatting"}

    citations = citations_from_links(
        text, navigated, titles={"entities/mlx": "MLX"}
    )

    assert [(c.slug, c.title) for c in citations] == [
        ("entities/mlx", "MLX"),
        ("concepts/3d-gaussian-splatting", "3DGS"),
    ]
    # No uncited navigated links: every navigated link present in the text
    # resolved to a citation; the fabricated one is dropped.
    assert {c.slug for c in citations} == navigated


def test_basename_link_resolves_to_navigated_slug() -> None:
    """A [[MLX]] display link resolves to the navigated entities/mlx slug."""
    citations = citations_from_links(
        "[[MLX]] is Apple's framework.", {"entities/mlx"}, titles={}
    )
    assert [c.slug for c in citations] == ["entities/mlx"]


# --- confidence inference --------------------------------------------------------


def test_confidence_high_when_navigated_and_cited() -> None:
    navigated = {"entities/mlx"}
    qa = build_final_answer(
        [AIMessage(content="MLX ([[entities/mlx]]) is Apple's framework.")],
        navigated,
    )
    assert qa.citations and qa.confidence == "high"


def test_confidence_medium_when_navigated_but_nothing_cited() -> None:
    navigated = {"entities/mlx"}
    qa = build_final_answer(
        [AIMessage(content="MLX is Apple's machine learning framework.")],
        navigated,
    )
    assert qa.citations == []
    assert qa.confidence == "medium"


def test_confidence_low_when_nothing_navigated() -> None:
    qa = build_final_answer([AIMessage(content="I don't know.")], set())
    assert qa.citations == []
    assert qa.confidence == "low"


# --- end-to-end NavCapture through a scripted query agent -----------------------


def test_end_to_end_nav_capture_citations_are_navigated(eval_wiki) -> None:
    """Scripted agent navigates via tools; every citation is a navigated slug.

    Attaches ``_nav_capture = new_nav_capture()`` BEFORE invoke so the nav
    tools record into it via ``record_navigated``; the final answer's citation
    must both be recorded as navigated and resolve to the navigated page.
    """
    init_shared_tools(str(eval_wiki))
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="wiki_search", args={"query": "mlx"}, id="tc-1")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="wiki_read_page", args={"slug": "entities/mlx"}, id="tc-2")
                ],
            ),
            AIMessage(content="MLX is Apple's ML framework ([[entities/mlx]])."),
        ]
    )
    agent = build_agent(
        model=model,
        tools=[wiki_search, wiki_read_page, wiki_summary],
        system_prompt=build_query_prompt("# Test schema"),
    )
    agent._nav_capture = new_nav_capture()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is MLX?"}]},
        config={"configurable": {"thread_id": str(uuid4())}},
    )

    navigated = agent._nav_capture.navigated
    # The capture actually recorded the navigated page (tool runs -> record_navigated).
    assert "entities/mlx" in navigated

    qa = build_final_answer(result["messages"], navigated)
    assert qa.citations, "expected at least one citation from the final answer"
    # Cite-or-die end-to-end: every citation slug was genuinely navigated this turn.
    assert all(c.slug in navigated for c in qa.citations)
    assert [c.slug for c in qa.citations] == ["entities/mlx"]


# --- calibrated judge: real-model anchor separation via DeepEval (T7) ---------
# The hand-rolled judge (tests/fixtures/eval_judge.py) was retired 2026-08-06
# after a cross-calibration run scored IDENTICALLY on the anchors (grounded
# 1.00/1.00, fabricated 0.00/0.00, direct relevancy 1.00/1.00). Anchors are
# pinned here and must never be loosened.

MLX_PAGE_CONTEXT_PATH = (
    Path(__file__).parents[2] / "fixtures" / "eval_wiki" / "entities" / "mlx.md"
)

GROUNDED_MIN: float = 0.7  # T1 pin: a directly-grounded answer must score >= 0.7
FABRICATED_MAX: float = 0.4  # T1 pin: a fabricated answer must score <= 0.4


def _mlx_context() -> str:
    """REAL page text from the committed eval corpus — never invented."""
    return MLX_PAGE_CONTEXT_PATH.read_text(encoding="utf-8")


QUESTION = "What is MLX?"
GROUNDED_ANSWER = (
    "MLX is a machine learning framework developed by Apple for Apple "
    "Silicon, using the Metal API."
)
FABRICATED_ANSWER = "MLX was developed by Google for TPU clusters."


@requires_llm
def test_real_judge_grounded_answer_meets_anchor_min() -> None:
    """A claim directly supported by the corpus context must score >= 0.7."""
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    judge = deepeval_judge()
    assert judge is not None, "requires_llm should have skipped without a key"
    metric = FaithfulnessMetric(model=judge, threshold=0.0, async_mode=False)
    # Average 3 measurements: single-shot LLM-as-judge calls are noisy, the
    # anchor is about the model's calibration band, not one sample.
    scores = []
    for _ in range(3):
        metric.measure(
            LLMTestCase(
                input=QUESTION,
                actual_output=GROUNDED_ANSWER,
                retrieval_context=[_mlx_context()],
            )
        )
        scores.append(metric.score)
    avg = sum(scores) / len(scores)
    assert avg >= GROUNDED_MIN, (
        f"grounded answer averaged {avg:.2f} (samples {[round(s, 2) for s in scores]}) "
        f"< pinned GROUNDED_MIN={GROUNDED_MIN} — anchor separation missed by the "
        f"model under test (measured outcome, pins not loosened)"
    )


@requires_llm
def test_real_judge_fabricated_answer_stays_below_anchor_max() -> None:
    """A claim contradicted by the corpus context must score <= 0.4."""
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    judge = deepeval_judge()
    assert judge is not None, "requires_llm should have skipped without a key"
    metric = FaithfulnessMetric(model=judge, threshold=0.0, async_mode=False)
    scores = []
    for _ in range(3):
        metric.measure(
            LLMTestCase(
                input=QUESTION,
                actual_output=FABRICATED_ANSWER,
                retrieval_context=[_mlx_context()],
            )
        )
        scores.append(metric.score)
    avg = sum(scores) / len(scores)
    assert avg <= FABRICATED_MAX, (
        f"fabricated answer averaged {avg:.2f} (samples {[round(s, 2) for s in scores]}) "
        f"> pinned FABRICATED_MAX={FABRICATED_MAX} — anchor separation missed by the "
        f"model under test (measured outcome, pins not loosened)"
    )
