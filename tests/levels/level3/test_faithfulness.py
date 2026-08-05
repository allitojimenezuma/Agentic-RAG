"""Level 3 — deterministic faithfulness proxies over the grounding engine.

Zero LLM calls, zero network, fully headless. Runs ALWAYS (no skip marker).
The query agent's answer is synthesized by ``build_final_answer`` from the
model's own final message: ``[[Page]]`` links become citations validated
against the per-turn navigated set (cite-or-die). These tests pin the
deterministic core of that contract without any judge model:

1. cite-or-die: a fabricated ``[[link]]`` is dropped; only navigated pages are
   citable.
2. link-citation consistency: every navigated ``[[link]]`` in the answer text
   becomes a citation (first-seen order); there are no uncited navigated links.
3. confidence inference: navigated+cited -> "high", navigated but nothing
   cited -> "medium", nothing navigated -> "low".
4. end-to-end NavCapture: a scripted query agent (wiki_search ->
   wiki_read_page -> final answer) records the navigated slug, and every
   citation built from the final answer resolves to a genuinely navigated
   page.

The NavCapture is a module-global in ``agentic_rag.tools.grounding``; the
autouse fixture below resets it around every test so no capture leaks between
tests.
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
from tests.fixtures.eval_judge import (
    FABRICATED_MAX,
    FAITHFULNESS_SYSTEM,
    GROUNDED_MIN,
    FaithfulnessScore,
    _judge,
    judge_faithfulness,
)
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


# --- calibrated judge: stub round-trip + real-model anchors (T7) -----------------
# The judge harness lives in tests/fixtures/eval_judge.py. Deterministic tiers
# exercise its strict-JSON + corrective-retry semantics with a ScriptedChatModel
# stub (0 LLM, ALWAYS run); the @requires_llm tier calls judge_faithfulness on
# the real model and asserts the pinned anchor separation.

MLX_PAGE_CONTEXT_PATH = (
    Path(__file__).parents[2] / "fixtures" / "eval_wiki" / "entities" / "mlx.md"
)


def _mlx_context() -> str:
    """REAL page text from the committed eval corpus — never invented."""
    return MLX_PAGE_CONTEXT_PATH.read_text(encoding="utf-8")


def test_judge_stub_round_trip_parses_score() -> None:
    """A scripted valid-JSON response parses to a FaithfulnessScore."""
    stub = ScriptedChatModel(
        responses=[
            AIMessage(content='{"score": 0.9, "rationale": "directly supported"}')
        ]
    )
    score = _judge(
        stub,
        FAITHFULNESS_SYSTEM,
        "Question: What is MLX?\nAnswer: MLX is Apple's framework.\nContext: [1] MLX.",
        FaithfulnessScore,
    )
    assert isinstance(score, FaithfulnessScore)
    assert score.score == 0.9
    assert score.rationale == "directly supported"


def test_judge_corrective_retry_recovers_valid_json() -> None:
    """First response unparseable -> ONE corrective retry parses the second."""
    stub = ScriptedChatModel(
        responses=[
            AIMessage(content="not json at all"),
            AIMessage(content='{"score": 0.3, "rationale": "recovered on retry"}'),
        ]
    )
    score = _judge(
        stub,
        FAITHFULNESS_SYSTEM,
        "Question: What is MLX?\nAnswer: unsupported.\nContext: [1] MLX.",
        FaithfulnessScore,
    )
    assert score.score == 0.3
    assert score.rationale == "recovered on retry"


def test_judge_out_of_bounds_score_raises_runtime_error_never_silent() -> None:
    """A score outside [0, 1] fails validation -> RuntimeError with raw output.

    Two identical scripted responses so the corrective retry also yields the
    invalid score and the RuntimeError carries the raw output (never a
    silent passing score).
    """
    raw = '{"score": 1.7, "rationale": "invalid"}'
    stub = ScriptedChatModel(responses=[AIMessage(content=raw), AIMessage(content=raw)])
    with pytest.raises(RuntimeError) as exc_info:
        _judge(
            stub,
            FAITHFULNESS_SYSTEM,
            "Question: What is MLX?\nAnswer: MLX.\nContext: [1] MLX.",
            FaithfulnessScore,
        )
    message = str(exc_info.value)
    assert "Raw output" in message
    assert "1.7" in message


# --- real-judge tier: anchor separation (requires_llm) --------------------------
QUESTION = "What is MLX?"
GROUNDED_ANSWER = (
    "MLX is a machine learning framework developed by Apple for Apple "
    "Silicon, using the Metal API."
)
FABRICATED_ANSWER = "MLX was developed by Google for TPU clusters."


@requires_llm
def test_real_judge_grounded_answer_meets_anchor_min() -> None:
    """A claim directly supported by the corpus context must score >= 0.7."""
    score = judge_faithfulness(QUESTION, GROUNDED_ANSWER, [_mlx_context()])
    assert score.score >= GROUNDED_MIN, (
        f"grounded answer scored {score.score:.2f} < pinned "
        f"GROUNDED_MIN={GROUNDED_MIN} — anchor separation missed by the model "
        f"under test (measured outcome, pins not loosened)"
    )


@requires_llm
def test_real_judge_fabricated_answer_stays_below_anchor_max() -> None:
    """A claim contradicted by the corpus context must score <= 0.4."""
    score = judge_faithfulness(QUESTION, FABRICATED_ANSWER, [_mlx_context()])
    assert score.score <= FABRICATED_MAX, (
        f"fabricated answer scored {score.score:.2f} > pinned "
        f"FABRICATED_MAX={FABRICATED_MAX} — anchor separation missed by the "
        f"model under test (measured outcome, pins not loosened)"
    )
