"""Unit tests for frontend/chat_driver: extract_answer_so_far + stream_query.

The scripted-turn tests drive a real compiled agent built with the
non-streaming ``ScriptedChatModel`` harness (full tool-call args in a single
AIMessage.tool_calls); the multi-chunk test exercises the real incremental
``tool_call_chunks`` path with a duck-typed fake agent. Both paths must behave
identically (delta AnswerTokens).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCall, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

import agentic_rag.tools.grounding as grounding
from agentic_rag.schemas.query import QueryAnswer
from agentic_rag.tools.grounding import submit_query_answer
from agentic_rag.tools.nav import wiki_read_page, wiki_search
from agentic_rag.tools.shared import init_shared_tools
from frontend.chat_driver import (
    AnswerToken,
    extract_answer_so_far,
    FinalAnswer,
    stream_query,
    ToolEnd,
    ToolStart,
)
from tests.fixtures.fake_llm import ScriptedChatModel

MLX_ANSWER = "MLX is Apple's machine learning framework for Apple Silicon."


class TestExtractAnswerSoFar:
    """The pure partial-JSON helper — the primary unit-test target."""

    def test_empty_string_returns_empty(self):
        assert extract_answer_so_far("") == ""

    def test_partial_key_returns_empty(self):
        assert extract_answer_so_far('{"answ') == ""
        assert extract_answer_so_far('{"answer"') == ""

    def test_key_without_value_returns_empty(self):
        assert extract_answer_so_far('{"answer":') == ""
        assert extract_answer_so_far('{"answer": ') == ""

    def test_open_quote_with_incomplete_text(self):
        assert extract_answer_so_far('{"answer": "MLX') == "MLX"
        assert extract_answer_so_far('{"answer": "MLX is') == "MLX is"

    def test_complete_text(self):
        raw = json.dumps(
            {
                "answer": MLX_ANSWER,
                "citations": [],
                "confidence": "high",
                "suggestion": "",
            }
        )
        assert extract_answer_so_far(raw) == MLX_ANSWER

    def test_escaped_quotes_honoured(self):
        raw = json.dumps({"answer": 'He said "hi" to me.', "confidence": "low"})
        assert extract_answer_so_far(raw) == 'He said "hi" to me.'

    def test_text_followed_by_more_fields(self):
        assert (
            extract_answer_so_far(
                '{"answer":"MLX","citations":[{"slug":"entities/mlx"}],'
                '"confidence":"high","suggestion":""}'
            )
            == "MLX"
        )

    def test_whitespace_around_key_and_colon(self):
        assert extract_answer_so_far('{"answer"  :  "MLX"}') == "MLX"

    def test_escaped_backslash(self):
        assert extract_answer_so_far('{"answer": "C:\\\\mlx\\\\run"}') == "C:\\mlx\\run"

    def test_never_raises_on_garbage(self):
        for garbage in ["not json at all", "{{{{", '{"answer": "unterminated', "\x00\x01"]:
            extract_answer_so_far(garbage)  # must not raise


@pytest.fixture
def wiki_with_mlx(tmp_path: Path) -> Path:
    """A minimal wiki whose search hits slug ``entities/mlx``."""
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "comparisons").mkdir()
    (wiki / "entities" / "mlx.md").write_text(
        "---\n"
        "slug: entities/mlx\n"
        "type: entity\n"
        "title: MLX\n"
        "sources:\n"
        "  - sample.md\n"
        "updated: 2025-01-01\n"
        "---\n"
        "\n"
        "# MLX\n"
        "\n"
        "MLX is a machine learning framework by Apple for Apple Silicon.\n"
    )
    return wiki


def _build_query_agent(responses: list[AIMessage]) -> object:
    model = ScriptedChatModel(responses=responses)
    return create_agent(
        model=model,
        tools=[wiki_search, wiki_read_page, submit_query_answer],
        system_prompt="You are a query agent. Answer from the wiki.",
        checkpointer=MemorySaver(),  # mirrors build_agent; needed for get_state
    )


async def _collect(agent, message: str) -> list:
    return [
        event
        async for event in stream_query(agent, message, str(uuid4()), 30)
    ]


class TestStreamQueryScriptedTurn:
    """Acceptance: scripted two-tool turn (wiki_search -> submit_query_answer)."""

    async def test_event_order_and_structured_final(self, wiki_with_mlx):
        init_shared_tools(str(wiki_with_mlx))
        agent = _build_query_agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_search", args={"query": "What is MLX?"}, id="tc-1")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="submit_query_answer",
                            args={
                                "answer": MLX_ANSWER,
                                "citations": [
                                    {"slug": "entities/mlx", "title": "MLX"},
                                    {"slug": "entities/fabricated", "title": "Fabricated"},
                                ],
                                "confidence": "high",
                                "suggestion": "",
                            },
                            id="tc-2",
                        )
                    ],
                ),
            ]
        )

        events = await _collect(agent, "What is MLX?")

        kinds = [e.kind for e in events]
        assert kinds == [
            "tool_start",
            "tool_end",
            "tool_start",
            "answer_token",
            "tool_end",
            "final",
        ]

        # ToolStart for each tool, in order, with best-effort args.
        starts = [e for e in events if e.kind == "tool_start"]
        assert [e.name for e in starts] == ["wiki_search", "submit_query_answer"]
        assert starts[0].args == {"query": "What is MLX?"}
        assert starts[1].args["answer"] == MLX_ANSWER

        # ToolEnd for wiki_search (and for the submit tool).
        ends = [e for e in events if e.kind == "tool_end"]
        assert [e.name for e in ends] == ["wiki_search", "submit_query_answer"]
        assert "entities/mlx" in ends[0].output

        # One or more AnswerToken whose concatenation equals the scripted answer.
        tokens = [e for e in events if e.kind == "answer_token"]
        assert tokens, "expected at least one AnswerToken"
        assert "".join(t.text for t in tokens) == MLX_ANSWER

        # FinalAnswer: cite-or-die drops the non-navigated citation.
        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.answer == MLX_ANSWER
        assert [c.slug for c in final.answer.citations] == ["entities/mlx"]
        assert final.answer.confidence == "high"
        assert final.answer.suggestion == ""

    async def test_nav_capture_resets_per_turn(self, wiki_with_mlx):
        """stream_query installs a fresh cite-or-die capture every turn."""
        init_shared_tools(str(wiki_with_mlx))
        agent = _build_query_agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_search", args={"query": "MLX"}, id="tc-1")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="submit_query_answer",
                            args={
                                "answer": "MLX is Apple's ML framework.",
                                "citations": [{"slug": "entities/mlx", "title": "MLX"}],
                                "confidence": "high",
                                "suggestion": "",
                            },
                            id="tc-2",
                        )
                    ],
                ),
            ]
        )

        events = await _collect(agent, "What is MLX?")
        # During turn 1 the capture was populated by wiki_search.
        assert grounding._NAV_CAPTURE is not None
        assert "entities/mlx" in grounding._NAV_CAPTURE.navigated
        assert [c.slug for c in events[-1].answer.citations] == ["entities/mlx"]

        # Simulate residue from an earlier turn: the active capture already saw
        # entities/mlx before this turn starts.
        poisoned = grounding.new_nav_capture()
        poisoned.navigated.add("entities/mlx")

        # Turn 2 (fresh agent, fresh ScriptedChatModel): submit only, no
        # navigation. stream_query must replace the poisoned capture at start.
        agent2 = _build_query_agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="submit_query_answer",
                            args={
                                "answer": "MLX is Apple's ML framework.",
                                "citations": [{"slug": "entities/mlx", "title": "MLX"}],
                                "confidence": "high",
                                "suggestion": "",
                            },
                            id="tc-3",
                        )
                    ],
                ),
            ]
        )
        events2 = await _collect(agent2, "Again, what is MLX?")
        # The slug was navigated in a PREVIOUS turn only -> dropped.
        assert [c.slug for c in events2[-1].answer.citations] == []
        assert grounding._NAV_CAPTURE is not poisoned


class _FakeAgent:
    """Duck-typed compiled agent: scripted message chunks + scripted state."""

    def __init__(self, chunks: list, final_messages: list):
        self._chunks = chunks
        self._final_messages = final_messages
        self._nav_capture = None  # stream_query assigns a fresh capture

    async def astream(self, inputs, config, stream_mode="messages"):
        assert stream_mode == "messages"
        for chunk in self._chunks:
            yield chunk, {"langgraph_node": "tools" if isinstance(chunk, ToolMessage) else "model"}

    def get_state(self, config):
        return SimpleNamespace(values={"messages": self._final_messages})


def _submit_tool_call_chunks(*args_fragments: str) -> list[dict]:
    return [
        {"name": "submit_query_answer", "args": frag, "id": "tc-s", "index": 0}
        for frag in args_fragments
    ]


class TestStreamQueryMultiChunk:
    """Real incremental path: multiple tool_call_chunks with partial args."""

    async def test_partial_args_fragments_produce_delta_tokens(self):
        full_json = json.dumps(
            {
                "answer": "MLX is a great framework.",
                "citations": [],
                "confidence": "high",
                "suggestion": "",
            }
        )
        tool_msg = ToolMessage(content=full_json, name="submit_query_answer", tool_call_id="tc-s")
        fake = _FakeAgent(
            chunks=[
                AIMessageChunk(
                    content="",
                    tool_call_chunks=_submit_tool_call_chunks('{"answer": "MLX is'),
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=_submit_tool_call_chunks(
                        ' a great framework.","citations":[],"confidence":"high","suggestion":""}'
                    ),
                ),
                tool_msg,
                AIMessage(content="[done]"),
            ],
            final_messages=[tool_msg],
        )

        events = [event async for event in stream_query(fake, "What is MLX?", "t-mc", 30)]

        starts = [e for e in events if e.kind == "tool_start"]
        assert len(starts) == 1
        assert starts[0].name == "submit_query_answer"
        assert starts[0].args == {}  # first fragment not yet parseable

        tokens = [e for e in events if e.kind == "answer_token"]
        assert [t.text for t in tokens] == ["MLX is", " a great framework."]
        assert "".join(t.text for t in tokens) == "MLX is a great framework."

        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.answer == "MLX is a great framework."
        assert final.answer.citations == []

    async def test_tool_end_output_truncated_to_500(self):
        long_output = "x" * 1200
        tool_msg = ToolMessage(content=long_output, name="wiki_search", tool_call_id="t-1")
        fake = _FakeAgent(
            chunks=[tool_msg, AIMessage(content="[done]")],
            final_messages=[tool_msg],
        )

        events = [event async for event in stream_query(fake, "q", "t-trunc", 30)]

        ends = [e for e in events if e.kind == "tool_end"]
        assert len(ends) == 1
        assert ends[0].name == "wiki_search"
        assert len(ends[0].output) == 500
        assert ends[0].output == long_output[:500]


class TestStreamQueryFallbacks:
    async def test_plain_content_ignored_no_answer_token(self):
        """Free content (no tool calls) must not emit AnswerToken."""
        fake = _FakeAgent(
            chunks=[AIMessage(content="thinking out loud…")],
            final_messages=[AIMessage(content="thinking out loud…")],
        )

        events = [event async for event in stream_query(fake, "q", "t-plain", 30)]

        assert not any(e.kind == "answer_token" for e in events)
        final = events[-1]
        assert isinstance(final, FinalAnswer)

    async def test_no_submit_tool_message_yields_fallback_answer(self):
        fake = _FakeAgent(chunks=[], final_messages=[AIMessage(content="nothing")])

        events = [event async for event in stream_query(fake, "q", "t-none", 30)]

        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.answer == ""
        assert final.answer.citations == []
        assert final.answer.confidence == "low"
        assert final.answer.suggestion == "(no structured answer produced)"

    async def test_unparseable_submit_output_yields_fallback_answer(self):
        tool_msg = ToolMessage(
            content="not valid json", name="submit_query_answer", tool_call_id="tc-bad"
        )
        fake = _FakeAgent(
            chunks=[tool_msg],
            final_messages=[tool_msg],
        )

        events = [event async for event in stream_query(fake, "q", "t-bad", 30)]

        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.citations == []
        assert final.answer.confidence == "low"
        assert final.answer.suggestion == "(no structured answer produced)"

    async def test_thread_id_and_recursion_limit_reach_config(self):
        """stream_query must pass config in the exact spec shape."""
        seen: dict = {}

        class _ProbeAgent:
            _nav_capture = None

            def __init__(self, seen: dict):
                self.seen = seen

            async def astream(self, inputs, config, stream_mode="messages"):
                self.seen["inputs"] = inputs
                self.seen["config"] = config
                yield AIMessage(content="probe"), {"langgraph_node": "model"}

            def get_state(self, config):
                return SimpleNamespace(values={"messages": []})

        events = [
            event
            async for event in stream_query(_ProbeAgent(seen), "hello", "tid-42", 7)
        ]

        assert seen["inputs"] == {"messages": [{"role": "user", "content": "hello"}]}
        assert seen["config"] == {
            "configurable": {"thread_id": "tid-42"},
            "recursion_limit": 7,
        }
        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.suggestion == "(no structured answer produced)"

    async def test_fresh_nav_capture_installed_on_fake_agent(self):
        fake = _FakeAgent(chunks=[], final_messages=[])
        assert fake._nav_capture is None

        events = [event async for event in stream_query(fake, "q", "t-nav", 30)]
        events[-1]  # noqa: B018 — stream fully consumed

        assert fake._nav_capture is not None
        assert grounding._NAV_CAPTURE is fake._nav_capture


def test_query_answer_model_roundtrip():
    """The FinalAnswer.answer is a real QueryAnswer instance."""
    qa = QueryAnswer(
        answer=MLX_ANSWER,
        citations=[{"slug": "entities/mlx", "title": "MLX"}],
        confidence="high",
        suggestion="",
    )
    assert QueryAnswer.model_validate_json(qa.model_dump_json()) == qa
