"""Unit tests for frontend/query_driver: stream_query event translation.

There is NO finalization tool: the model's free-text message is the answer. The
scripted-turn tests drive a real compiled agent built with the non-streaming
``ScriptedChatModel`` harness (full tool-call args in a single
AIMessage.tool_calls); the multi-chunk tests exercise the real incremental
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
from agentic_rag.tools.nav import wiki_command
from agentic_rag.tools.shared import init_shared_tools
from frontend.query_driver import (
    AnswerToken,
    FinalAnswer,
    stream_query,
    ToolEnd,
    ToolStart,
)
from tests.fixtures.fake_llm import ScriptedChatModel

MLX_ANSWER = "MLX is Apple's machine learning framework for Apple Silicon."


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
        tools=[wiki_command],
        system_prompt="You are a query agent. Answer from the wiki.",
        checkpointer=MemorySaver(),  # mirrors build_agent; needed for get_state
    )


async def _collect(agent, message: str) -> list:
    return [
        event
        async for event in stream_query(agent, message, str(uuid4()), 30)
    ]


class TestStreamQueryScriptedTurn:
    """Acceptance: scripted turn (wiki_command -> free-text final answer)."""

    async def test_event_order_and_auto_built_final(self, wiki_with_mlx):
        init_shared_tools(str(wiki_with_mlx))
        agent = _build_query_agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'search "What is MLX?"'}, id="tc-1")
                    ],
                ),
                # Open the page so it is navigated and citable.
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'read entities/mlx'}, id="tc-2")
                    ],
                ),
                AIMessage(
                    content=(
                        f"{MLX_ANSWER} "
                        "([[entities/mlx]], [[entities/fabricated]])."
                    )
                ),
            ]
        )

        events = await _collect(agent, "What is MLX?")

        kinds = [e.kind for e in events]
        assert kinds == ["tool_start", "tool_end", "tool_start", "tool_end", "answer_token", "final"]

        # ToolStart for wiki_command: search then read.
        starts = [e for e in events if e.kind == "tool_start"]
        assert [e.name for e in starts] == ["wiki_command", "wiki_command"]
        assert starts[0].args == {'command': 'search "What is MLX?"'}
        assert starts[1].args == {'command': 'read entities/mlx'}

        # ToolEnd for wiki_command (search output then read output).
        ends = [e for e in events if e.kind == "tool_end"]
        assert [e.name for e in ends] == ["wiki_command", "wiki_command"]
        assert "entities/mlx" in ends[0].output
        assert "entities/mlx" in ends[1].output

        # The final message streams live as AnswerTokens.
        tokens = [e for e in events if e.kind == "answer_token"]
        assert tokens, "expected at least one AnswerToken"
        assert "".join(t.text for t in tokens) == (
            f"{MLX_ANSWER} ([[entities/mlx]], [[entities/fabricated]])."
        )

        # FinalAnswer: auto-built from the model output; cite-or-die drops the
        # non-navigated citation.
        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.answer == (
            f"{MLX_ANSWER} ([[entities/mlx]], [[entities/fabricated]])."
        )
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
                        ToolCall(name="wiki_command", args={"command": 'search "MLX"'}, id="tc-1")
                    ],
                ),
                # Open the page so it is navigated and citable.
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'read entities/mlx'}, id="tc-2")
                    ],
                ),
                AIMessage(
                    content="MLX is Apple's ML framework ([[entities/mlx]])."
                ),
            ]
        )

        events = await _collect(agent, "What is MLX?")
        # During turn 1 the capture was populated by the `read` of wiki_command.
        assert agent._nav_capture is not None
        assert "entities/mlx" in agent._nav_capture.navigated
        assert [c.slug for c in events[-1].answer.citations] == ["entities/mlx"]

        # Simulate residue from an earlier turn: the active capture already saw
        # entities/mlx before this turn starts.
        poisoned = grounding.new_nav_capture()
        poisoned.navigated.add("entities/mlx")

        # Turn 2 (fresh agent, fresh ScriptedChatModel): no navigation, the
        # answer still links entities/mlx. stream_query must replace the
        # poisoned capture at start -> the link is dropped.
        agent2 = _build_query_agent(
            [
                AIMessage(
                    content="MLX is Apple's ML framework ([[entities/mlx]])."
                ),
            ]
        )
        events2 = await _collect(agent2, "Again, what is MLX?")
        # The slug was navigated in a PREVIOUS turn only -> dropped.
        assert [c.slug for c in events2[-1].answer.citations] == []
        # stream_query installed a FRESH capture for turn 2 (not the poisoned one),
        # so a link only seen in a prior turn is dropped.
        assert agent2._nav_capture is not poisoned


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


class TestStreamQueryMultiChunk:
    """Real incremental path: multi-chunk tool args and free text."""

    async def test_tool_end_output_truncated_to_500(self):
        long_output = "x" * 1200
        tool_msg = ToolMessage(content=long_output, name="wiki_command", tool_call_id="t-1")
        fake = _FakeAgent(
            chunks=[tool_msg, AIMessage(content="[done]")],
            final_messages=[tool_msg],
        )

        events = [event async for event in stream_query(fake, "q", "t-trunc", 30)]

        ends = [e for e in events if e.kind == "tool_end"]
        assert len(ends) == 1
        assert ends[0].name == "wiki_command"
        assert len(ends[0].output) == 500
        assert ends[0].output == long_output[:500]

    async def test_free_text_chunks_stream_as_answer_tokens(self):
        """Multi-chunk free text accumulates into delta AnswerTokens, and the
        FinalAnswer uses the last AI message from the thread state."""
        fake = _FakeAgent(
            chunks=[
                AIMessageChunk(content="MLX is "),
                AIMessageChunk(content="a great framework."),
            ],
            final_messages=[AIMessage(content="MLX is a great framework.")],
        )

        events = [event async for event in stream_query(fake, "q", "t-ft", 30)]

        tokens = [e for e in events if e.kind == "answer_token"]
        assert [t.text for t in tokens] == ["MLX is ", "a great framework."]
        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.answer == "MLX is a great framework."
        assert final.answer.citations == []
        assert final.answer.confidence == "low"

    async def test_sequential_tool_calls_both_emit_tool_start(self):
        """Real streaming: sequential tool calls arrive in SEPARATE model
        messages and each restarts ``index`` at 0. An index-only key would
        collide and swallow the second call's ToolStart (its chip never
        renders); each call must emit its own ToolStart with full args."""
        fake = _FakeAgent(
            chunks=[
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-1", "name": "wiki_command", "args": '{"command": "search mlx"}'}
                    ],
                ),
                ToolMessage(
                    content="Found: entities/mlx", name="wiki_command", tool_call_id="call-1"
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-2", "name": "wiki_command", "args": '{"command": "read entities/mlx"}'}
                    ],
                ),
                ToolMessage(
                    content="MLX page…", name="wiki_command", tool_call_id="call-2"
                ),
                AIMessage(content="Done ([[entities/mlx]])."),
            ],
            final_messages=[AIMessage(content="Done ([[entities/mlx]]).")],
        )

        events = [event async for event in stream_query(fake, "q", "t-seq", 30)]

        starts = [e for e in events if e.kind == "tool_start"]
        assert [e.name for e in starts] == ["wiki_command", "wiki_command"]
        assert [e.args for e in starts] == [
            {'command': 'search mlx'},
            {'command': 'read entities/mlx'},
        ]
        # ToolEnd pairs with the right ToolStart via the streaming call id.
        ends = [e for e in events if e.kind == "tool_end"]
        assert [e.call_id for e in ends] == ["call-1", "call-2"]

    async def test_multi_chunk_args_tool_start_carries_full_args(self):
        """ToolStart is deferred until the streamed args parse into a complete
        JSON object, so the chip shows the FULL parameters (the first chunk
        alone is partial JSON)."""
        fake = _FakeAgent(
            chunks=[
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-1", "name": "wiki_command", "args": '{"command": "search Mál'}
                    ],
                ),
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": None, "name": None, "args": 'aga"}'}
                    ],
                ),
                ToolMessage(
                    content="Found", name="wiki_command", tool_call_id="call-1"
                ),
                AIMessage(content="Answer ([[entities/mlx]])."),
            ],
            final_messages=[AIMessage(content="Answer ([[entities/mlx]]).")],
        )

        events = [event async for event in stream_query(fake, "q", "t-multi", 30)]

        starts = [e for e in events if e.kind == "tool_start"]
        assert [e.name for e in starts] == ["wiki_command"]
        assert starts[0].args == {"command": "search Málaga"}
        assert starts[0].call_id == "call-1"
        assert [e.kind for e in events] == [
            "tool_start",
            "tool_end",
            "answer_token",
            "final",
        ]

    async def test_unparseable_args_fall_back_to_tool_message(self):
        """Args that never parse (malformed JSON) still produce a ToolStart:
        the ToolMessage branch synthesizes one so no tool call is missed."""
        fake = _FakeAgent(
            chunks=[
                AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {"index": 0, "id": "call-1", "name": "wiki_command", "args": "not json"}
                    ],
                ),
                ToolMessage(
                    content="Found", name="wiki_command", tool_call_id="call-1"
                ),
                AIMessage(content="Answer."),
            ],
            final_messages=[AIMessage(content="Answer.")],
        )

        events = [event async for event in stream_query(fake, "q", "t-bad", 30)]

        assert [e.kind for e in events] == [
            "tool_start",
            "tool_end",
            "answer_token",
            "final",
        ]
        start = next(e for e in events if e.kind == "tool_start")
        assert start.name == "wiki_command"
        assert start.args == {}

    async def test_free_text_after_tool_calls_streams_as_answer(self):
        """The model's free text AFTER navigation is the answer (no finalization
        tool): it streams live and the FinalAnswer reflects it."""
        tool_msg = ToolMessage(
            content="Found 1 relevant: entities/mlx", name="wiki_command", tool_call_id="t-1"
        )
        fake = _FakeAgent(
            chunks=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'search "mlx"'}, id="t-1")
                    ],
                ),
                tool_msg,
                AIMessage(content="MLX is Apple's framework ([[entities/mlx]])."),
            ],
            final_messages=[AIMessage(content="MLX is Apple's framework ([[entities/mlx]]).")],
        )

        events = [event async for event in stream_query(fake, "q", "t-after", 30)]

        assert [e.kind for e in events] == [
            "tool_start",
            "tool_end",
            "answer_token",
            "final",
        ]
        tokens = [e for e in events if e.kind == "answer_token"]
        assert "".join(t.text for t in tokens) == "MLX is Apple's framework ([[entities/mlx]])."
        final = events[-1]
        assert final.answer.answer == "MLX is Apple's framework ([[entities/mlx]])."


class TestStreamQueryFallbacks:
    async def test_plain_content_streams_as_answer_token(self):
        """Free content (no tool calls) streams live as the answer text; the
        final answer reuses it."""
        fake = _FakeAgent(
            chunks=[AIMessage(content="thinking out loud…")],
            final_messages=[AIMessage(content="thinking out loud…")],
        )

        events = [event async for event in stream_query(fake, "q", "t-plain", 30)]

        tokens = [e for e in events if e.kind == "answer_token"]
        assert [t.text for t in tokens] == ["thinking out loud…"]
        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.answer == "thinking out loud…"
        assert final.answer.confidence == "low"
        assert final.answer.suggestion == ""

    async def test_answer_from_thread_state_when_nothing_streamed(self):
        # chunks=[] -> nothing streamed; the answer is synthesized from the last
        # AIMessage content in the thread state.
        fake = _FakeAgent(chunks=[], final_messages=[AIMessage(content="nothing")])

        events = [event async for event in stream_query(fake, "q", "t-none", 30)]

        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.answer == "nothing"
        assert final.answer.citations == []
        assert final.answer.confidence == "low"
        assert final.answer.suggestion == ""

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

        probe = _ProbeAgent(seen)
        events = [
            event
            async for event in stream_query(probe, "hello", "tid-42", 7)
        ]

        assert seen["inputs"] == {"messages": [{"role": "user", "content": "hello"}]}
        assert seen["config"] == {
            "configurable": {"thread_id": "tid-42", "nav_capture": probe._nav_capture},
            "recursion_limit": 7,
        }
        final = events[-1]
        assert isinstance(final, FinalAnswer)
        assert final.answer.suggestion == ""

    async def test_fresh_nav_capture_installed_on_fake_agent(self):
        fake = _FakeAgent(chunks=[], final_messages=[])
        assert fake._nav_capture is None

        events = [event async for event in stream_query(fake, "q", "t-nav", 30)]
        events[-1]  # noqa: B018 — stream fully consumed

        assert fake._nav_capture is not None


class TestConcurrentNavCapture:
    """Concurrency guard for the NavCapture fix.

    Two overlapping stream_query turns must own independent cite-or-die
    captures. This is the exact regression the per-run config change prevents:
    with the old module-level global, concurrent queries shared (and corrupted)
    one navigated set."""

    @pytest.fixture
    def wiki_two_entities(self, tmp_path: Path) -> Path:
        wiki = tmp_path / "wiki"
        for d in ("entities", "concepts", "sources", "comparisons"):
            (wiki / d).mkdir(parents=True)
        (wiki / "entities" / "mlx.md").write_text(
            "---\n"
            "slug: entities/mlx\n"
            "type: entity\n"
            "title: MLX\n"
            "sources:\n"
            "  - s.md\n"
            "updated: 2025-01-01\n"
            "---\n\n# MLX\n\nMLX is Apple's ML framework.\n"
        )
        (wiki / "entities" / "other.md").write_text(
            "---\n"
            "slug: entities/other\n"
            "type: entity\n"
            "title: Other\n"
            "sources:\n"
            "  - s.md\n"
            "updated: 2025-01-01\n"
            "---\n\n# Other\n\nOther thing.\n"
        )
        return wiki

    async def test_concurrent_streams_have_isolated_captures(self, wiki_two_entities: Path):
        init_shared_tools(str(wiki_two_entities))

        agent_a = _build_query_agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'search "MLX"'}, id="a-1")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'read entities/mlx'}, id="a-2")
                    ],
                ),
                AIMessage(content="MLX answer ([[entities/mlx]])."),
            ]
        )
        agent_b = _build_query_agent(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'search "Other"'}, id="b-1")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="wiki_command", args={"command": 'read entities/other'}, id="b-2")
                    ],
                ),
                AIMessage(content="Other answer ([[entities/other]])."),
            ]
        )

        # Run both streams concurrently so their tool calls interleave.
        await asyncio.gather(
            _collect(agent_a, "What is MLX?"),
            _collect(agent_b, "What is Other?"),
        )

        # Each turn owns a distinct capture object...
        assert agent_a._nav_capture is not agent_b._nav_capture
        # ...and each capture only holds the slug its own query navigated.
        assert agent_a._nav_capture.navigated == {"entities/mlx"}
        assert agent_b._nav_capture.navigated == {"entities/other"}


def test_query_answer_model_roundtrip():
    """The FinalAnswer.answer is a real QueryAnswer instance."""
    qa = QueryAnswer(
        answer=MLX_ANSWER,
        citations=[{"slug": "entities/mlx", "title": "MLX"}],
        confidence="high",
        suggestion="",
    )
    assert QueryAnswer.model_validate_json(qa.model_dump_json()) == qa
