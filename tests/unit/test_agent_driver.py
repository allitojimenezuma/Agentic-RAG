"""Unit tests for frontend/agent_driver: HITL streaming, interrupt detection,
resume cycles, decision-building, and tolerant interrupt parsing.

Uses duck-typed fake agents (no real langgraph) that mirror the
``astream(..., stream_mode=["messages", "values"])`` contract — same approach
as test_query_driver's fake agent — so the event translation, interrupt capture
from values snapshots, and resume semantics are exercised deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCall, ToolMessage
from langgraph.types import Command

from frontend.agent_driver import (
    ALLOWED_DECISIONS,
    build_decisions,
    build_fix_message,
    extract_interrupts,
    FinalMessage,
    InterruptEvent,
    resume_turn,
    stream_turn,
)
from frontend.query_driver import ToolEnd, ToolStart

CONFIG = {"configurable": {"thread_id": "tid"}}
MODEL_META = {"langgraph_node": "model"}
TOOLS_META = {"langgraph_node": "tools"}


def _interrupt(*action_requests: dict) -> SimpleNamespace:
    """Duck-typed langgraph Interrupt: .value holds the action_requests dict."""
    return SimpleNamespace(value={"action_requests": list(action_requests)})


class _FakeAgent:
    """Duck-typed compiled HITL agent: scripted (mode, chunk) steps + state."""

    def __init__(self, steps=None, final_messages=None):
        self._steps = steps or []
        self._final_messages = final_messages or []
        self.calls: list = []  # (inputs, config, stream_mode) per astream call

    async def astream(self, inputs, config, stream_mode="messages"):
        self.calls.append((inputs, config, stream_mode))
        for mode, chunk in self._steps:
            yield mode, chunk

    def get_state(self, config):
        return SimpleNamespace(values={"messages": self._final_messages})


async def _collect(agen) -> list:
    return [event async for event in agen]


class TestStreamTurn:
    async def test_plain_turn_event_order(self):
        """Acceptance: ToolStart -> ToolEnd -> AnswerToken* -> FinalMessage."""
        agent = _FakeAgent(
            steps=[
                (
                    "messages",
                    (
                        AIMessage(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    name="delete_wiki_page",
                                    args={"page_slug": "entities/mlx"},
                                    id="tc-1",
                                )
                            ],
                        ),
                        MODEL_META,
                    ),
                ),
                (
                    "messages",
                    (
                        ToolMessage(
                            content="Wiki page deleted",
                            name="delete_wiki_page",
                            tool_call_id="tc-1",
                        ),
                        TOOLS_META,
                    ),
                ),
                ("messages", (AIMessage(content="Deleted entities/mlx."), MODEL_META)),
                ("values", {"messages": [AIMessage(content="Deleted entities/mlx.")]}),
            ],
            final_messages=[AIMessage(content="Deleted entities/mlx.")],
        )

        events = await _collect(
            stream_turn(agent, "delete entities/mlx", CONFIG, "ingest")
        )

        assert [e.kind for e in events] == [
            "tool_start",
            "tool_end",
            "answer_token",
            "final_message",
        ]
        start = next(e for e in events if isinstance(e, ToolStart))
        assert start.name == "delete_wiki_page"
        assert start.args == {"page_slug": "entities/mlx"}
        end = next(e for e in events if isinstance(e, ToolEnd))
        assert end.name == "delete_wiki_page"
        assert end.output == "Wiki page deleted"
        tokens = [e for e in events if e.kind == "answer_token"]
        assert "".join(t.text for t in tokens) == "Deleted entities/mlx."
        final = events[-1]
        assert isinstance(final, FinalMessage)
        assert final.text == "Deleted entities/mlx."

    async def test_inputs_and_config_shape(self):
        """stream_turn passes the exact inputs/config shapes and both stream modes."""
        agent = _FakeAgent(
            steps=[("values", {"messages": []})],
            final_messages=[AIMessage(content="ok")],
        )

        events = await _collect(
            stream_turn(
                agent,
                "hello",
                {"configurable": {"thread_id": "tid"}, "recursion_limit": 25},
                "ingest",
            )
        )

        inputs, config, stream_mode = agent.calls[0]
        assert inputs == {"messages": [{"role": "user", "content": "hello"}]}
        assert config == {
            "configurable": {"thread_id": "tid"},
            "recursion_limit": 25,
        }
        assert stream_mode == ["messages", "values"]
        assert [e.kind for e in events] == ["final_message"]
        assert events[-1].text == "ok"

    async def test_interrupt_turn_closes_tool_lifecycle(self):
        """Acceptance: interrupt turn -> ... -> ToolEnd("⏸ awaiting human
        approval") per action -> ONE InterruptEvent; no FinalMessage."""
        actions = [
            {"name": "delete_wiki_page", "args": {"page_slug": "entities/mlx"}},
            {"name": "delete_wiki_page", "args": {"page_slug": "entities/other"}},
        ]
        agent = _FakeAgent(
            steps=[
                (
                    "messages",
                    (
                        AIMessage(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    name="delete_wiki_page",
                                    args={"page_slug": "entities/mlx"},
                                    id="tc-1",
                                ),
                                ToolCall(
                                    name="delete_wiki_page",
                                    args={"page_slug": "entities/other"},
                                    id="tc-2",
                                ),
                            ],
                        ),
                        MODEL_META,
                    ),
                ),
                ("values", {"__interrupt__": [_interrupt(*actions)], "messages": []}),
            ],
            final_messages=[AIMessage(content="must never surface")],
        )

        events = await _collect(
            stream_turn(agent, "delete both", CONFIG, "ingest")
        )

        kinds = [e.kind for e in events]
        assert kinds == ["tool_start", "tool_start", "tool_end", "tool_end", "interrupt"]
        assert "final_message" not in kinds

        paused = [e for e in events if isinstance(e, ToolEnd)]
        assert [e.name for e in paused] == ["delete_wiki_page", "delete_wiki_page"]
        assert all(e.output == "⏸ awaiting human approval" for e in paused)

        interrupts = [e for e in events if isinstance(e, InterruptEvent)]
        assert len(interrupts) == 1
        assert interrupts[0].actions == actions

    async def test_incremental_tool_call_chunks_translation(self):
        """Real streaming path: partial args fragments accumulate; ToolStart
        is deferred until the tool name is known (continuation chunks carry
        name=None)."""
        agent = _FakeAgent(
            steps=[
                (
                    "messages",
                    (
                        AIMessageChunk(
                            content="",
                            tool_call_chunks=[
                                {
                                    "name": None,
                                    "args": '{"slug": "entities/a"',
                                    "id": None,
                                    "index": 0,
                                }
                            ],
                        ),
                        MODEL_META,
                    ),
                ),
                (
                    "messages",
                    (
                        AIMessageChunk(
                            content="",
                            tool_call_chunks=[
                                {
                                    "name": "flag_contradiction",
                                    "args": "}",
                                    "id": "tc-1",
                                    "index": 0,
                                }
                            ],
                        ),
                        MODEL_META,
                    ),
                ),
                (
                    "values",
                    {
                        "__interrupt__": [
                            _interrupt(
                                {
                                    "name": "flag_contradiction",
                                    "args": {"slug": "entities/a"},
                                }
                            )
                        ],
                        "messages": [],
                    },
                ),
            ]
        )

        events = await _collect(stream_turn(agent, "check a", CONFIG, "lint"))

        # First (nameless) chunk produces no event; the named chunk emits
        # ToolStart; the interrupt emits synthetic ToolEnd + InterruptEvent.
        assert [e.kind for e in events] == ["tool_start", "tool_end", "interrupt"]
        start = next(e for e in events if isinstance(e, ToolStart))
        assert start.name == "flag_contradiction"
        assert isinstance(events[-1], InterruptEvent)
        assert events[-1].actions == [
            {"name": "flag_contradiction", "args": {"slug": "entities/a"}}
        ]

    async def test_final_message_falls_back_to_free_text(self):
        """No AI message in the thread state -> FinalMessage reuses the
        streamed free text (cli.py echoes the last message content)."""
        agent = _FakeAgent(
            steps=[
                ("messages", (AIMessageChunk(content="Hello "), MODEL_META)),
                ("messages", (AIMessageChunk(content="world"), MODEL_META)),
                ("values", {"messages": []}),
            ],
            final_messages=[ToolMessage(content="x", name="t", tool_call_id="1")],
        )

        events = await _collect(stream_turn(agent, "hi", CONFIG, "ingest"))

        assert [e.kind for e in events] == [
            "answer_token",
            "answer_token",
            "final_message",
        ]
        assert events[-1].text == "Hello world"

    async def test_agent_exception_propagates(self):
        """Agent failures are re-raised (the Streamlit shell catches them)."""

        class _BoomAgent:
            async def astream(self, inputs, config, stream_mode="messages"):
                raise RuntimeError("agent exploded")
                yield  # pragma: no cover — make astream an async generator

        with pytest.raises(RuntimeError, match="agent exploded"):
            await _collect(stream_turn(_BoomAgent(), "x", CONFIG, "ingest"))


class TestResumeTurn:
    async def test_full_resume_cycle(self):
        """Acceptance: stream_turn -> InterruptEvent -> resume_turn -> FinalMessage."""
        interrupt_steps = [
            (
                "messages",
                (
                    AIMessage(
                        content="",
                        tool_calls=[
                            ToolCall(
                                name="flag_contradiction",
                                args={
                                    "slug": "entities/a",
                                    "proposed_resolution": "fix it",
                                },
                                id="tc-1",
                            )
                        ],
                    ),
                    MODEL_META,
                ),
            ),
            (
                "values",
                {
                    "__interrupt__": [
                        _interrupt(
                            {
                                "name": "flag_contradiction",
                                "args": {
                                    "slug": "entities/a",
                                    "proposed_resolution": "fix it",
                                },
                            }
                        )
                    ],
                    "messages": [],
                },
            ),
        ]
        agent = _FakeAgent(interrupt_steps)

        events = await _collect(stream_turn(agent, "check a", CONFIG, "lint"))
        interrupt = next(e for e in events if isinstance(e, InterruptEvent))
        assert interrupt.actions == [
            {
                "name": "flag_contradiction",
                "args": {"slug": "entities/a", "proposed_resolution": "fix it"},
            }
        ]

        decisions = build_decisions("approve", interrupt.actions)

        # The resumed turn runs the tool and streams the final answer.
        agent._steps = [
            (
                "messages",
                (
                    ToolMessage(
                        content="ok",
                        name="flag_contradiction",
                        tool_call_id="tc-1",
                    ),
                    TOOLS_META,
                ),
            ),
            ("messages", (AIMessage(content="Contradiction flagged."), MODEL_META)),
            ("values", {"messages": []}),
        ]
        agent._final_messages = [AIMessage(content="Contradiction flagged.")]

        events2 = await _collect(
            resume_turn(agent, decisions, CONFIG, "lint")
        )

        assert [e.kind for e in events2] == [
            "tool_end",
            "answer_token",
            "final_message",
        ]
        assert events2[-1].text == "Contradiction flagged."

        # Resume drives a Command(resume=...) with the built decisions.
        inputs, config, stream_mode = agent.calls[1]
        assert isinstance(inputs, Command)
        assert inputs.resume == {"decisions": decisions}
        assert config == CONFIG
        assert stream_mode == ["messages", "values"]

    async def test_resume_can_yield_second_interrupt(self):
        """Multi-interrupt chains: resume may interrupt again."""
        agent = _FakeAgent(
            steps=[
                (
                    "values",
                    {
                        "__interrupt__": [
                            _interrupt(
                                {"name": "delete_wiki_page", "args": {"page_slug": "b"}}
                            )
                        ],
                        "messages": [],
                    },
                )
            ]
        )

        events = await _collect(
            resume_turn(agent, [{"type": "approve"}], CONFIG, "fix")
        )

        assert [e.kind for e in events] == ["tool_end", "interrupt"]
        assert events[0].output == "⏸ awaiting human approval"
        assert events[1].actions == [
            {"name": "delete_wiki_page", "args": {"page_slug": "b"}}
        ]


class TestExtractInterrupts:
    def test_no_interrupt_key(self):
        assert extract_interrupts({}) == []
        assert extract_interrupts({"messages": []}) == []

    def test_interrupt_object_with_value(self):
        state = {
            "__interrupt__": [
                _interrupt({"name": "delete_wiki_page", "args": {"page_slug": "x"}})
            ]
        }
        assert extract_interrupts(state) == [
            {"name": "delete_wiki_page", "args": {"page_slug": "x"}}
        ]

    def test_bare_dict_interrupt(self):
        state = {
            "__interrupt__": [
                {"action_requests": [{"name": "flag_contradiction", "args": {"slug": "y"}}]}
            ]
        }
        assert extract_interrupts(state) == [
            {"name": "flag_contradiction", "args": {"slug": "y"}}
        ]

    def test_single_interrupt_not_wrapped_in_list(self):
        state = {"__interrupt__": _interrupt({"name": "delete_wiki_page", "args": {}})}
        assert extract_interrupts(state) == [
            {"name": "delete_wiki_page", "args": {}}
        ]

    def test_flattens_across_multiple_interrupts(self):
        state = {
            "__interrupt__": [
                _interrupt({"name": "a", "args": {"x": 1}}),
                _interrupt({"name": "b", "args": {"y": 2}}, {"name": "c", "args": {}}),
            ]
        }
        assert extract_interrupts(state) == [
            {"name": "a", "args": {"x": 1}},
            {"name": "b", "args": {"y": 2}},
            {"name": "c", "args": {}},
        ]

    def test_tolerates_garbage(self):
        class _BoomValue:
            def get(self, key, default=None):
                raise RuntimeError("boom")

        assert extract_interrupts(None) == []
        assert extract_interrupts({"__interrupt__": [42, None, "junk"]}) == []
        assert extract_interrupts({"__interrupt__": [SimpleNamespace(value=object())]}) == []
        # raw.get raises -> the whole interrupt entry degrades to no actions
        assert (
            extract_interrupts({"__interrupt__": [SimpleNamespace(value=_BoomValue())]})
            == []
        )
        # non-dict action entries are skipped, non-dict args normalize to {}
        state = {
            "__interrupt__": [
                _interrupt({"name": "ok", "args": "not-a-dict"}, "naked-string")
            ]
        }
        assert extract_interrupts(state) == [{"name": "ok", "args": {}}]


class TestBuildDecisions:
    def test_approve_shapes(self):
        actions = [{"name": "a", "args": {}}, {"name": "b", "args": {}}]
        assert build_decisions("approve", actions) == [
            {"type": "approve"},
            {"type": "approve"},
        ]
        # empty actions -> at least one approval (mirrors cli.py)
        assert build_decisions("approve", []) == [{"type": "approve"}]

    def test_reject_shapes(self):
        actions = [{"name": "a", "args": {}}]
        assert build_decisions("reject", actions, feedback="no") == [
            {"type": "reject", "feedback": "no"}
        ]
        assert build_decisions("reject", []) == [{"type": "reject", "feedback": ""}]

    def test_edit_shape(self):
        actions = [
            {"name": "flag_contradiction", "args": {"slug": "entities/a"}},
            {"name": "flag_contradiction", "args": {"slug": "entities/b"}},
        ]
        decisions = build_decisions(
            "edit", actions, index=1, new_resolution="new text"
        )
        assert decisions == [
            {"type": "approve"},
            {
                "type": "edit",
                "edited_action": {
                    "name": "flag_contradiction",
                    "args": {"slug": "entities/b", "proposed_resolution": "new text"},
                },
            },
        ]

    def test_edit_overrides_and_preserves_other_args(self):
        actions = [
            {
                "name": "flag_contradiction",
                "args": {"slug": "x", "proposed_resolution": "old"},
            }
        ]
        decisions = build_decisions(
            "edit", actions, index=0, new_resolution="override"
        )
        assert decisions[0]["edited_action"]["args"] == {
            "slug": "x",
            "proposed_resolution": "override",
        }

    def test_unknown_choice_raises(self):
        with pytest.raises(ValueError):
            build_decisions("nuke", [])

    def test_allowed_decisions_map_mirrors_middleware(self):
        assert ALLOWED_DECISIONS == {
            "delete_wiki_page": ["approve", "reject"],
            "flag_contradiction": ["approve", "edit", "reject"],
        }


class TestBuildFixMessage:
    """build_fix_message mirrors cli.py's fix-command message construction.

    health_check is monkeypatched (the function lazy-imports it) so no real
    wiki is touched and no LLM is involved.
    """

    @staticmethod
    def _patch_report(monkeypatch, issues):
        import agentic_rag.wiki.health as health_mod

        report = SimpleNamespace(issues=issues)
        monkeypatch.setattr(health_mod, "health_check", lambda _path: report)
        return report

    @staticmethod
    def _issue(kind, slug, detail):
        return SimpleNamespace(kind=kind, slug=slug, detail=detail)

    def test_issue_lines_shape(self, monkeypatch):
        """Acceptance: "Fix these lint issues:\n- [kind] slug: detail" per issue."""
        issues = [
            self._issue("orphan", "entities/a", "No inbound links from other pages"),
            self._issue(
                "broken-link", "entities/b", "Unresolved link(s): [[missing]]"
            ),
        ]
        self._patch_report(monkeypatch, issues)
        assert build_fix_message("latest", Path("/wiki")) == (
            "Fix these lint issues:\n"
            "- [orphan] entities/a: No inbound links from other pages\n"
            "- [broken-link] entities/b: Unresolved link(s): [[missing]]"
        )

    def test_no_issues(self, monkeypatch):
        self._patch_report(monkeypatch, [])
        assert build_fix_message("latest", Path("/wiki")) == "No issues"

    def test_empty_issue_means_no_filter(self, monkeypatch):
        issues = [self._issue("orphan", "entities/a", "d")]
        self._patch_report(monkeypatch, issues)
        assert build_fix_message("", Path("/wiki")) == (
            "Fix these lint issues:\n- [orphan] entities/a: d"
        )

    def test_filter_by_kind(self, monkeypatch):
        issues = [
            self._issue("orphan", "entities/a", "d1"),
            self._issue("missing-index", "entities/b", "d2"),
        ]
        self._patch_report(monkeypatch, issues)
        assert build_fix_message("missing-index", Path("/wiki")) == (
            "Fix these lint issues:\n- [missing-index] entities/b: d2"
        )

    def test_filter_by_slug(self, monkeypatch):
        issues = [
            self._issue("orphan", "entities/mlx", "d1"),
            self._issue("orphan", "entities/other", "d2"),
        ]
        self._patch_report(monkeypatch, issues)
        assert build_fix_message("entities/mlx", Path("/wiki")) == (
            "Fix these lint issues:\n- [orphan] entities/mlx: d1"
        )

    def test_no_match_after_filter_passes_request_through_with_context(self, monkeypatch):
        """cli.py filter_mismatch branch: unmatched filter/instruction keeps the
        user's words and appends the deterministic issues as context."""
        issues = [self._issue("orphan", "entities/a", "d")]
        self._patch_report(monkeypatch, issues)
        assert build_fix_message(
            "please fix the mlx page frontmatter", Path("/wiki")
        ) == (
            "please fix the mlx page frontmatter\n\n"
            "The deterministic health check found 1 issues "
            "(for context — address them only if relevant to the request):\n"
            "- [orphan] entities/a: d"
        )

    def test_no_match_after_filter_passes_request_through_when_clean(self, monkeypatch):
        """cli.py filter_mismatch branch with a clean wiki: user's words plus
        the structurally-clean note — never the bare "No issues" swallow."""
        self._patch_report(monkeypatch, [])
        assert build_fix_message("stale", Path("/wiki")) == (
            "stale\n\n"
            "Note: the deterministic health check found no issues — "
            "the wiki is structurally clean."
        )

    def test_health_check_receives_wiki_path(self, monkeypatch):
        import agentic_rag.wiki.health as health_mod

        captured = {}

        def fake_check(path):
            captured["path"] = path
            return SimpleNamespace(issues=[])

        monkeypatch.setattr(health_mod, "health_check", fake_check)
        assert build_fix_message("latest", Path("/tmp/wiki")) == "No issues"
        assert captured["path"] == Path("/tmp/wiki")
