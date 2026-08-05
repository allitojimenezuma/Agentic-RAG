"""Generic streaming driver for HITL-capable agents (ingest / lint / fix).

Pure Python: no Streamlit import here, so this module is unit-testable.

Mirrors the CLI's HITL loop (``src/agentic_rag/cli.py``): the same event
translation as ``query_driver.stream_query`` for ``stream_mode="messages"``
(reusing its ToolStart/ToolEnd/AnswerToken event types), plus interrupt
detection from ``stream_mode="values"`` snapshots (the ``__interrupt__`` key),
synthetic ``ToolEnd("⏸ awaiting human approval")`` events so every ToolStart
chip closes, and ``Command(resume=...)`` resume with CLI-identical decision
shapes built by :func:`build_decisions`.

Interrupts are captured from values snapshots DURING streaming — a
``get_state()`` call afterwards does not expose them — so the driver watches
the values channel while consuming the stream. There is no cite-or-die
finalization here (that is query-only): the terminal event is
:class:`FinalMessage`, equivalent to the CLI's ``result["messages"][-1].content``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel

from frontend.query_driver import (
    _on_tool_args_fragment,
    _tool_call_key,
    AnswerToken,
    StreamEvent,
    ToolEnd,
    ToolStart,
)

logger = logging.getLogger(__name__)

PAUSED_TOOL_OUTPUT = "⏸ awaiting human approval"


class InterruptEvent(BaseModel):
    kind: Literal["interrupt"] = "interrupt"
    actions: list[dict]  # parsed action_requests: [{"name": str, "args": dict}]


class FinalMessage(BaseModel):
    kind: Literal["final_message"] = "final_message"
    text: str  # last assistant message content


AgentEvent = StreamEvent | InterruptEvent | FinalMessage

# Mirrors the middleware HITL configs in agents/ingest.py and agents/fix.py.
ALLOWED_DECISIONS: dict[str, list[str]] = {
    "delete_wiki_page": ["approve", "reject"],
    "flag_contradiction": ["approve", "edit", "reject"],
}


def extract_interrupts(state: dict) -> list[dict]:
    """Parse pending action requests from a state snapshot. Tolerant, never raises.

    Mirrors cli.py: each ``__interrupt__`` entry may be a langgraph Interrupt
    (``.value`` attribute) or a bare dict; ``action_requests`` is read off the
    raw value when it supports ``.get``; any exception yields ``[]``. Actions
    are normalized to the ``{"name": str, "args": dict}`` shape.
    """
    if not isinstance(state, dict):
        return []
    raw_interrupts = state.get("__interrupt__")
    if raw_interrupts is None:
        return []
    if not isinstance(raw_interrupts, (list, tuple)):
        raw_interrupts = [raw_interrupts]

    actions: list[dict] = []
    for interrupt in raw_interrupts:
        raw = getattr(interrupt, "value", interrupt)
        try:
            found = raw.get("action_requests", []) if hasattr(raw, "get") else []
        except Exception:
            found = []
        if not isinstance(found, (list, tuple)):
            continue
        for action in found:
            if not isinstance(action, dict):
                continue
            args = action.get("args", {})
            actions.append(
                {
                    "name": action.get("name", ""),
                    "args": args if isinstance(args, dict) else {},
                }
            )
    return actions


def build_decisions(
    choice: str,
    actions: list[dict],
    *,
    feedback: str = "",
    index: int = 0,
    new_resolution: str = "",
) -> list[dict]:
    """Build the CLI-identical ``Command(resume={"decisions": [...]})`` payload.

    Exactly mirrors cli.py's ingest HITL shapes:

    - ``approve``: ``[{"type": "approve"}] * max(len(actions), 1)``
    - ``reject``: ``[{"type": "reject", "feedback": feedback}] * max(len(actions), 1)``
    - ``edit``: ``[{"type": "approve"}] * len(actions)`` with ``decisions[index]``
      replaced by ``{"type": "edit", "edited_action": {"name": ...,
      "args": {**args, "proposed_resolution": new_resolution}}}`` (actions
      assumed non-empty — the UI only offers edit for flag_contradiction).
    """
    if choice == "approve":
        return [{"type": "approve"}] * max(len(actions), 1)
    if choice == "reject":
        return [{"type": "reject", "feedback": feedback}] * max(len(actions), 1)
    if choice == "edit":
        target = actions[index] if isinstance(actions[index], dict) else {}
        target_args = target.get("args", {}) if isinstance(target, dict) else {}
        decisions = [{"type": "approve"}] * len(actions)
        decisions[index] = {
            "type": "edit",
            "edited_action": {
                "name": target.get("name", "flag_contradiction") if isinstance(target, dict) else "flag_contradiction",
                "args": {**target_args, "proposed_resolution": new_resolution},
            },
        }
        return decisions
    raise ValueError(f"Unknown decision choice: {choice!r} (expected approve/reject/edit)")


def build_fix_message(issue: str, wiki_path: Path) -> str:
    """Build the fix-agent user message exactly like cli.py's ``fix`` command.

    Runs the deterministic ``health_check(wiki_path)``, applies the CLI's
    ``issue`` filter (needle matched case-folded against ``kind``/``slug``;
    ``"latest"``/empty means no filter), and returns either
    ``"Fix these lint issues:\n- [kind] slug: detail"`` — one line per issue,
    byte-identical to cli.py's ``fix`` construction — or ``"No issues"``
    when the wiki is clean. When the filter matches nothing, cli.py's
    ``filter_mismatch`` branch applies: the user's own words become the
    instruction, with the deterministic issues appended as context (or a
    "no issues" note when the wiki is clean). Pure (no Streamlit import);
    ``health_check`` exceptions propagate (the Streamlit shell surfaces them).
    """
    from agentic_rag.wiki.health import health_check

    report = health_check(wiki_path)
    issues = report.issues
    filter_mismatch = False
    if issue and issue != "latest":
        needle = issue.lower()
        issues = [
            i for i in issues
            if needle in i.kind.lower() or needle in i.slug.lower()
        ]
        if not issues:
            # Natural-language instruction (or typo'd filter), NOT a clean
            # bill of health — mirror cli.py: keep the user's words as the
            # instruction, append the deterministic issues as context.
            filter_mismatch = True
    if filter_mismatch:
        if report.issues:
            context = "\n".join(
                f"- [{i.kind}] {i.slug}: {i.detail}" for i in report.issues
            )
            return (
                f"{issue}\n\n"
                f"The deterministic health check found {len(report.issues)} issues "
                f"(for context — address them only if relevant to the request):\n{context}"
            )
        return (
            f"{issue}\n\n"
            "Note: the deterministic health check found no issues — "
            "the wiki is structurally clean."
        )
    if issues:
        lines = ["Fix these lint issues:"]
        for i in issues:
            lines.append(f"- [{i.kind}] {i.slug}: {i.detail}")
        return "\n".join(lines)
    return "No issues"


def _state_values(state: Any) -> dict:
    """Best-effort ``values`` dict from a langgraph state snapshot."""
    if state is None:
        return {}
    values = getattr(state, "values", None)
    if isinstance(values, dict):
        return values
    if isinstance(state, dict):
        nested = state.get("values")
        return nested if isinstance(nested, dict) else state
    return {}


def _last_ai_text(state: Any, fallback: str) -> str:
    """Last assistant message content — cli.py's ``result["messages"][-1].content``."""
    messages = _state_values(state).get("messages", [])
    if isinstance(messages, (list, tuple)):
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                text = content if isinstance(content, str) else str(content)
                if text:
                    return text
    return fallback


async def _drive_turn(agent, inputs, config) -> AsyncGenerator[AgentEvent, None]:
    """Shared streaming loop for :func:`stream_turn` / :func:`resume_turn`.

    Watches ``stream_mode=["messages", "values"]``: ``messages`` is translated
    exactly like ``query_driver.stream_query`` (ToolMessage -> ToolEnd,
    tool_call_chunks/tool_calls -> ToolStart, plain content -> AnswerToken);
    ``values`` is ignored except for the ``__interrupt__`` key, which emits one
    synthetic ``ToolEnd("⏸ awaiting human approval")`` per pending action plus
    ONE :class:`InterruptEvent` and suppresses the FinalMessage.
    """
    # Per-tool-call accumulation state for this turn (see query_driver.py).
    accumulated: dict[Any, str] = {}  # key -> accumulated args JSON string
    seen_starts: set[Any] = set()
    names: dict[Any, str] = {}  # key -> known tool name (first non-empty wins)
    free_text: list[str] = []  # free-text AIMessage content (the live answer)
    interrupt_fired = False

    async for mode, chunk in agent.astream(
        inputs, config, stream_mode=["messages", "values"]
    ):
        if mode == "messages":
            msg, _metadata = chunk if isinstance(chunk, tuple) else (chunk, None)

            if isinstance(msg, ToolMessage):
                yield ToolEnd(name=msg.name or "", output=str(msg.content)[:500])
                continue

            if not isinstance(msg, (AIMessage, AIMessageChunk)):
                continue

            # Real streaming path: incremental tool_call_chunks with partial args.
            tool_call_chunks = getattr(msg, "tool_call_chunks", None)
            if tool_call_chunks:
                for tc in tool_call_chunks:
                    for event in _on_tool_args_fragment(
                        _tool_call_key(tc),
                        tc.get("name") or "",
                        tc.get("args", "") or "",
                        accumulated,
                        seen_starts,
                        names,
                    ):
                        yield event
                continue

            # Non-streaming path (e.g. ScriptedChatModel): full args dict in one
            # AIMessage.tool_calls — normalize to the same single-fragment handling.
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    for event in _on_tool_args_fragment(
                        tc.get("id") or _tool_call_key(tc),
                        tc.get("name") or "",
                        json.dumps(tc.get("args", {})),
                        accumulated,
                        seen_starts,
                        names,
                    ):
                        yield event
                continue

            # Plain content with no tool calls: the model's answer text.
            content = getattr(msg, "content", "")
            if content:
                free_text.append(content)
                yield AnswerToken(text=content)

        elif mode == "values" and isinstance(chunk, dict) and "__interrupt__" in chunk:
            # Interrupts appear in the values snapshot DURING streaming only.
            interrupt_fired = True
            actions = extract_interrupts(chunk)
            for action in actions:
                yield ToolEnd(name=action.get("name", ""), output=PAUSED_TOOL_OUTPUT)
            yield InterruptEvent(actions=actions)

    if interrupt_fired:
        return  # no FinalMessage after an interrupt

    state = agent.get_state(config)
    yield FinalMessage(text=_last_ai_text(state, "".join(free_text)))


async def stream_turn(
    agent, message: str, config: dict, agent_name: str
) -> AsyncGenerator[AgentEvent, None]:
    """Drive one fresh HITL turn and yield typed events in order.

    ``config`` is the full pinned per-agent config dict. On interrupt: emits
    one synthetic ``ToolEnd("⏸ awaiting human approval")`` per pending action
    (so every ToolStart has a matching ToolEnd) followed by ONE
    :class:`InterruptEvent` and nothing further. Otherwise emits a terminal
    :class:`FinalMessage`. Agent exceptions are re-raised (the Streamlit shell
    catches them).
    """
    logger.info("agent_turn start agent=%s", agent_name)
    inputs = {"messages": [{"role": "user", "content": message}]}
    async for event in _drive_turn(agent, inputs, config):
        yield event
    logger.info("agent_turn end agent=%s", agent_name)


async def resume_turn(
    agent, decisions: list[dict], config: dict, agent_name: str
) -> AsyncGenerator[AgentEvent, None]:
    """Resume an interrupted turn with the user's decisions.

    Drives ``agent.astream(Command(resume={"decisions": decisions}), ...)`` with
    the same translation as :func:`stream_turn`. May yield another
    :class:`InterruptEvent` (multi-interrupt chains) or a :class:`FinalMessage`.
    """
    logger.info("agent_resume start agent=%s", agent_name)
    inputs = Command(resume={"decisions": decisions})
    async for event in _drive_turn(agent, inputs, config):
        yield event
    logger.info("agent_resume end agent=%s", agent_name)
