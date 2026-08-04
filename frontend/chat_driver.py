"""Streaming chat driver — in-process adapter from the compiled query agent's
``agent.astream(..., stream_mode="messages")`` to a typed event stream that the
Streamlit shell consumes.

Pure Python: no Streamlit import here, so this module is unit-testable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import BaseModel

from agentic_rag.schemas.query import QueryAnswer
from agentic_rag.tools.grounding import build_final_answer

logger = logging.getLogger(__name__)


class ToolStart(BaseModel):
    kind: Literal["tool_start"] = "tool_start"
    name: str
    args: dict  # best-effort parsed args; {} if not parseable yet


class ToolEnd(BaseModel):
    kind: Literal["tool_end"] = "tool_end"
    name: str
    output: str  # tool result string (truncated to ~500 chars for display)


class AnswerToken(BaseModel):
    kind: Literal["answer_token"] = "answer_token"
    text: str  # delta to append to the streaming answer bubble


class FinalAnswer(BaseModel):
    kind: Literal["final"] = "final"
    answer: QueryAnswer  # the validated, cite-or-die-filtered QueryAnswer


StreamEvent = ToolStart | ToolEnd | AnswerToken | FinalAnswer


def _best_effort_args(raw: Any) -> dict:
    """Parse accumulated tool-call args (string fragments or a full dict)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def _on_tool_args_fragment(
    key: Any,
    name: str | None,
    fragment: str,
    accumulated: dict[Any, str],
    seen_starts: set[Any],
    names: dict[Any, str],
) -> list[StreamEvent]:
    """Accumulate one args fragment for a tool call and produce events.

    Shared by the one-chunk path (full args in a single fragment) and the
    multi-chunk incremental path (partial fragments) so both behave identically.

    ``name`` may be None on continuation fragments (real streaming only sends
    the tool name on the first chunk of a call). We track the known name per
    key in ``names`` and defer ``ToolStart`` until the name is known, so a
    nameless fragment never creates an invalid event.
    """
    events: list[StreamEvent] = []
    if name:  # non-empty string -> remember it for this call
        names[key] = name
    accumulated[key] = accumulated.get(key, "") + fragment

    effective_name = names.get(key, "")
    if key not in seen_starts and effective_name:
        seen_starts.add(key)
        events.append(ToolStart(name=effective_name, args=_best_effort_args(accumulated[key])))
    return events


def _tool_call_key(tc: dict) -> Any:
    """A stable per-call identity for streaming fragments.

    Prefer ``index`` over ``id``: in real token streaming the ``id`` appears
    only on the FIRST chunk of a tool call while continuation fragments carry
    ``id=None`` but keep the same ``index``. Keying on ``id`` would split one
    call into two keys and treat the nameless continuation as a new call.
    """
    key = tc.get("index")
    if key is None:
        key = tc.get("id")
    if key is None:
        key = f"tc-{id(tc)}"
    return key


async def stream_query(
    agent,  # compiled query agent (CompiledStateGraph) from build_query_agent
    message: str,
    thread_id: str,
    recursion_limit: int,
) -> AsyncGenerator[StreamEvent, None]:
    """Drive one query turn and yield StreamEvents in order.

    Per-turn setup MUST run first: reset the module-level cite-or-die capture
    so citations from turn N never bleed into turn N+1. Then stream the turn
    with ``agent.astream(..., stream_mode="messages")`` and translate the raw
    message chunks into typed events.
    """
    from agentic_rag.tools.grounding import new_nav_capture

    agent._nav_capture = new_nav_capture()  # fresh cite-or-die capture for this turn

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}

    # Per-tool-call accumulation state for this turn.
    accumulated: dict[Any, str] = {}  # key -> accumulated args JSON string
    seen_starts: set[Any] = set()
    names: dict[Any, str] = {}  # key -> known tool name (first non-empty wins)
    free_text: list[str] = []  # free-text AIMessage content (the live answer)

    async for chunk, _metadata in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage):
            yield ToolEnd(name=chunk.name or "", output=str(chunk.content)[:500])
            continue

        if not isinstance(chunk, (AIMessage, AIMessageChunk)):
            continue

        # Real streaming path: incremental tool_call_chunks with partial args.
        tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
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
        tool_calls = getattr(chunk, "tool_calls", None)
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

        # Plain content with no tool calls: the model's answer text. Stream it
        # live as AnswerTokens; the FinalAnswer render overwrites the bubble
        # at the end with the cite-or-die-validated QueryAnswer.
        content = getattr(chunk, "content", "")
        if content:
            free_text.append(content)
            yield AnswerToken(text=content)

    # Final: finalization is automatic — no finalization tool exists. The
    # answer is synthesized from the streamed free text / last AI message, with
    # [[Page]] links validated against the turn's navigated set (cite-or-die).
    nav_capture = getattr(agent, "_nav_capture", None)
    navigated = nav_capture.navigated if nav_capture is not None else set()
    state_messages = agent.get_state(config).values.get("messages", [])
    answer = build_final_answer(
        state_messages,
        navigated,
        free_text="".join(free_text),
    )
    yield FinalAnswer(answer=answer)
