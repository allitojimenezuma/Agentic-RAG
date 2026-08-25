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
    args: dict  # full parsed args; {} when they never parsed (fallback path)
    call_id: str = ""  # stable streaming key (tool_call id when known)


class ToolEnd(BaseModel):
    kind: Literal["tool_end"] = "tool_end"
    name: str
    output: str  # tool result string (truncated to ~500 chars for display)
    call_id: str = ""  # tool_call_id of the completed call, when known


class AnswerToken(BaseModel):
    kind: Literal["answer_token"] = "answer_token"
    text: str  # delta to append to the streaming answer bubble


class FinalAnswer(BaseModel):
    kind: Literal["final"] = "final"
    answer: QueryAnswer  # the validated, cite-or-die-filtered QueryAnswer


StreamEvent = ToolStart | ToolEnd | AnswerToken | FinalAnswer


def _best_effort_args(raw: Any) -> dict | None:
    """Parse accumulated tool-call args; ``None`` while still incomplete.

    Returns a dict once the accumulated text parses into a complete JSON
    object, and ``None`` for partial JSON (mid-stream), non-dict payloads, or
    empty text. The streaming drivers defer ``ToolStart`` until args are
    complete so chips always show the full parameters instead of a partial
    first-chunk dict.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, TypeError):
        return None


def _on_tool_args_fragment(
    key: Any,
    name: str | None,
    fragment: str,
    accumulated: dict[Any, str],
    seen_starts: set[Any],
    names: dict[Any, str],
    started_names: set[str],
) -> list[StreamEvent]:
    """Accumulate one args fragment for a tool call and produce events.

    Shared by the one-chunk path (full args in a single fragment) and the
    multi-chunk incremental path (partial fragments) so both behave identically.

    ``name`` may be None on continuation fragments (real streaming only sends
    the tool name on the first chunk of a call). We track the known name per
    key in ``names`` and defer ``ToolStart`` until the name is known AND the
    accumulated args parse into a complete JSON object — the chip then shows
    the FULL parameters. Calls whose args never parse are surfaced by the
    caller's ToolMessage/interrupt fallback, so no tool call is ever missed.
    """
    events: list[StreamEvent] = []
    if name:  # non-empty string -> remember it for this call
        names[key] = name
    accumulated[key] = accumulated.get(key, "") + fragment

    effective_name = names.get(key, "")
    if key not in seen_starts and effective_name:
        args = _best_effort_args(accumulated[key])
        if args is not None:  # complete, parseable JSON object
            seen_starts.add(key)
            started_names.add(effective_name)
            events.append(
                ToolStart(name=effective_name, args=args, call_id=str(key))
            )
    return events


def _tool_call_key(tc: dict, last_id: dict | None = None) -> Any:
    """A stable per-call identity for streaming fragments.

    A call's first chunk carries ``index`` + ``id`` + ``name``; continuation
    chunks repeat the same ``index`` with ``id=None``. Sequential calls in
    LATER model messages restart ``index`` at 0, so keying on ``index`` alone
    would merge the second call into the first and swallow its ``ToolStart``
    — chips would silently miss tool calls. Key on ``id`` when present; for
    id-less continuations fall back to the key last seen on that ``index``
    (``last_id`` is updated BEFORE the key is computed, so a fresh id on a
    reused index starts a new call). Providers with no ids at all fall back to
    an index placeholder — safe for single-fragment calls, the same limit the
    old index-only key had.
    """
    call_id = tc.get("id")
    index = tc.get("index")
    if call_id:
        key = call_id
    elif index is not None and last_id is not None and index in last_id:
        key = last_id[index]
    elif index is not None:
        key = ("index", index)
    else:
        key = f"tc-{id(tc)}"
    if index is not None and last_id is not None:
        last_id[index] = key
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

    config = {
        "configurable": {"thread_id": thread_id, "nav_capture": agent._nav_capture},
        "recursion_limit": recursion_limit,
    }

    # Per-tool-call accumulation state for this turn.
    accumulated: dict[Any, str] = {}  # key -> accumulated args JSON string
    seen_starts: set[Any] = set()
    names: dict[Any, str] = {}  # key -> known tool name (first non-empty wins)
    last_id: dict[Any, Any] = {}  # index -> key currently owning it
    started_names: set[str] = set()  # tool names already surfaced as ToolStart
    free_text: list[str] = []  # free-text AIMessage content (the live answer)

    async for chunk, _metadata in agent.astream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage):
            name = chunk.name or ""
            call_id = chunk.tool_call_id or ""
            if name and name not in started_names:
                # The call's args never completed/parsed: synthesize the
                # missing ToolStart so the tool call still shows on screen.
                started_names.add(name)
                yield ToolStart(name=name, args={}, call_id=call_id)
            yield ToolEnd(name=name, output=str(chunk.content)[:500], call_id=call_id)
            continue

        if not isinstance(chunk, (AIMessage, AIMessageChunk)):
            continue

        # Real streaming path: incremental tool_call_chunks with partial args.
        tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
        if tool_call_chunks:
            for tc in tool_call_chunks:
                for event in _on_tool_args_fragment(
                    _tool_call_key(tc, last_id),
                    tc.get("name") or "",
                    tc.get("args", "") or "",
                    accumulated,
                    seen_starts,
                    names,
                    started_names,
                ):
                    yield event
            continue

        # Non-streaming path (e.g. ScriptedChatModel): full args dict in one
        # AIMessage.tool_calls — normalize to the same single-fragment handling.
        tool_calls = getattr(chunk, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                for event in _on_tool_args_fragment(
                    tc.get("id") or f"raw-{id(tc)}",
                    tc.get("name") or "",
                    json.dumps(tc.get("args", {})),
                    accumulated,
                    seen_starts,
                    names,
                    started_names,
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
