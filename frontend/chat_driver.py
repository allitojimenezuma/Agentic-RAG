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


def extract_answer_so_far(accumulated_args_json: str) -> str:
    """Tolerant extraction of the `answer` field value from a (possibly partial)
    JSON string matching submit_query_answer's args schema, whose `answer` key is
    first. Reads from the opening quote after `"answer"` to the next unescaped
    quote, honouring `\\"` escapes. Returns the decoded substring accumulated so
    far (may be incomplete if the JSON is mid-stream). Returns "" if the answer
    field has not started or no opening quote is seen yet. Never raises.
    """
    marker = '"answer"'
    idx = accumulated_args_json.find(marker)
    if idx == -1:
        return ""
    pos = idx + len(marker)
    # Skip whitespace, ':', whitespace up to the opening quote of the value.
    while pos < len(accumulated_args_json) and accumulated_args_json[pos].isspace():
        pos += 1
    if pos < len(accumulated_args_json) and accumulated_args_json[pos] == ":":
        pos += 1
    while pos < len(accumulated_args_json) and accumulated_args_json[pos].isspace():
        pos += 1
    if pos >= len(accumulated_args_json) or accumulated_args_json[pos] != '"':
        return ""
    pos += 1  # skip the opening quote

    out: list[str] = []
    while pos < len(accumulated_args_json):
        ch = accumulated_args_json[pos]
        if ch == '"':
            return "".join(out)  # closing quote -> complete value
        if ch == "\\":
            nxt = accumulated_args_json[pos + 1] if pos + 1 < len(accumulated_args_json) else ""
            if nxt == '"':
                out.append('"')
            elif nxt == "\\":
                out.append("\\")
            elif nxt == "n":
                out.append("\n")
            elif nxt == "t":
                out.append("\t")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "":
                return "".join(out)  # dangling backslash at end of stream
            else:
                out.append(nxt)
            pos += 2
            continue
        out.append(ch)
        pos += 1
    return "".join(out)  # end of stream -> incomplete, return what we have


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


def _fallback_answer(text: str) -> QueryAnswer:
    """QueryAnswer used when no structured submit_query_answer result exists."""
    return QueryAnswer(
        answer=text,
        citations=[],
        confidence="low",
        suggestion="(no structured answer produced)",
    )


def _on_tool_args_fragment(
    key: Any,
    name: str | None,
    fragment: str,
    accumulated: dict[Any, str],
    seen_starts: set[Any],
    emitted: dict[Any, str],
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

    if effective_name == "submit_query_answer":
        text = extract_answer_so_far(accumulated[key])
        prev = emitted.get(key, "")
        if text != prev:
            delta = text[len(prev):] if text.startswith(prev) else text
            events.append(AnswerToken(text=delta))
            emitted[key] = text
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
    emitted: dict[Any, str] = {}  # key -> last emitted answer text
    names: dict[Any, str] = {}  # key -> known tool name (first non-empty wins)
    free_text: list[str] = []  # free-text AIMessage content (no tool calls)

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
                    emitted,
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
                    emitted,
                    names,
                ):
                    yield event
            continue

        # Plain content with no tool calls: stream it live as the answer text,
        # but ONLY before any tool has been called this turn. The query agent
        # normally finalizes via submit_query_answer (whose structured render
        # overwrites the bubble at the end); trailing free text after the tool
        # call (e.g. "Done.") is closing chatter, not the answer. For inputs
        # where the model replies with free text only (e.g. a casual "hello"),
        # this is the answer -- mirrors the CLI's compat fallback.
        content = getattr(chunk, "content", "")
        if content and not seen_starts:
            free_text.append(content)
            yield AnswerToken(text=content)

    # Final: emit the structured, cite-or-die-filtered QueryAnswer from the last
    # submit_query_answer ToolMessage in the thread state; fall back if absent.
    state_messages = agent.get_state(config).values.get("messages", [])
    last_submit = None
    last_ai_content = ""
    for msg in state_messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "submit_query_answer":
            last_submit = msg
        if isinstance(msg, (AIMessage, AIMessageChunk)) and getattr(msg, "content", ""):
            last_ai_content = msg.content

    if last_submit is not None:
        try:
            answer = QueryAnswer.model_validate_json(str(last_submit.content))
        except Exception:
            logger.exception("Could not parse submit_query_answer output; falling back")
            answer = _fallback_answer("".join(free_text) or last_ai_content)
    else:
        # No structured finalization: use the streamed free text, or the last
        # AIMessage content (mirrors cli.py's compat fallback).
        answer = _fallback_answer("".join(free_text) or last_ai_content)
    yield FinalAnswer(answer=answer)
