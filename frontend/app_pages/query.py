"""Streamlit page: the wiki query chat (moved from the old single-file app.py).

Streaming chat over ``frontend/chat_driver.stream_query``: live tool chips,
streamed answer tokens, and the structured ``QueryAnswer`` render (mirrors the
CLI ``query`` command). Durable per-agent transcripts via ``HistoryStore`` and
the shared sidebar thread manager (``ui_common.sidebar`` — thread selector,
"New chat", delete-selected). Contains no business logic — everything testable
lives in ``frontend/chat_driver.py``. The per-turn cite-or-die ``_nav_capture``
reset lives inside ``stream_query``, so every turn starts with a fresh capture
(turn N's citations never bleed into turn N+1).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Streamlit puts the script's directory on sys.path, not the repo root. Make
# the repo root importable so `from frontend.… import …` works both under
# st.navigation and standalone AppTest.from_file (app_pages/ is 3 levels down
# from the repo root).
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend import agents as _agents  # noqa: E402  (needs repo root on sys.path)
from frontend.chat_driver import (  # noqa: E402
    AnswerToken,
    FinalAnswer,
    stream_query,
    ToolEnd,
    ToolStart,
)
from frontend.history_store import DEFAULT_ROOT, HistoryStore  # noqa: E402
from frontend.ui_common import (  # noqa: E402
    init_page,
    render_history,
    sidebar,
)

logger = logging.getLogger(__name__)

# Durable transcript store (survives app restarts).
store = HistoryStore(DEFAULT_ROOT)


def reset_chat() -> None:
    """Start a fresh thread (the old one stays in the durable transcript).

    Preserved from the old single-file app.py; ``ui_common.sidebar``'s
    "New chat" button is the affordance performing exactly this reset.
    """
    st.session_state["query_thread_id"] = str(uuid4())
    st.session_state["query_messages"] = []


# --- Per-session state (per-agent namespace, shared across pages) -----------
tid, messages = init_page("query")

# --- Sidebar: durable thread manager (selector + New chat + delete). ---------
sidebar("query", store)


# --- Rendering helpers (mirror cli.py's query command render). --------------
def _tool_start_label(event: ToolStart) -> str:
    """Chip label: include the query/slug for the navigation tools."""
    args = event.args or {}
    if event.name == "wiki_search" and args.get("query"):
        return f"wiki_search(query={args['query']!r})"
    if event.name == "wiki_read_page" and args.get("slug"):
        return f"wiki_read_page(slug={args['slug']!r})"
    return event.name


def _render_query_answer(answer) -> str:
    """Structured markdown mirroring cli.py: answer + Confidence + Citations
    (``- {slug} - {title}{ (section: {section})}``) + Suggestion if non-empty."""
    from agentic_rag.tools.grounding import render_answer_text

    out = [
        f"**Answer:**\n{render_answer_text(answer.answer)}",
        f"**Confidence:** {answer.confidence}",
    ]
    if answer.citations:
        lines = [
            f"- {c.slug} - {c.title}" + (f" (section: {c.section})" if c.section else "")
            for c in answer.citations
        ]
        out.append("**Citations:**\n" + "\n".join(lines))
    if answer.suggestion:
        out.append(f"**Suggestion:** {answer.suggestion}")
    return "\n\n".join(out)


# --- Streaming bridge --------------------------------------------------------
def _sync_events(agent, message: str, thread_id: str, recursion_limit: int):
    """Iterate stream_query's async generator synchronously.

    Streamlit scripts have no event loop, so drive the async generator with a
    dedicated loop. st.* calls stay on the script thread (no threads involved).
    The generator is explicitly aclosed on exit so a mid-stream exception does
    not leave pending asyncio tasks ("Task was destroyed but it is pending").
    """
    loop = asyncio.new_event_loop()
    agen = stream_query(agent, message, thread_id, recursion_limit)
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        except Exception:
            pass
        loop.close()


def stream_turn(agent, message: str, thread_id: str, recursion_limit: int, chips, answer_ph):
    """Run one turn, updating live tool chips + the streaming answer bubble.

    Returns the final rendered text (structured QueryAnswer block) for the
    session-state chat log. Raises if ``stream_query`` raises.
    """
    running_chips: list[dict] = []  # {"name", "label", "ph", "done"}
    streamed_answer = ""
    final_text = ""

    for event in _sync_events(agent, message, thread_id, recursion_limit):
        if isinstance(event, ToolStart):
            label = _tool_start_label(event)
            ph = chips.empty()
            ph.markdown(f"🔍 {label}…")
            running_chips.append({"name": event.name, "label": label, "ph": ph, "done": False})
        elif isinstance(event, ToolEnd):
            for chip in reversed(running_chips):
                if chip["name"] == event.name and not chip["done"]:
                    chip["ph"].markdown(f"✅ {chip['label']}")
                    chip["done"] = True
                    break
        elif isinstance(event, AnswerToken):
            streamed_answer += event.text
            answer_ph.markdown(streamed_answer)
        elif isinstance(event, FinalAnswer):
            final_text = _render_query_answer(event.answer)
            answer_ph.markdown(final_text)

    # chat_driver always emits FinalAnswer; this is only a defensive fallback.
    return final_text or streamed_answer


# --- Layout ------------------------------------------------------------------
st.title("Wiki Q&A")
st.caption("Streaming chat over the wiki query agent")

# --- Chat history (current thread's durable transcript) ----------------------
render_history(messages)

# --- Chat input ------------------------------------------------------------
if prompt := st.chat_input("Ask the wiki…"):
    messages.append({"role": "user", "content": prompt})
    store.append("query", tid, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chips = st.container()
        answer_ph = st.empty()
        try:
            rendered = stream_turn(
                agent=_agents.get_query_agent(),
                message=prompt,
                thread_id=tid,
                recursion_limit=_agents.agent_config("query", tid)["recursion_limit"],
                chips=chips,
                answer_ph=answer_ph,
            )
            # Record the rendered assistant reply in the chat log + transcript.
            messages.append({"role": "assistant", "content": rendered})
            store.append("query", tid, "assistant", rendered)
        except Exception as exc:
            logger.exception("Query turn failed")
            st.error(f"Query failed: {exc}")
