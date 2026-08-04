"""Streamlit chat shell for the wiki query agent.

Single-file UI over ``frontend/chat_driver.stream_query``: live tool chips,
streamed answer tokens, and the structured ``QueryAnswer`` render (mirrors the
CLI ``query`` command). Contains no business logic — everything testable lives
in ``frontend/chat_driver.py``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Streamlit puts the script's directory (frontend/) on sys.path, not the repo
# root. Make the repo root importable so `from frontend.chat_driver import …`
# works exactly as in tests/unit/test_chat_driver.py.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.chat_driver import (  # noqa: E402  (needs repo root on sys.path)
    AnswerToken,
    FinalAnswer,
    stream_query,
    ToolEnd,
    ToolStart,
)

logger = logging.getLogger(__name__)

# --- 1. Settings: surface config errors in the UI, never crash. -------------
try:
    from agentic_rag.config import Settings

    settings = Settings()
except Exception as exc:
    st.error(
        f"Could not load configuration: {exc}\n\n"
        "Make sure `OPENAI_API_KEY` is set in the environment or in a `.env` file."
    )
    st.stop()


# --- 2. Build the query agent ONCE per process (survives reruns). -----------
@st.cache_resource
def get_query_agent():
    """Compiled query agent; cached so it is not rebuilt on every rerun."""
    from agentic_rag.agents.query import build_query_agent

    return build_query_agent(settings)


# --- 3. Per-session state: thread id + chat log. -----------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def reset_chat() -> None:
    """Start a fresh thread (old one stays in the in-memory checkpointer)."""
    st.session_state.thread_id = str(uuid4())
    st.session_state.messages = []


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
    out = [f"**Answer:**\n{answer.answer}", f"**Confidence:** {answer.confidence}"]
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

with st.sidebar:
    st.header("Session")
    if st.button("🧹 New chat"):
        reset_chat()

# --- 4. Chat history ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. Chat input ------------------------------------------------------------
if prompt := st.chat_input("Ask the wiki…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chips = st.container()
        answer_ph = st.empty()
        try:
            rendered = stream_turn(
                agent=get_query_agent(),
                message=prompt,
                thread_id=st.session_state.thread_id,
                recursion_limit=settings.recursion_limit,
                chips=chips,
                answer_ph=answer_ph,
            )
            # --- 7. record the rendered assistant reply in the chat log -----
            st.session_state.messages.append({"role": "assistant", "content": rendered})
        except Exception as exc:
            logger.exception("Query turn failed")
            st.error(f"Query failed: {exc}")
