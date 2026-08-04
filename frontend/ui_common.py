"""Shared Streamlit shell for the four agent pages.

Session-state init (one namespace per agent), the sidebar thread manager over
:class:`~frontend.history_store.HistoryStore`, the HITL action renderer +
decision widgets, and :func:`run_turn` — the single turn/HITL-rerun loop shared
by every page (query uses ``chat_driver.stream_query`` via its own page).

Streamlit-dependent by design (verified via AppTest, not unit tests). The pure
helpers (:func:`render_final`, :func:`action_summary`, :func:`make_pending`)
stay unit-testable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator
from uuid import uuid4

import streamlit as st

from frontend import agents as _agents
from frontend.agent_driver import (
    build_decisions,
    FinalMessage,
    InterruptEvent,
    resume_turn,
    stream_turn,
)
from frontend.chat_driver import AnswerToken, ToolEnd, ToolStart
from frontend.history_store import HistoryStore

logger = logging.getLogger(__name__)

# Agent page key -> display name (pages import these for titles/captions).
AGENTS: dict[str, str] = {
    "query": "Wiki Q&A",
    "ingest": "Ingest",
    "lint": "Lint",
    "fix": "Fix",
}

# Material icon per agent (native Streamlit emoji shortcodes).
AGENT_ICONS: dict[str, str] = {
    "query": "forum",
    "ingest": "upload_file",
    "lint": "health_and_safety",
    "fix": "construction",
}

# Persisted in the durable transcript when an interrupt pauses before any
# answer text streamed (mirrors agent_driver's synthetic ToolEnd output).
PAUSED_OUTPUT = "⏸ awaiting human approval"


def init_page(agent: str) -> tuple[str, list]:
    """Ensure the per-agent session namespace; return (thread_id, messages).

    Keys are ``f"{agent}_thread_id"`` (str uuid) and ``f"{agent}_messages"``
    (list of ``{"role", "content"}``), one namespace per agent so pages never
    collide. The ``f"{agent}_pending"`` marker is created by :func:`run_turn`.
    """
    if f"{agent}_thread_id" not in st.session_state:
        st.session_state[f"{agent}_thread_id"] = str(uuid4())
    if f"{agent}_messages" not in st.session_state:
        st.session_state[f"{agent}_messages"] = []
    return st.session_state[f"{agent}_thread_id"], st.session_state[f"{agent}_messages"]


def render_history(messages: list) -> None:
    """Render past chat bubbles from the session transcript."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def sidebar(agent: str, store: HistoryStore) -> None:
    """Thread selector (``store.list_threads``) + New chat + delete-selected.

    Selecting a saved thread swaps ``thread_id``/``messages`` in session state
    and drops any pending HITL marker (a decision belongs to the prior thread).
    """
    with st.sidebar:
        st.header(f":material/{AGENT_ICONS[agent]}: {AGENTS[agent]}")
        current = st.session_state[f"{agent}_thread_id"]
        threads = store.list_threads(agent)
        if threads:
            options = [current] + [t for t in threads if t != current]
            chosen = st.selectbox(
                "Thread",
                options,
                format_func=lambda t: t if t != current else f"{t} (current)",
                key=f"{agent}_thread_select",
            )
            if chosen != current:
                st.session_state[f"{agent}_thread_id"] = chosen
                st.session_state[f"{agent}_messages"] = store.load(agent, chosen)
                st.session_state.pop(f"{agent}_pending", None)
                st.rerun()
        else:
            st.caption("No saved threads yet.")
        if st.button("🧹 New chat", key=f"{agent}_new_chat"):
            st.session_state[f"{agent}_thread_id"] = str(uuid4())
            st.session_state[f"{agent}_messages"] = []
            st.session_state.pop(f"{agent}_pending", None)
            st.rerun()
        if threads:
            if st.button("Delete selected", key=f"{agent}_delete_thread"):
                store.delete(agent, chosen)
                st.session_state.pop(f"{agent}_pending", None)
                st.rerun()


def action_summary(action: dict) -> str:
    """CLI-mirroring one-line action summary: ``name(slug[, rest...])``.

    Matches cli.py's pending-action display: the ``page_slug``/``slug`` arg
    first, then the remaining args (``key=value``), truncated to 120 chars.
    Also used for live tool-chip labels.
    """
    if not isinstance(action, dict):
        return "unknown()"
    name = action.get("name") or "unknown"
    args = action.get("args", {})
    if not isinstance(args, dict):
        args = {}
    desc = args.get("page_slug", args.get("slug", ""))
    rest = [
        f"{key}={value!r}"
        for key, value in args.items()
        if key not in ("page_slug", "slug") and value not in ("", None)
    ]
    parts = ([str(desc)] if desc else []) + rest
    out = f"{name}({', '.join(parts)})"
    if len(out) > 120:
        out = out[:117] + "..."
    return out


def render_actions(actions: list[dict]) -> None:
    """One bordered card per pending action, mirroring cli.py's display."""
    for action in actions:
        with st.container(border=True):
            st.markdown(f"**{action_summary(action)}**")


def make_pending(actions: list[dict], turn_text: str) -> dict:
    """The pending marker stored under ``st.session_state[f"{agent}_pending"]``.

    ``decisions`` is absent until a decision-widget button writes it;
    :func:`run_turn`'s resume path is gated on its presence — exactly one
    resume per decision.
    """
    return {"actions": actions, "turn_text": turn_text}


def render_final(agent_name: str, final: FinalMessage) -> str:
    """Markdown for the terminal assistant bubble; ingest/lint/fix just show
    the text (query renders its structured answer in its own page)."""
    return final.text


def _decision_widgets(agent: str, actions: list[dict]) -> None:
    """Approve all / Reject all (+feedback) / Edit resolution (flag_contradiction only).

    Each button writes the built decision list into the pending marker and
    calls ``st.rerun()``; the next script run's :func:`run_turn` consumes it
    exactly once.
    """
    pending_key = f"{agent}_pending"
    st.markdown("### Human decision required")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Approve all", key=f"{agent}_approve_all"):
            st.session_state[pending_key]["decisions"] = build_decisions("approve", actions)
            st.rerun()
    with col_b:
        feedback = st.text_input("Feedback (optional)", key=f"{agent}_reject_feedback")
        if st.button("❌ Reject all", key=f"{agent}_reject_all"):
            st.session_state[pending_key]["decisions"] = build_decisions(
                "reject", actions, feedback=feedback
            )
            st.rerun()
    if actions and all(
        isinstance(a, dict) and a.get("name") == "flag_contradiction" for a in actions
    ):
        st.markdown("**Edit a contradiction**")
        resolution = st.text_input(
            "New proposed resolution", key=f"{agent}_edit_resolution"
        )
        index = 0
        if len(actions) > 1:
            index = st.selectbox(
                "Which action",
                range(len(actions)),
                format_func=lambda i: f"Action {i + 1}",
                key=f"{agent}_edit_index",
            )
        if st.button("✏️ Edit & approve", key=f"{agent}_edit"):
            st.session_state[pending_key]["decisions"] = build_decisions(
                "edit", actions, index=index, new_resolution=resolution
            )
            st.rerun()


def _get_agent(agent: str) -> object:
    """Compiled agent for a page key.

    Indirection through the ``frontend.agents`` module keeps AppTest stubs able
    to monkeypatch the builder (``agents.get_ingest_agent = lambda: fake``).
    """
    builders = {
        "query": _agents.get_query_agent,
        "ingest": _agents.get_ingest_agent,
        "lint": _agents.get_lint_agent,
        "fix": _agents.get_fix_agent,
    }
    return builders[agent]()


class _ChipTracker:
    """Live tool chips (mirrors app.py's streaming chip rendering)."""

    def __init__(self, container) -> None:
        self._container = container
        self._running: list[dict] = []

    def on_start(self, event: ToolStart) -> None:
        label = action_summary({"name": event.name, "args": event.args})
        ph = self._container.empty()
        ph.markdown(f"🔍 {label}…")
        self._running.append({"name": event.name, "label": label, "ph": ph, "done": False})

    def on_end(self, event: ToolEnd) -> None:
        for chip in reversed(self._running):
            if chip["name"] == event.name and not chip["done"]:
                chip["ph"].markdown(f"✅ {chip['label']}")
                chip["done"] = True
                break


def _sync_events(agen: AsyncGenerator[Any, None]):
    """Iterate an async event generator synchronously on a dedicated loop.

    Streamlit scripts have no event loop, so drive the async generator with a
    fresh loop (``st.*`` calls stay on the script thread). The generator is
    explicitly closed on exit so a mid-stream exception never leaves pending
    asyncio tasks.
    """
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                return
    finally:
        try:
            loop.run_until_complete(agen.aclose())
            # Drain tasks scheduled by the async-generator athrow machinery so
            # loop.close() never destroys a pending task (stderr noise).
            loop.run_until_complete(asyncio.sleep(0))
        except Exception:
            pass
        loop.close()


def _drive(
    agen,
    agent: str,
    agent_name: str,
    store: HistoryStore,
    chips: _ChipTracker,
    answer_ph,
    streamed: str,
) -> tuple[str | None, str]:
    """Consume one turn's events into chips + answer placeholder.

    Returns ``(final_text, streamed)``. On an :class:`InterruptEvent` the
    partial transcript is persisted, the pending marker is stored, and the
    actions + decision widgets are rendered — ``final_text`` is ``None`` and
    the caller must stop consuming events. On :class:`FinalMessage`,
    ``final_text`` is the rendered terminal text.
    """
    pending_key = f"{agent}_pending"
    tid = st.session_state[f"{agent}_thread_id"]
    for event in agen:
        if isinstance(event, ToolStart):
            chips.on_start(event)
        elif isinstance(event, ToolEnd):
            chips.on_end(event)
        elif isinstance(event, AnswerToken):
            streamed += event.text
            answer_ph.markdown(streamed)
        elif isinstance(event, InterruptEvent):
            store.append(agent, tid, "assistant", streamed or PAUSED_OUTPUT)
            st.session_state[pending_key] = make_pending(event.actions, streamed)
            render_actions(event.actions)
            _decision_widgets(agent, event.actions)
            return None, streamed
        elif isinstance(event, FinalMessage):
            text = render_final(agent_name, event)
            answer_ph.markdown(text)
            return text, streamed
    return None, streamed  # generator ended without a terminal event (defensive)


def run_turn(agent: str, agent_name: str, message: str, store: HistoryStore) -> None:
    """THE shared turn/HITL-rerun shell.

    1. Resume path — ``st.session_state[f"{agent}_pending"]`` holds a user
       decision: rebuild chips + answer from the pending partial text, drive
       :func:`~frontend.agent_driver.resume_turn` with the built decisions. A
       further :class:`InterruptEvent` re-renders actions + widgets and stores
       a new pending marker; a :class:`FinalMessage` renders, persists and
       clears the marker.
    2. Fresh submit — record + render the user message, stream the turn via
       :func:`~frontend.agent_driver.stream_turn`. On interrupt: persist the
       partial transcript, store the pending marker, render actions + decision
       widgets (Approve all / Reject all + feedback / Edit resolution for
       flag_contradiction only). Buttons write the decision into session_state
       and call ``st.rerun()``.

    Rerun protocol: a turn never re-invokes the agent after an interrupt; the
    pending marker gates exactly one resume per decision.
    """
    pending_key = f"{agent}_pending"
    tid = st.session_state[f"{agent}_thread_id"]
    messages = st.session_state[f"{agent}_messages"]
    pending = st.session_state.get(pending_key)
    config = _agents.agent_config(agent, tid)
    agent_obj = _get_agent(agent)

    if pending is not None:
        decisions = pending.get("decisions")
        if decisions is None:
            # Stale marker (e.g. a thread switch raced): re-offer the widgets.
            render_actions(pending.get("actions", []))
            _decision_widgets(agent, pending.get("actions", []))
            return
        with st.chat_message("assistant"):
            chips = st.container()
            answer_ph = st.empty()
            streamed = pending.get("turn_text", "")
            if streamed:
                answer_ph.markdown(streamed)
            try:
                final_text, _ = _drive(
                    _sync_events(
                        resume_turn(agent_obj, decisions, config, agent_name)
                    ),
                    agent,
                    agent_name,
                    store,
                    _ChipTracker(chips),
                    answer_ph,
                    streamed,
                )
            except Exception as exc:
                logger.exception("Resume failed agent=%s", agent)
                st.error(f"{agent_name} turn failed: {exc}")
                return
        if final_text is not None:
            messages.append({"role": "assistant", "content": final_text})
            store.append(agent, tid, "assistant", final_text)
            st.session_state.pop(pending_key, None)
        return

    # --- Fresh submit -----------------------------------------------------
    messages.append({"role": "user", "content": message})
    store.append(agent, tid, "user", message)
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        chips = st.container()
        answer_ph = st.empty()
        try:
            final_text, _ = _drive(
                _sync_events(stream_turn(agent_obj, message, config, agent_name)),
                agent,
                agent_name,
                store,
                _ChipTracker(chips),
                answer_ph,
                "",
            )
        except Exception as exc:
            logger.exception("Turn failed agent=%s", agent)
            st.error(f"{agent_name} turn failed: {exc}")
            return
    if final_text is not None:
        messages.append({"role": "assistant", "content": final_text})
        store.append(agent, tid, "assistant", final_text)
        st.session_state.pop(pending_key, None)
