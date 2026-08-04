"""Streamlit page: full wiki health check with HITL approve/reject.

Chat input plus a "Run full health check" button that sends the CLI-pinned
health-check prompt byte-for-byte (``cli.py``'s ``lint`` command): ``"Run a
full wiki health check. Report orphans, contradictions, missing links, and
data gaps."``. Free text goes through as-is. Everything turn-related —
streaming chips, the HITL decision widgets (Approve all / Reject all +
feedback for ``delete_wiki_page``), durable history — comes from the shared
``ui_common.run_turn("lint", …)`` shell; this page only wires the inputs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Streamlit puts the script's directory on sys.path, not the repo root. Make
# the repo root importable so `from frontend.… import …` works both under
# st.navigation and standalone AppTest.from_file (app_pages/ is 3 levels down
# from the repo root).
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.history_store import DEFAULT_ROOT, HistoryStore  # noqa: E402
from frontend.ui_common import (  # noqa: E402
    init_page,
    render_history,
    run_turn,
    sidebar,
)

logger = logging.getLogger(__name__)

AGENT = "lint"
AGENT_NAME = "Lint"

# Byte-for-byte the CLI's pinned health-check message (src/agentic_rag/cli.py
# `lint` command). Sent as-is when the "Run full health check" button is hit.
FULL_HEALTH_CHECK_MESSAGE = "Run a full wiki health check. Report orphans, contradictions, missing links, and data gaps."

# Durable transcript store (survives app restarts).
store = HistoryStore(DEFAULT_ROOT)


# --- Per-session state (per-agent namespace, shared across pages) -----------
tid, messages = init_page(AGENT)

# --- Sidebar: durable thread manager (selector + New chat + delete). ---------
sidebar(AGENT, store)

# --- Layout ------------------------------------------------------------------
st.title("Lint")
st.caption("Run the full wiki health check with approve/reject human review")

# --- One-click pinned health check (CLI-identical message). ------------------
run_clicked = st.button("Run full health check", key="lint_run")

# --- Chat history (current thread's durable transcript) ----------------------
render_history(messages)

# --- Turn shell: resume a pending HITL decision first, then fresh submits. ---
if st.session_state.get("lint_pending") is not None:
    # A decision awaits: the shared shell resumes (or re-offers the widgets
    # for a stale marker). The message arg is unused on the resume path.
    run_turn(AGENT, AGENT_NAME, "", store)
elif run_clicked:
    run_turn(AGENT, AGENT_NAME, FULL_HEALTH_CHECK_MESSAGE, store)
elif prompt := st.chat_input("Ask the lint agent…"):
    run_turn(AGENT, AGENT_NAME, prompt, store)
