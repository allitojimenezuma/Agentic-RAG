"""Streamlit page: fix wiki lint issues with quick-action pills + HITL.

Chat input plus ``st.pills`` quick-actions (latest / missing-frontmatter /
broken-link / missing-related / missing-index) mirroring the CLI ``fix``
command's ``issue`` argument. Whatever is submitted — a pill selection or free
text — becomes the fix request exactly as the CLI shapes it: the page pre-runs
``health_check(settings.wiki_path)`` and builds the message via
:func:`frontend.agent_driver.build_fix_message` ("Fix these lint issues:…";
"No issues" when the wiki is clean; otherwise the user's words pass through
with the health report as context, exactly like cli.py's ``fix`` command).
If health_check is slow, that's acceptable — the page renders the
result message. Everything turn-related — streaming chips, the HITL decision
widgets (Approve all / Reject all + feedback for ``delete_wiki_page``), durable
history — comes from the shared ``ui_common.run_turn("fix", …)`` shell.
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

from frontend import builders as _builders  # noqa: E402  (needs repo root on sys.path)
from frontend.agent_driver import build_fix_message  # noqa: E402
from frontend.history_store import DEFAULT_ROOT, HistoryStore  # noqa: E402
from frontend.ui_common import (  # noqa: E402
    init_page,
    render_history,
    run_turn,
    sidebar,
)

logger = logging.getLogger(__name__)

AGENT = "fix"
AGENT_NAME = "Fix"

# Quick-action pills mirror the CLI `fix` command's `issue` argument; the
# selected value becomes the fix request via build_fix_message.
QUICK_ACTIONS = [
    "latest",
    "missing-frontmatter",
    "broken-link",
    "missing-related",
    "missing-index",
]

# Durable transcript store (survives app restarts).
store = HistoryStore(DEFAULT_ROOT)

# A pill-triggered turn has started; clear the pill selection on this rerun so
# the same quick action can run again. Must run BEFORE st.pills instantiates —
# writing a widget key after instantiation raises StreamlitAPIException.
if st.session_state.get("fix_pill_reset"):
    st.session_state["fix_pills"] = None
    st.session_state["fix_pill_reset"] = False

# --- Per-session state (per-agent namespace, shared across pages) -----------
tid, messages = init_page(AGENT)

# --- Sidebar: durable thread manager (selector + New chat + delete). ---------
sidebar(AGENT, store)

# --- Layout ------------------------------------------------------------------
st.title("Fix")
st.caption("Fix wiki lint issues — pick a quick action or ask in your own words")

# --- Quick-action pills (fill the request; the value becomes the fix issue).
selection = st.pills("Quick actions", QUICK_ACTIONS, key="fix_pills")

# --- Chat history (current thread's durable transcript) ----------------------
render_history(messages)

# --- Turn shell: resume a pending HITL decision first, then fresh submits. ---
if st.session_state.get("fix_pending") is not None:
    # A decision awaits: the shared shell resumes (or re-offers the widgets
    # for a stale marker). The message arg is unused on the resume path.
    run_turn(AGENT, AGENT_NAME, "", store)
elif selection:
    # Pills: the selected value becomes the fix request (CLI `fix` shaping).
    st.session_state["fix_pill_reset"] = True
    wiki_path = _builders.get_settings().wiki_path
    run_turn(AGENT, AGENT_NAME, build_fix_message(selection, wiki_path), store)
elif prompt := st.chat_input("Ask the fix agent…"):
    # Free text: shaped exactly like a CLI `fix` argument would be.
    wiki_path = _builders.get_settings().wiki_path
    run_turn(AGENT, AGENT_NAME, build_fix_message(prompt, wiki_path), store)
