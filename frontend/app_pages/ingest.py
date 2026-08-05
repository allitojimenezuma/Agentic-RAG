"""Streamlit page: raw-source ingest with full HITL (approve/reject/edit).

Natural-language chat input (passed through as-is) plus a raw-source picker:
``st.selectbox("Source in raw/", …)`` lists paths of files under
``settings.raw_sources_path`` (recursive) relative to that dir; "Run" submits
``Ingest <raw-path>/<relpath>`` (e.g. ``Ingest raw/foo.pdf``) — exactly the
message the CLI's ``ingest`` command builds when the user types the file path
from the repo root, which is how ``read_source`` resolves paths. The picker
reads ONLY: the UI never writes to ``raw/`` (users drop raw documents there
themselves). Everything turn-related — streaming chips, the HITL decision
widgets (Approve all / Reject all + feedback / Edit resolution for
flag_contradiction), durable history — comes from the shared
``ui_common.run_turn("ingest", …)`` shell; this page only wires the inputs.
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
from frontend.history_store import DEFAULT_ROOT, HistoryStore  # noqa: E402
from frontend.ui_common import (  # noqa: E402
    init_page,
    render_history,
    run_turn,
    sidebar,
)

logger = logging.getLogger(__name__)

AGENT = "ingest"
AGENT_NAME = "Ingest"

# Durable transcript store (survives app restarts).
store = HistoryStore(DEFAULT_ROOT)


def list_raw_sources(raw_path: Path) -> list[str]:
    """Relative POSIX paths of files under ``raw_path``, recursive, sorted.

    Reads only — the picker never writes to ``raw/``. Missing dir -> [].
    """
    if not raw_path.is_dir():
        return []
    return sorted(
        p.relative_to(raw_path).as_posix()
        for p in raw_path.rglob("*")
        if p.is_file()
    )


# --- Per-session state (per-agent namespace, shared across pages) -----------
tid, messages = init_page(AGENT)

# --- Sidebar: durable thread manager (selector + New chat + delete). ---------
sidebar(AGENT, store)

# --- Layout ------------------------------------------------------------------
st.title("Ingest")
st.caption("Chat with the ingest agent or pick a raw/ source")

# --- Raw-source picker (reads raw/ ONLY; users upload files there themselves).
run_clicked = False
raw_path = _builders.get_settings().raw_sources_path
raw_sources = list_raw_sources(raw_path)
if raw_sources:
    with st.container(border=True):
        selected = st.selectbox("Source in raw/", raw_sources)
        run_clicked = st.button("Run", key="ingest_run")
else:
    st.caption("No files in raw/ — drop source documents there first.")

# --- Chat history (current thread's durable transcript) ----------------------
render_history(messages)

# --- Turn shell: resume a pending HITL decision first, then fresh submits. ---
if st.session_state.get("ingest_pending") is not None:
    # A decision awaits: the shared shell resumes (or re-offers the widgets
    # for a stale marker). The message arg is unused on the resume path.
    run_turn(AGENT, AGENT_NAME, "", store)
elif run_clicked:
    # The picker yields paths relative to raw/, but read_source resolves against
    # the process CWD (repo root). Submit `Ingest raw/<relpath>` like the CLI.
    run_turn(AGENT, AGENT_NAME, f"Ingest {(raw_path / selected).as_posix()}", store)
elif prompt := st.chat_input("Ask the ingest agent…"):
    run_turn(AGENT, AGENT_NAME, prompt, store)
