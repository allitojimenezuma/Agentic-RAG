"""Streamlit entry point: multi-page navigation shell for the four agents.

Owns only the repo-root bootstrap, the Settings guard, and the page list.
All agent/chat logic lives in the pages under ``frontend/app_pages/``. Page
paths are relative to the main script dir (``frontend/``); a page joins the
navigation as its file lands (query.py now, ingest/lint/fix in later tasks),
so missing page files never break the entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit puts the script's directory (frontend/) on sys.path, not the repo
# root. Make the repo root importable so `from frontend.… import …` works
# exactly as in tests/unit/test_chat_driver.py.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# --- 1. Settings: surface config errors in the UI, never crash. -------------
try:
    from agentic_rag.config import Settings

    Settings()
except Exception as exc:
    st.error(
        f"Could not load configuration: {exc}\n\n"
        "Make sure `OPENAI_API_KEY` is set in the environment or in a `.env` file."
    )
    st.stop()

st.set_page_config(page_title="Agentic RAG", layout="wide")

# --- 2. Page list -----------------------------------------------------------
# Build the nav from the page files that actually exist, so T4 renders with
# query.py alone and T5/T6 pages join automatically (st.Page raises if the
# referenced file is missing).
_PAGES = [
    ("app_pages/query.py", "Wiki Q&A", ":material/forum:", True),
    ("app_pages/ingest.py", "Ingest", ":material/upload_file:", False),
    ("app_pages/lint.py", "Lint", ":material/health_and_safety:", False),
    ("app_pages/fix.py", "Fix", ":material/construction:", False),
]

_FRONTEND = Path(__file__).resolve().parent
pages = [
    st.Page(path, title=title, icon=icon, default=default)
    for path, title, icon, default in _PAGES
    if (_FRONTEND / path).is_file()
]

st.navigation(pages, position="top").run()
