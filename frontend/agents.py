"""Cached agent builders + per-agent config factory (Streamlit-aware shell).

Builds each compiled agent ONCE per process via ``@st.cache_resource`` and
exposes :func:`agent_config`, the per-agent invocation config that EXACTLY
mirrors the CLI (``src/agentic_rag/cli.py``): query/ingest pin their
recursion limits, lint/fix omit the key entirely.

Streamlit-aware by design: the builders are only called from the UI pages and
AppTest stubs, never from the pure drivers. Unit tests exercise
:func:`agent_config` with ``get_settings`` monkeypatched.
"""

from __future__ import annotations

import streamlit as st

from agentic_rag.config import Settings


@st.cache_resource
def get_settings() -> Settings:
    """Application settings, built once per process.

    app.py's top-level try/except guard runs first and surfaces config errors
    in the UI; pages rely on this cache instead of constructing ``Settings``.
    """
    return Settings()


@st.cache_resource
def get_query_agent() -> object:
    """Compiled query agent, cached per process."""
    from agentic_rag.agents.query import build_query_agent

    return build_query_agent(get_settings())


@st.cache_resource
def get_ingest_agent() -> object:
    """Compiled ingest agent, cached per process."""
    from agentic_rag.agents.ingest import build_ingest_agent

    return build_ingest_agent(get_settings())


@st.cache_resource
def get_lint_agent() -> object:
    """Compiled lint agent, cached per process."""
    from agentic_rag.agents.lint import build_lint_agent

    return build_lint_agent(get_settings())


@st.cache_resource
def get_fix_agent() -> object:
    """Compiled fix agent, cached per process."""
    from agentic_rag.agents.fix import build_fix_agent

    return build_fix_agent(get_settings())


def agent_config(agent: str, thread_id: str) -> dict:
    """Invocation config for one agent, EXACTLY mirroring cli.py's shapes.

    - ``query``: ``{"configurable": {"thread_id": tid},
      "recursion_limit": settings.recursion_limit}``
    - ``ingest``: ``{"configurable": {"thread_id": tid},
      "recursion_limit": settings.ingest_recursion_limit}``
    - ``lint``/``fix``: ``{"configurable": {"thread_id": tid}}`` — the CLI
      omits ``recursion_limit`` for these, so the key must NOT appear here.
    """
    if agent == "query":
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": get_settings().recursion_limit,
        }
    if agent == "ingest":
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": get_settings().ingest_recursion_limit,
        }
    if agent in ("lint", "fix"):
        return {"configurable": {"thread_id": thread_id}}
    raise ValueError(
        f"Unknown agent: {agent!r} (expected query/ingest/lint/fix)"
    )
