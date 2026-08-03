"""Query agent — read-only wiki Q&A with grounded citations (cite-or-die)."""

from __future__ import annotations

import logging

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.model import get_model
from agentic_rag.agents.prompts import build_query_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.grounding import new_nav_capture, submit_query_answer
from agentic_rag.tools.nav import wiki_read_page, wiki_search, wiki_summary
from agentic_rag.tools.shared import init_shared_tools

logger = logging.getLogger("agentic_rag.agents.query")


def build_query_agent(settings) -> object:
    """Build the query agent (read-only, no HITL).

    Navigation uses the consolidated ``wiki_search`` / ``wiki_read_page`` /
    ``wiki_summary`` tools; finalization goes through the cite-or-die
    ``submit_query_answer`` tool. A fresh ``NavCapture`` is registered per
    invocation so every cited slug must be a page navigated this turn.

    Args:
        settings: Settings instance with openai_model, agents_md_path.
    """
    init_shared_tools(settings.wiki_path)
    agents_md = load_agents_md(settings.agents_md_path)
    tools = [wiki_search, wiki_read_page, wiki_summary, submit_query_answer]
    agent = build_agent(
        model=get_model(settings),
        tools=tools,
        system_prompt=build_query_prompt(agents_md),
        model_name=settings.openai_model,
    )
    # Fresh per-invocation capture, stashed for the CLI renderer (mirrors
    # agent._token_tracker). No response_format: finalization is a tool.
    agent._nav_capture = new_nav_capture()
    return agent
