"""Query agent — read-only wiki Q&A with citations."""

from __future__ import annotations

import logging

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.model import get_model
from agentic_rag.agents.prompts import build_query_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.query_tools import find_relevant_pages
from agentic_rag.tools.shared import read_index, read_wiki_page, search_index

logger = logging.getLogger("agentic_rag.agents.query")


def build_query_agent(settings) -> object:
    """Build the query agent (read-only, no HITL).

    Args:
        settings: Settings instance with openai_model, agents_md_path.
    """
    logger.info("Building query agent (read-only)")
    agents_md = load_agents_md(settings.agents_md_path)
    tools = [read_index, search_index, read_wiki_page, find_relevant_pages]
    logger.info("Query agent built with read-only tools")
    return build_agent(
        model=get_model(settings),
        tools=tools,
        system_prompt=build_query_prompt(agents_md),
        model_name=settings.openai_model,
    )
