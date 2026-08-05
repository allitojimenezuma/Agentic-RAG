"""Ingest agent — read sources, extract, integrate into wiki with HITL."""

from __future__ import annotations

import logging

from langchain.agents.middleware import HumanInTheLoopMiddleware

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.llm import get_model
from agentic_rag.agents.prompts import build_ingest_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.extraction import submit_extraction
from agentic_rag.tools.ingest_tools import (
    append_log,
    create_page,
    delete_wiki_page,
    flag_contradiction,
    read_source,
    update_page,
)
from agentic_rag.tools.nav import (
    regenerate_index,
    wiki_link_graph,
    wiki_read_page,
    wiki_scan,
)
from agentic_rag.tools.shared import get_index_summary, init_shared_tools
from agentic_rag.wiki.match import match_page_tool

logger = logging.getLogger("agentic_rag.agents.ingest")


def build_ingest_agent(settings) -> object:
    """Build the ingest agent with HITL on delete and contradictions.

    The current wiki index is injected into the system prompt so the model
    knows which pages already exist before planning creates/updates.

    Args:
        settings: Settings instance with openai_model, agents_md_path, wiki_path.
    """
    init_shared_tools(settings.wiki_path)
    agents_md = load_agents_md(settings.agents_md_path)
    wiki_index = get_index_summary(settings.wiki_path)
    tools = [
        read_source,
        submit_extraction,
        match_page_tool,
        wiki_read_page,
        wiki_scan,
        wiki_link_graph,
        create_page,
        update_page,
        flag_contradiction,
        regenerate_index,
        append_log,
        delete_wiki_page,
    ]
    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_wiki_page": {"allowed_decisions": ["approve", "reject"]},
                "flag_contradiction": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
            }
        )
    ]
    return build_agent(
        model=get_model(settings),
        tools=tools,
        system_prompt=build_ingest_prompt(agents_md, wiki_index=wiki_index),
        middleware=middleware,
        model_name=settings.openai_model,
    )
