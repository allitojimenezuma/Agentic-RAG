"""Fix agent — resolves wiki lint issues with HITL on page deletion."""

from __future__ import annotations

import logging

from langchain.agents.middleware import HumanInTheLoopMiddleware

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.llm import get_model
from agentic_rag.agents.prompts import build_fix_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.fix_tools import (
    add_frontmatter,
    append_related_section,
    edit_wiki_page,
    fix_link,
)
from agentic_rag.tools.ingest_tools import delete_wiki_page
from agentic_rag.tools.nav import regenerate_index, wiki_read_page, wiki_link_graph
from agentic_rag.tools.shared import get_index_summary, init_shared_tools

logger = logging.getLogger(__name__)


def build_fix_agent(settings) -> object:
    """Build the fix agent with HITL on delete_wiki_page, auto-approve the rest.

    The current wiki index is injected into the system prompt so the model
    already knows which pages exist without blind guessing.

    Args:
        settings: Settings instance with openai_model, agents_md_path, wiki_path.
    """
    init_shared_tools(settings.wiki_path)
    agents_md = load_agents_md(settings.agents_md_path)
    wiki_index = get_index_summary(settings.wiki_path)

    tools = [
        wiki_read_page,
        edit_wiki_page,
        add_frontmatter,
        fix_link,
        append_related_section,
        regenerate_index,
        delete_wiki_page,
        wiki_link_graph
    ]

    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_wiki_page": {"allowed_decisions": ["approve", "reject"]},
            }
        ),
    ]

    return build_agent(
        model=get_model(settings, cache_key="wiki-fix"),
        tools=tools,
        system_prompt=build_fix_prompt(agents_md, wiki_index=wiki_index),
        middleware=middleware,
        model_name=settings.openai_model,
    )
