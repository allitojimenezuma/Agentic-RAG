"""Fix agent — executes commands within wiki_path with human approval."""

from __future__ import annotations

import logging

from langchain.agents.middleware import HumanInTheLoopMiddleware

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.model import get_model
from agentic_rag.agents.prompts import build_fix_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.fix_tools import edit_wiki_page
from agentic_rag.tools.shared import init_shared_tools, read_index, read_wiki_page

logger = logging.getLogger(__name__)


# Tools that never need approval (safe by design)
_AUTO_APPROVE_TOOLS = {"edit_wiki_page", "read_index", "read_wiki_page"}


def build_fix_agent(settings) -> object:
    """Build the fix agent with HITL on write commands, auto-approve read-only.

    Args:
        settings: Settings instance with openai_model, agents_md_path, wiki_path.
    """
    init_shared_tools(settings.wiki_path)
    agents_md = load_agents_md(settings.agents_md_path)

    tools = [
        read_index,
        read_wiki_page,
        edit_wiki_page,
    ]

    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={}
        ),
    ]

    return build_agent(
        model=get_model(settings),
        tools=tools,
        system_prompt=build_fix_prompt(agents_md),
        middleware=middleware,
        model_name=settings.openai_model,
    )
