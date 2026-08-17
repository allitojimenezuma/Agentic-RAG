"""Lint agent — wiki health audit with report output."""

from __future__ import annotations

import logging

from langchain.agents.middleware import HumanInTheLoopMiddleware

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.llm import get_model
from agentic_rag.agents.prompts import build_lint_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.lint_tools import write_lint_report
from agentic_rag.tools.nav import wiki_command
from agentic_rag.tools.shared import get_index_summary, init_shared_tools



def build_lint_agent(settings) -> object:
    """Build the lint agent with HITL on delete_wiki_page.

    The current wiki index is injected into the system prompt so the model
    has full page inventory before running the deterministic health check.
    Navigation/health go through the single read-only ``wiki_command`` tool.

    Args:
        settings: Settings instance with openai_model, agents_md_path.
    """
    init_shared_tools(settings.wiki_path)
    agents_md = load_agents_md(settings.agents_md_path)
    wiki_index = get_index_summary(settings.wiki_path)
    tools = [wiki_command, write_lint_report]
    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_wiki_page": {
                    "allowed_decisions": ["approve", "reject"]
                },
            }
        )
    ]
    return build_agent(
        model=get_model(settings),
        tools=tools,
        system_prompt=build_lint_prompt(agents_md, wiki_index=wiki_index),
        middleware=middleware,
        model_name=settings.openai_model,
    )
