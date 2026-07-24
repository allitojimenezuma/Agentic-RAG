"""Lint agent — wiki health audit with report output."""

from __future__ import annotations

from langchain.agents.middleware import HumanInTheLoopMiddleware

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_lint_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.lint_tools import (
    extract_concepts,
    find_inbound_links,
    read_all_pages,
    write_lint_report,
)
from agentic_rag.tools.shared import read_index


def build_lint_agent(settings) -> object:
    """Build the lint agent with HITL on delete_wiki_page.

    Args:
        settings: Settings instance with openai_model, agents_md_path.
    """
    agents_md = load_agents_md(settings.agents_md_path)
    tools = [
        read_all_pages,
        read_index,
        find_inbound_links,
        extract_concepts,
        write_lint_report,
    ]
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
        model=settings.openai_model,
        tools=tools,
        system_prompt=build_lint_prompt(agents_md),
        middleware=middleware,
    )
