"""Fix agent — executes commands within wiki_path with human approval."""

from __future__ import annotations

import logging

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import ToolCallRequest

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.model import get_model
from agentic_rag.agents.prompts import build_fix_prompt
from agentic_rag.schemas.agents_md import load_agents_md
from agentic_rag.tools.fix_tools import edit_wiki_page, execute_command, remove_index_entry
from agentic_rag.tools.shared import init_shared_tools, read_index, read_wiki_page

logger = logging.getLogger(__name__)


# Tools that never need approval (safe by design)
_AUTO_APPROVE_TOOLS = {"edit_wiki_page", "remove_index_entry", "read_index", "read_wiki_page"}

# Shell commands that are always safe
_READ_ONLY_PREFIXES = (
    "ls", "head", "tail", "wc", "grep", "find",
    "file", "stat", "tree", "diff",
)


def _needs_approval(request: ToolCallRequest) -> bool:
    """Only interrupt for execute_command write ops. Safe tools auto-approve."""
    tool_name = request.tool_call["name"]
    if tool_name in _AUTO_APPROVE_TOOLS:
        logger.info("Auto-approved (safe tool): %s", tool_name)
        return False
    cmd = request.tool_call["args"].get("command", "")
    stripped = cmd.strip()
    for skip in ("time ", "env ", "nohup "):
        if stripped.startswith(skip):
            stripped = stripped[len(skip):]
    for prefix in _READ_ONLY_PREFIXES:
        if stripped == prefix or stripped.startswith(prefix + " ") or stripped.startswith(prefix + "\t"):
            logger.info("Auto-approved (read-only): %s", cmd)
            return False
    return True


def build_fix_agent(settings) -> object:
    """Build the fix agent with HITL on write commands, auto-approve read-only.

    Args:
        settings: Settings instance with openai_model, agents_md_path, wiki_path.
    """
    init_shared_tools(settings.wiki_path)
    agents_md = load_agents_md(settings.agents_md_path)

    tools = [
        execute_command,
        read_index,
        read_wiki_page,
        edit_wiki_page,
        remove_index_entry,
    ]

    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "execute_command": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "when": _needs_approval,
                },
            }
        ),
    ]

    return build_agent(
        model=get_model(settings),
        tools=tools,
        system_prompt=build_fix_prompt(agents_md),
        middleware=middleware,
        model_name=settings.openai_model,
    )
