"""Agent factory — builds LangChain agents with create_agent + middleware."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("agentic_rag.agents")


def build_agent(
    model: Any,
    tools: list,
    system_prompt: str,
    middleware: list | None = None,
) -> create_agent:
    """Build a LangChain agent with create_agent.

    Args:
        model: Model string (e.g. 'openai:gpt-4.1-mini') or model instance
               (e.g. ChatOpenAI). Prefer passing an instance when using
               custom base_url/api_key.
        tools: List of tool instances for the agent.
        system_prompt: System prompt to inject.
        middleware: Optional middleware list (e.g. HumanInTheLoopMiddleware).

    Returns:
        A compiled agent runnable.
    """
    logger.info(f"Building agent with {len(tools)} tools")
    logger.debug(f"Tools: {[t.name for t in tools]}")
    logger.debug(f"Middleware: {[m.__class__.__name__ for m in (middleware or [])]}")

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        checkpointer=MemorySaver(),
    )
