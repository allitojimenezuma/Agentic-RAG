"""Agent factory — builds LangChain agents with create_agent + middleware."""

from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


def build_agent(
    model: str,
    tools: list,
    system_prompt: str,
    middleware: list | None = None,
) -> create_agent:
    """Build a LangChain agent with create_agent.

    Args:
        model: Model string (e.g. 'openai:gpt-4.1-mini') or model instance.
        tools: List of tool instances for the agent.
        system_prompt: System prompt to inject.
        middleware: Optional middleware list (e.g. HumanInTheLoopMiddleware).

    Returns:
        A compiled agent runnable.
    """
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        checkpointer=MemorySaver(),
    )
