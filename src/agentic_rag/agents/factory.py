"""Agent factory — builds LangChain agents with create_agent + middleware."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.token_tracker import TokenTracker
from agentic_rag.middleware.logging import set_tracker, audit_logging_middleware, token_capture_middleware



def build_agent(
    model: Any,
    tools: list,
    system_prompt: str,
    middleware: list | None = None,
    model_name: str = "unknown",
) -> create_agent:
    """Build a LangChain agent with create_agent.

    Args:
        model: Model string (e.g. 'openai:gpt-4.1-mini') or model instance
               (e.g. ChatOpenAI). Prefer passing an instance when using
               custom base_url/api_key.
        tools: List of tool instances for the agent.
        system_prompt: System prompt to inject.
        middleware: Optional middleware list (e.g. HumanInTheLoopMiddleware).
        model_name: Name of the model for token pricing lookup.

    Returns:
        A compiled agent runnable with _token_tracker attribute.
    """

    # Create and attach token tracker
    tracker = TokenTracker(model_name)
    set_tracker(tracker)

    # Always include logging and token capture middleware
    all_middleware = [audit_logging_middleware, token_capture_middleware] + (middleware or [])

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=all_middleware,
        checkpointer=MemorySaver(),
    )
    agent._token_tracker = tracker
    return agent
