"""Agent factory — builds LangChain agents with create_agent + middleware."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.token_tracker import TokenTracker
from agentic_rag.middleware.guardrails import path_guard_middleware
from agentic_rag.middleware.logging import (
    audit_logging_middleware,
    make_token_capture,
)



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

    # Create the token tracker and bind a capture middleware to THIS agent's
    # tracker (no module globals: several agents can share one process, e.g.
    # the Streamlit frontend, and each must record its own usage).
    tracker = TokenTracker(model_name)

    # Always include logging, guardrails, and token capture middleware
    all_middleware = [
        audit_logging_middleware,
        path_guard_middleware,
        make_token_capture(tracker),
    ] + (middleware or [])

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=all_middleware,
        checkpointer=MemorySaver(),
    )
    agent._token_tracker = tracker
    return agent
