"""Audit logging middleware — logs tool calls and captures LLM token usage."""

from __future__ import annotations

import logging
import time
from typing import Optional

from langchain.agents.middleware import wrap_tool_call, after_model

from agentic_rag.token_tracker import TokenTracker

logger = logging.getLogger("agentic_rag.tools")

# Global tracker per agent (set when agent is created)
_current_tracker: Optional[TokenTracker] = None


def set_tracker(tracker: TokenTracker) -> None:
    """Set the current token tracker for the session."""
    global _current_tracker
    _current_tracker = tracker


def get_tracker() -> Optional[TokenTracker]:
    """Get the current token tracker."""
    return _current_tracker


@wrap_tool_call
def audit_logging_middleware(request, handler):
    """Log every tool call with args, result, and duration."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    logger.info(f"TOOL CALL: {tool_name}({tool_args})")
    start_time = time.time()

    try:
        result = handler(request)
        duration = time.time() - start_time
        logger.info(f"TOOL RESULT: {tool_name} completed in {duration:.3f}s")
        logger.debug(f"TOOL OUTPUT: {tool_name} -> {str(result)[:500]}")
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"TOOL ERROR: {tool_name} failed after {duration:.3f}s: {e}")
        raise


@after_model
def token_capture_middleware(state, runtime):
    """Capture token usage from LLM responses after each model call.

    This hook runs after every LLM call and extracts token usage from the
    last AI message's response_metadata.
    """
    if not _current_tracker:
        return

    messages = state.get("messages", [])
    if not messages:
        return

    last_msg = messages[-1]
    if not hasattr(last_msg, "response_metadata"):
        return

    usage = last_msg.response_metadata.get("token_usage", {})
    if not usage:
        # Try alternatives: some providers use different keys
        usage = last_msg.response_metadata.get("usage", {})

    if usage:
        input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        if input_tokens or output_tokens:
            _current_tracker.record_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration=0.0,  # duration tracked per-call is not meaningful here
            )
