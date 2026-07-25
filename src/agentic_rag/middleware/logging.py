"""Audit logging middleware — logs every tool call, result, duration, and token usage."""

from __future__ import annotations

import logging
import time
from typing import Optional

from langchain.agents.middleware import wrap_tool_call

from agentic_rag.token_tracker import TokenTracker

logger = logging.getLogger("agentic_rag.tools")

# Global tracker per agent (set when agent is created)
_current_tracker: Optional[TokenTracker] = None


def set_tracker(tracker: TokenTracker) -> None:
    """Set the current token tracker for the session.

    Args:
        tracker: TokenTracker instance to use for tracking.
    """
    global _current_tracker
    _current_tracker = tracker


def get_tracker() -> Optional[TokenTracker]:
    """Get the current token tracker.

    Returns:
        Current TokenTracker instance or None.
    """
    return _current_tracker


@wrap_tool_call
def audit_logging_middleware(request, handler):
    """Log every tool call with args, result, duration, and token usage.

    This middleware intercepts tool calls and logs:
    - Tool name and arguments
    - Execution duration
    - Result summary
    - Token usage (if available from LLM response metadata)
    """
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

    logger.info(f"TOOL CALL: {tool_name}({tool_args})")
    start_time = time.time()

    try:
        result = handler(request)
        duration = time.time() - start_time

        # Extract token usage from result if present
        if _current_tracker and hasattr(result, "response_metadata"):
            usage = result.response_metadata.get("token_usage", {})
            if usage:
                _current_tracker.record_call(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    duration=duration,
                )

        logger.info(f"TOOL RESULT: {tool_name} completed in {duration:.3f}s")
        logger.debug(f"TOOL OUTPUT: {tool_name} -> {str(result)[:500]}")
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"TOOL ERROR: {tool_name} failed after {duration:.3f}s: {e}")
        raise
