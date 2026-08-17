"""Audit logging middleware — logs tool calls and captures LLM token usage.

Implements BOTH the sync ``wrap_tool_call`` and async ``awrap_tool_call`` hooks so
this middleware works for agents invoked synchronously (``invoke()``/``stream()``
in cli.py) and asynchronously (``astream()`` in the Streamlit frontend). A
sync-only implementation crashes async runs with ``NotImplementedError`` (and an
async-only one crashes sync runs with the same error in reverse).
"""

from __future__ import annotations

import logging
import time

from langchain.agents.middleware import AgentMiddleware, after_model

from agentic_rag.token_tracker import TokenTracker

logger = logging.getLogger("agentic_rag.tools")


class AuditLoggingMiddleware(AgentMiddleware):
    """Log every tool call with args, result, and duration (sync + async)."""

    def wrap_tool_call(self, request, handler):
        """Synchronous path (CLI: invoke()/stream())."""
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})

        logger.info(f"TOOL CALL: {tool_name}({tool_args})")
        start_time = time.time()

        try:
            result = handler(request)
            duration = time.time() - start_time
            logger.debug(f"TOOL OUTPUT: {tool_name} -> {str(result)[:500]}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"TOOL ERROR: {tool_name} failed after {duration:.3f}s: {e}")
            raise

    async def awrap_tool_call(self, request, handler):
        """Asynchronous path (Streamlit frontend: astream())."""
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})

        logger.info(f"TOOL CALL: {tool_name}({tool_args})")
        start_time = time.time()

        try:
            result = await handler(request)
            duration = time.time() - start_time
            logger.debug(f"TOOL OUTPUT: {tool_name} -> {str(result)[:500]}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"TOOL ERROR: {tool_name} failed after {duration:.3f}s: {e}")
            raise


audit_logging_middleware = AuditLoggingMiddleware()


def make_token_capture(tracker: TokenTracker) -> AgentMiddleware:
    """Build an after-model middleware that records LLM usage on ``tracker``.

    One instance per compiled agent: the tracker is bound at build time rather
    than read from a module global, so each agent records its own usage even
    when several agents share one process (e.g. the Streamlit frontend). The
    hook runs after every LLM call and extracts token usage from the last AI
    message's ``response_metadata``.
    """

    @after_model(name="token_capture")
    def _capture(state, runtime) -> None:
        messages = state.get("messages", [])
        if not messages:
            return

        last_msg = messages[-1]
        if not hasattr(last_msg, "response_metadata"):
            return

        usage = last_msg.response_metadata.get("token_usage", {}) or {}
        if not usage:
            # Some providers use a different key.
            usage = last_msg.response_metadata.get("usage", {}) or {}

        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
            if input_tokens or output_tokens:
                tracker.record_call(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=0.0,  # duration tracked per-call is not meaningful here
                )

    return _capture
