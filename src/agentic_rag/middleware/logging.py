"""Audit logging middleware — logs every tool call, result, and duration."""

from __future__ import annotations

import logging
import time
from langchain.agents.middleware import wrap_tool_call

logger = logging.getLogger("agentic_rag.tools")


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
