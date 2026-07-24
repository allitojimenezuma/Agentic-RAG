"""Audit logging middleware — logs every tool call and result."""

from __future__ import annotations

from langchain.agents.middleware import wrap_tool_call


@wrap_tool_call
def audit_logging_middleware(request, handler):
    """Log every tool call with args to stdout, then log the result."""
    tool_name = request.tool_call["name"]
    args = request.tool_call.get("args", {})
    print(f"[TOOL CALL] {tool_name}({args})")
    result = handler(request)
    result_str = str(result)[:200]
    print(f"[TOOL RESULT] {tool_name} -> {result_str}")
    return result
