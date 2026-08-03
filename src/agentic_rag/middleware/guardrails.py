"""Guardrails middleware — rejects writes outside wiki_path and touching raw/."""

from __future__ import annotations

from langchain.agents.middleware import wrap_tool_call


@wrap_tool_call
def path_guard_middleware(request, handler):
    """Reject any tool call that tries to write outside wiki_path or touch raw/."""
    tool_name = request.tool_call["name"]
    args = request.tool_call.get("args", {})

    # Hard block: any write-like tool must not receive raw/ paths
    write_tools = {
        "create_page",
        "update_page",
        "delete_wiki_page",
        "write_lint_report",
    }
    if tool_name in write_tools:
        for key in ("slug", "path", "file_path", "source_path"):
            val = str(args.get(key, ""))
            if "raw/" in val or val.startswith("/") or ".." in val:
                return f"ERROR: Path '{val}' is outside allowed wiki directory"

    # Block write-like tools that receive a raw/ source_path. Note: this must
    # NOT apply to read_source (a read tool whose source_path legitimately
    # points into raw/) — write tools are already fully covered by the
    # write_tools check above.
    if "source_path" in args and tool_name in write_tools:
        val = str(args["source_path"])
        if "raw/" in val and not val.startswith("./raw"):
            return f"ERROR: Cannot write to raw/ directory: {val}"

    return handler(request)
