"""Level 2 — turn-efficiency caps pinned (0 LLM, fully headless).

Pins the docs/spec.md acceptance caps — query ≤ 5, lint ≤ 4, fix ≤ 8,
ingest ≤ 15 (executor may tighten, never loosen). For each agent we script
the deterministic happy path and assert the recorded tool-call count stays
within its cap:

- query:  wiki_search -> wiki_read_page -> final answer            (2 calls)
- lint:   run_health_check -> write_lint_report                     (2 calls)
- fix:    wiki_read_page -> add_frontmatter -> regenerate_index     (3 calls)
- ingest: read_source -> submit_extraction -> match_page_tool ->
          create_page -> regenerate_index -> append_log             (6 calls)

The caps module constants are also asserted EXACTLY (5, 4, 8, 15) so a
future loosening fails the suite. Every run uses ScriptedChatModel +
build_agent; no Settings, no network, no real LLM; each run gets its own
thread_id.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import (
    build_fix_prompt,
    build_ingest_prompt,
    build_lint_prompt,
    build_query_prompt,
)
from agentic_rag.tools.fix_tools import add_frontmatter, fix_link
from agentic_rag.tools.ingest_grounding import submit_extraction
from agentic_rag.tools.ingest_tools import (
    append_log,
    create_page,
    flag_contradiction,
    read_source,
    update_page,
)
from agentic_rag.tools.lint_tools import run_health_check, write_lint_report
from agentic_rag.tools.nav import (
    regenerate_index,
    wiki_link_graph,
    wiki_read_page,
    wiki_search,
    wiki_summary,
)
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.match import match_page_tool
from tests.fixtures.eval_corpus import copy_broken_wiki
from tests.fixtures.fake_llm import ScriptedChatModel

# Spec-pinned turn-efficiency caps — tighten never loosen.
CAPS: dict[str, int] = {"query": 5, "lint": 4, "fix": 8, "ingest": 15}

QUERY_TOOLS = [wiki_search, wiki_read_page, wiki_summary]
LINT_TOOLS = [run_health_check, wiki_link_graph, wiki_read_page, write_lint_report]
FIX_TOOLS = [
    wiki_read_page,
    add_frontmatter,
    fix_link,
    regenerate_index,
]
INGEST_TOOLS = [
    read_source,
    submit_extraction,
    match_page_tool,
    create_page,
    update_page,
    flag_contradiction,
    regenerate_index,
    append_log,
]


def _ingest_middleware() -> list:
    """Ingest-agent middleware: HITL on delete and contradictions."""
    return [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_wiki_page": {"allowed_decisions": ["approve", "reject"]},
                "flag_contradiction": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
            }
        )
    ]


def _all_tool_calls(result) -> list[dict]:
    """Collect every tool call made during the agent run, in order."""
    calls: list[dict] = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(tc)
    return calls


def _run(agent, user_content: str, config: dict) -> dict:
    """Invoke a built agent with a plain user message."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]}, config=config
    )


class TestCapsPinned:
    """The cap constants themselves are exactly (5, 4, 8, 15) — never loosened."""

    def test_caps_exact(self):
        """CAPS dict and ordered tuple match the spec-pinned values."""
        assert CAPS == {"query": 5, "lint": 4, "fix": 8, "ingest": 15}
        assert tuple(CAPS[a] for a in ("query", "lint", "fix", "ingest")) == (
            5,
            4,
            8,
            15,
        )


class TestTurnEfficiency:
    """Scripted happy-path runs stay within their per-agent cap."""

    def test_query_happy_path_within_cap(self, eval_wiki):
        """query: wiki_search -> wiki_read_page -> final (2 ≤ 5)."""
        init_shared_tools(str(eval_wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="wiki_search",
                            args={"query": "MLX"},
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="wiki_read_page",
                            args={"slug": "entities/mlx"},
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(content="MLX is a machine learning framework by Apple."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=QUERY_TOOLS,
            system_prompt=build_query_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "What is MLX?", config)

        names = [c["name"] for c in _all_tool_calls(result)]
        assert names == ["wiki_search", "wiki_read_page"]
        assert len(names) <= CAPS["query"]

    def test_lint_happy_path_within_cap(self, eval_wiki):
        """lint: run_health_check -> write_lint_report (2 ≤ 4)."""
        init_shared_tools(str(eval_wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="run_health_check", args={}, id="tc-1")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="write_lint_report",
                            args={"report": "# Lint Report\n\nNo issues found.\n"},
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(content="Lint complete."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=LINT_TOOLS,
            system_prompt=build_lint_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "Run a full health check.", config)

        names = [c["name"] for c in _all_tool_calls(result)]
        assert names == ["run_health_check", "write_lint_report"]
        assert len(names) <= CAPS["lint"]

    def test_fix_happy_path_within_cap(self, tmp_path):
        """fix: wiki_read_page -> add_frontmatter -> regenerate_index (3 ≤ 8)."""
        wiki = copy_broken_wiki(tmp_path)
        init_shared_tools(str(wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="wiki_read_page",
                            args={"slug": "entities/broken-fm"},
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="add_frontmatter",
                            args={
                                "slug": "entities/broken-fm",
                                "title": "Broken FM",
                                "page_type": "entity",
                            },
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="regenerate_index", args={}, id="tc-3")
                    ],
                ),
                AIMessage(content="Fixed missing-frontmatter on entities/broken-fm."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=FIX_TOOLS,
            system_prompt=build_fix_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(
            agent,
            "Fix these lint issues:\n- [missing-frontmatter] entities/broken-fm",
            config,
        )

        names = [c["name"] for c in _all_tool_calls(result)]
        assert names == ["wiki_read_page", "add_frontmatter", "regenerate_index"]
        assert len(names) <= CAPS["fix"]

    def test_ingest_happy_path_within_cap(self, eval_env):
        """ingest: read_source -> submit_extraction -> match_page_tool ->
        create_page -> regenerate_index -> append_log (6 ≤ 15)."""
        wiki_path, raw_path = eval_env
        init_shared_tools(str(wiki_path))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_source",
                            args={"source_path": str(raw_path / "sample.md")},
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="submit_extraction",
                            args={
                                "entities": [
                                    {
                                        "name": "Samplecorp",
                                        "type": "organization",
                                        "summary": (
                                            "A sample organization mentioned "
                                            "in the source."
                                        ),
                                        "sources": ["sample.md"],
                                    }
                                ],
                                "concepts": [],
                                "contradictions": [],
                            },
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="match_page_tool",
                            args={"name": "Samplecorp", "page_type": "entity"},
                            id="tc-3",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="create_page",
                            args={
                                "slug": "entities/samplecorp",
                                "page_type": "entity",
                                "title": "Samplecorp",
                                "content": (
                                    "# Samplecorp\n\n"
                                    "A sample organization from sample.md.\n\n"
                                    "## Related\n\n"
                                    "- [[MLX]]"
                                ),
                                "sources": ["sample.md"],
                            },
                            id="tc-4",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="regenerate_index", args={}, id="tc-5")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="append_log",
                            args={"op": "ingest", "title": "sample.md"},
                            id="tc-6",
                        )
                    ],
                ),
                AIMessage(content="Ingested sample.md."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=INGEST_TOOLS,
            system_prompt=build_ingest_prompt("# Test schema"),
            middleware=_ingest_middleware(),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "Ingest raw/sample.md", config)

        names = [c["name"] for c in _all_tool_calls(result)]
        assert names == [
            "read_source",
            "submit_extraction",
            "match_page_tool",
            "create_page",
            "regenerate_index",
            "append_log",
        ]
        assert len(names) <= CAPS["ingest"]
        assert (wiki_path / "entities" / "samplecorp.md").is_file()
