"""Integration tests for the lint agent — fake LLM with scripted tool calls.

Exercises the CURRENT lint toolset (run_health_check / wiki_link_graph /
wiki_read_page / write_lint_report) end-to-end with the ``ScriptedChatModel``
harness: deterministic health check first, then the report writer.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.prompts import build_lint_prompt
from agentic_rag.tools.lint_tools import write_lint_report
from agentic_rag.tools.nav import wiki_command
from agentic_rag.tools.shared import init_shared_tools
from tests.fixtures.fake_llm import ScriptedChatModel


@pytest.fixture
def wiki_with_orphan(tmp_path: Path) -> Path:
    """Create a wiki with an orphan page (no inbound links)."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "entities").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "comparisons").mkdir()

    # Main page (not orphan)
    (wiki / "entities" / "python.md").write_text(
        "---\nslug: entities/python\ntype: entity\ntitle: Python\nsources:\n  - sample.md\nupdated: 2025-01-01\ntags: []\n---\n\n# Python\n\nPython is a programming language.\n\n## Related\n\n- [[Tool Calling]]\n"
    )

    # Orphan page (no inbound links from other pages)
    (wiki / "concepts" / "orphan-concept.md").write_text(
        "---\nslug: concepts/orphan-concept\ntype: concept\ntitle: Orphan Concept\nsources:\n  - sample.md\nupdated: 2025-01-01\ntags: []\n---\n\n# Orphan Concept\n\nThis concept is not linked from anywhere.\n"
    )

    # Index
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n\n## Comparisons\n"
    )

    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


CURRENT_LINT_TOOLS = [wiki_command, write_lint_report]


class TestLintAgentBasic:
    """Basic lint agent construction tests."""

    def test_agent_builds_with_fake_model(self, wiki_with_orphan):
        """Agent can be created with ScriptedChatModel."""
        model = ScriptedChatModel(
            responses=[AIMessage(content="Ready to lint.")]
        )
        agent = create_agent(
            model=model,
            tools=CURRENT_LINT_TOOLS,
            system_prompt="You are a lint agent.",
        )
        assert agent is not None


class TestLintFlow:
    """Test the full lint flow with scripted tool calls."""

    def test_lint_writes_report(self, wiki_with_orphan):
        """Lint flow: run_health_check -> write_lint_report."""
        wp = str(wiki_with_orphan)
        init_shared_tools(wp)

        model = ScriptedChatModel(
            responses=[
                # Step 1: deterministic health check (real tool — reads the wiki)
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="wiki_command",
                            args={"command": "health"},
                            id="tc-1",
                        )
                    ],
                ),
                # Step 2: write lint report
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="write_lint_report",
                            args={
                                "report": "# Lint Report\n\n## Orphan Pages\n\n- orphan-concept: No inbound links found.\n",
                            },
                            id="tc-2",
                        )
                    ],
                ),
                # Step 3: final summary
                AIMessage(
                    content="Lint complete. Found 1 orphan page: orphan-concept."
                ),
            ]
        )

        agent = create_agent(
            model=model,
            tools=CURRENT_LINT_TOOLS,
            system_prompt=build_lint_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Lint the wiki."}]},
            config=config,
        )

        # Verify final answer
        assert "Lint complete" in result["messages"][-1].content

        # Verify report file created on disk
        today = date.today().isoformat()
        report_path = wiki_with_orphan / f"lint-report-{today}.md"
        assert report_path.exists()
        report_content = report_path.read_text()
        assert "orphan-concept" in report_content

    def test_health_check_reports_orphan(self, wiki_with_orphan):
        """wiki_command health surfaces the orphan with zero LLM calls."""
        wp = str(wiki_with_orphan)
        init_shared_tools(wp)
        result = wiki_command.invoke({"command": "health"})
        assert "Pages audited: 2" in result
        assert "[high] orphan: concepts/orphan-concept" in result
