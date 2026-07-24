"""Integration tests for the lint agent — fake LLM with scripted tool calls."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.prompts import build_lint_prompt
from agentic_rag.tools.lint_tools import (
    extract_concepts,
    find_inbound_links,
    read_all_pages,
    write_lint_report,
)
from agentic_rag.tools.shared import read_index
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

    # Main page (not orphan, at root per write_page behavior)
    (wiki / "python.md").write_text(
        "---\nslug: python\ntype: entity\ntitle: Python\nsources:\n  - sample.md\nupdated: 2025-01-01\ntags: []\n---\n\n# Python\n\nPython is a programming language.\n\n## Related\n\n- [[Tool Calling]]\n"
    )

    # Orphan page (no inbound links from other pages)
    (wiki / "orphan-concept.md").write_text(
        "---\nslug: orphan-concept\ntype: concept\ntitle: Orphan Concept\nsources:\n  - sample.md\nupdated: 2025-01-01\ntags: []\n---\n\n# Orphan Concept\n\nThis concept is not linked from anywhere.\n"
    )

    # Index
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n## Entities\n\n- python (entity) - Programming language | Sources: sample.md | Updated: 2025-01-01\n\n## Concepts\n\n- orphan-concept (concept) - Unlinked concept | Sources: sample.md | Updated: 2025-01-01\n\n## Sources\n\n## Comparisons\n"
    )

    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


class TestLintAgentBasic:
    """Basic lint agent construction tests."""

    def test_agent_builds_with_fake_model(self, wiki_with_orphan):
        """Agent can be created with ScriptedChatModel."""
        model = ScriptedChatModel(
            responses=[AIMessage(content="Ready to lint.")]
        )
        agent = create_agent(
            model=model,
            tools=[read_all_pages, read_index, find_inbound_links, extract_concepts, write_lint_report],
            system_prompt="You are a lint agent.",
        )
        assert agent is not None


class TestLintFlow:
    """Test the full lint flow with scripted tool calls."""

    def test_lint_writes_report(self, wiki_with_orphan):
        """Lint flow: read_all_pages -> find_inbound_links -> write_lint_report."""
        wp = str(wiki_with_orphan)

        model = ScriptedChatModel(
            responses=[
                # Step 1: read all pages
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_all_pages",
                            args={"wiki_path": wp},
                            id="tc-1",
                        )
                    ],
                ),
                # Step 2: find inbound links for orphan
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="find_inbound_links",
                            args={"wiki_path": wp, "slug": "orphan-concept"},
                            id="tc-2",
                        )
                    ],
                ),
                # Step 3: write lint report
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="write_lint_report",
                            args={
                                "wiki_path": wp,
                                "report": "# Lint Report\n\n## Orphan Pages\n\n- orphan-concept: No inbound links found.\n",
                            },
                            id="tc-3",
                        )
                    ],
                ),
                # Step 4: final summary
                AIMessage(
                    content="Lint complete. Found 1 orphan page: orphan-concept."
                ),
            ]
        )

        agent = create_agent(
            model=model,
            tools=[read_all_pages, read_index, find_inbound_links, extract_concepts, write_lint_report],
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
        from datetime import date

        today = date.today().isoformat()
        report_path = wiki_with_orphan / f"lint-report-{today}.md"
        assert report_path.exists()
        report_content = report_path.read_text()
        assert "orphan-concept" in report_content
