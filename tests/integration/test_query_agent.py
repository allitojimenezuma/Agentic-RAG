"""Integration tests for the query agent — fake LLM with scripted tool calls."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.prompts import build_query_prompt
from agentic_rag.tools.query_tools import find_relevant_pages
from agentic_rag.tools.shared import read_index, read_wiki_page, search_index
from tests.fixtures.fake_llm import ScriptedChatModel


@pytest.fixture
def wiki_with_mlx(tmp_path: Path) -> Path:
    """Create a wiki with an MLX entity page."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "entities").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "comparisons").mkdir()

    # MLX entity page (write_page places at root of wiki_path)
    (wiki / "mlx.md").write_text(
        "---\nslug: mlx\ntype: entity\ntitle: MLX\nsources:\n  - sample.md\nupdated: 2025-01-01\ntags:\n  - ml\n  - apple\n---\n\n# MLX\n\nMLX is a machine learning framework by Apple for Apple Silicon.\n\n## Related\n\n- [[Tool Calling]]\n"
    )

    # Index entry
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n## Entities\n\n- mlx (entity) - Machine learning framework by Apple | Sources: sample.md | Updated: 2025-01-01\n\n## Concepts\n\n## Sources\n\n## Comparisons\n"
    )

    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


class TestQueryAgentBasic:
    """Basic query agent construction tests."""

    def test_agent_builds_with_fake_model(self, wiki_with_mlx):
        """Agent can be created with ScriptedChatModel."""
        model = ScriptedChatModel(
            responses=[AIMessage(content="Ready to query.")]
        )
        agent = create_agent(
            model=model,
            tools=[read_index, search_index, read_wiki_page, find_relevant_pages],
            system_prompt="You are a query agent.",
        )
        assert agent is not None


class TestQueryFlow:
    """Test the full query flow with scripted tool calls."""

    def test_query_read_index_then_read_page(self, wiki_with_mlx):
        """Query flow: read_index -> read_wiki_page -> final answer with citation."""
        wp = str(wiki_with_mlx)

        model = ScriptedChatModel(
            responses=[
                # Step 1: read the index
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_index",
                            args={"wiki_path": wp},
                            id="tc-1",
                        )
                    ],
                ),
                # Step 2: read the MLX page
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_wiki_page",
                            args={"wiki_path": wp, "slug": "mlx"},
                            id="tc-2",
                        )
                    ],
                ),
                # Step 3: final answer with citation
                AIMessage(
                    content=(
                        "MLX is a machine learning framework by Apple for Apple Silicon "
                        "([[mlx]]).\n\n"
                        "**Sources consulted:**\n- [[mlx]] - MLX entity page"
                    )
                ),
            ]
        )

        agent = create_agent(
            model=model,
            tools=[read_index, search_index, read_wiki_page, find_relevant_pages],
            system_prompt=build_query_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What is MLX?"}]},
            config=config,
        )

        answer = result["messages"][-1].content
        assert "[[mlx]]" in answer
        assert "Sources consulted" in answer

    def test_query_no_writes_called(self, wiki_with_mlx):
        """Query agent should never call write tools."""
        wp = str(wiki_with_mlx)
        write_tool_names = {"create_page", "update_page", "delete_wiki_page", "append_log", "update_index"}
        called_tools: list[str] = []

        # Intercept tool calls to track which tools are called
        original_read_index = read_index

        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_index",
                            args={"wiki_path": wp},
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(
                    content="No information available."
                ),
            ]
        )

        agent = create_agent(
            model=model,
            tools=[read_index, search_index, read_wiki_page, find_relevant_pages],
            system_prompt=build_query_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What is quantum computing?"}]},
            config=config,
        )

        # Verify no write tools were called (the query agent doesn't have them)
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] in write_tool_names:
                        pytest.fail(f"Write tool '{tc['name']}' was called by query agent")
