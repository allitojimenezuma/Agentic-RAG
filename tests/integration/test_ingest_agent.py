"""Integration tests for the ingest agent — fake LLM with scripted tool calls."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolCall
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.agents.prompts import build_ingest_prompt
from agentic_rag.tools.ingest_tools import (
    append_log,
    create_page,
    read_source,
    update_index,
)
from agentic_rag.tools.shared import read_index, read_wiki_page, search_index
from tests.fixtures.fake_llm import ScriptedChatModel


@pytest.fixture
def sample_source(tmp_path: Path) -> Path:
    """Create a small sample markdown source file."""
    src = tmp_path / "sample.md"
    src.write_text(
        "# Sample Document\n\n"
        "## MLX\n"
        "MLX is a machine learning framework by Apple for Apple Silicon.\n\n"
        "## Tool Calling\n"
        "Tool calling lets LLMs invoke external functions.\n"
    )
    return src


@pytest.fixture
def populated_wiki(tmp_path: Path) -> Path:
    """Create a wiki directory with empty index/log."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "entities").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "comparisons").mkdir()
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n\n## Comparisons\n"
    )
    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


class TestIngestAgentBasic:
    """Basic ingest agent construction and invocation tests."""

    def test_agent_builds_with_fake_model(self, populated_wiki, tmp_path):
        """Agent can be created with ScriptedChatModel as model."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="I need to ingest a file.", tool_calls=[]),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[read_source, read_index, search_index, read_wiki_page, create_page, update_index, append_log],
            system_prompt="You are a test agent.",
        )
        assert agent is not None

    def test_agent_runs_with_empty_tool_calls(self, populated_wiki):
        """Agent returns a final answer when model has no tool_calls."""
        model = ScriptedChatModel(
            responses=[
                AIMessage(content="Nothing to do here."),
            ]
        )
        agent = create_agent(
            model=model,
            tools=[read_source, read_index, search_index, read_wiki_page, create_page, update_index, append_log],
            system_prompt="You are a test agent.",
        )
        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Hello"}]},
            config=config,
        )
        assert result["messages"][-1].content == "Nothing to do here."


class TestIngestFlow:
    """Test the full ingest flow with scripted tool calls."""

    def test_ingest_creates_pages_and_updates_index(
        self, sample_source, populated_wiki, tmp_path
    ):
        """Full ingest flow: read_source -> search_index -> create_page x2 -> update_index x2 -> append_log."""
        wp = str(populated_wiki)
        sp = str(sample_source)

        model = ScriptedChatModel(
            responses=[
                # Step 1: read the source
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_source",
                            args={"source_path": sp},
                            id="tc-1",
                        )
                    ],
                ),
                # Step 2: search index for MLX (nothing found)
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="search_index",
                            args={"wiki_path": wp, "query": "MLX"},
                            id="tc-2",
                        )
                    ],
                ),
                # Step 3: create MLX entity page
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="create_page",
                            args={
                                "wiki_path": wp,
                                "slug": "mlx",
                                "page_type": "entity",
                                "title": "MLX",
                                "content": "# MLX\n\nMLX is a machine learning framework by Apple.",
                                "sources": ["sample.md"],
                            },
                            id="tc-3",
                        )
                    ],
                ),
                # Step 4: create Tool Calling concept page
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="create_page",
                            args={
                                "wiki_path": wp,
                                "slug": "tool-calling",
                                "page_type": "concept",
                                "title": "Tool Calling",
                                "content": "# Tool Calling\n\nTool calling lets LLMs invoke external functions.",
                                "sources": ["sample.md"],
                            },
                            id="tc-4",
                        )
                    ],
                ),
                # Step 5: update index for both pages (single call with MLX)
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="update_index",
                            args={
                                "wiki_path": wp,
                                "slug": "mlx",
                                "page_type": "entity",
                                "summary": "Machine learning framework by Apple for Apple Silicon",
                                "sources": ["sample.md"],
                            },
                            id="tc-5",
                        )
                    ],
                ),
                # Step 6: update index for Tool Calling
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="update_index",
                            args={
                                "wiki_path": wp,
                                "slug": "tool-calling",
                                "page_type": "concept",
                                "summary": "LLM function-invocation capability",
                                "sources": ["sample.md"],
                            },
                            id="tc-6",
                        )
                    ],
                ),
                # Step 7: append log
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="append_log",
                            args={
                                "wiki_path": wp,
                                "op": "ingest",
                                "title": "sample.md",
                                "details": "Created: [[MLX]], [[Tool Calling]]",
                            },
                            id="tc-7",
                        )
                    ],
                ),
                # Step 8: final answer
                AIMessage(content="Ingestion complete. Created 2 pages: mlx, tool-calling."),
            ]
        )

        agent = create_agent(
            model=model,
            tools=[read_source, read_index, search_index, read_wiki_page, create_page, update_index, append_log],
            system_prompt=build_ingest_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"Ingest {sp}"}]},
            config=config,
        )

        # Verify final answer
        assert "Ingestion complete" in result["messages"][-1].content

        # Verify pages created on disk (write_page places at root of wiki_path)
        assert (populated_wiki / "mlx.md").exists()
        assert (populated_wiki / "tool-calling.md").exists()

        # Verify index updated
        index_content = (populated_wiki / "index.md").read_text()
        assert "Mlx" in index_content or "mlx" in index_content
        assert "Tool Calling" in index_content or "tool-calling" in index_content

        # Verify log entry (format: ## [YYYY-MM-DD HH:MM] ingest | title)
        log_content = (populated_wiki / "log.md").read_text()
        assert "ingest" in log_content
        assert "sample.md" in log_content
