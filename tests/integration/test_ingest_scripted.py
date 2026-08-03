"""Integration tests for the ingest agent — fake LLM with scripted tool calls.

Verifies the Pass B tool contract: submit_extraction -> match_page_tool ->
create_page/update_page -> regenerate_index -> append_log, with NO legacy
index/read tools (read_index, search_index, find_relevant_pages, update_index).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.prompts import build_ingest_prompt
from agentic_rag.schemas.extraction import Concept, Entity
from agentic_rag.tools.ingest_grounding import submit_extraction
from agentic_rag.tools.ingest_tools import (
    append_log,
    create_page,
    delete_wiki_page,
    flag_contradiction,
    read_source,
    update_page,
)
from agentic_rag.tools.nav import regenerate_index, wiki_read_page
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.match import match_page_tool
from tests.fixtures.fake_llm import ScriptedChatModel

LEGACY_TOOL_NAMES = {
    "read_index",
    "search_index",
    "find_relevant_pages",
    "update_index",
    "read_wiki_page",  # the old shared one — nav.wiki_read_page is expected
}

INGEST_TOOLS = [
    read_source,
    submit_extraction,
    match_page_tool,
    wiki_read_page,
    create_page,
    update_page,
    flag_contradiction,
    regenerate_index,
    append_log,
    delete_wiki_page,
]


@pytest.fixture
def empty_wiki(tmp_path: Path) -> Path:
    """An empty wiki directory with a bare index and log."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "index.md").write_text("# Wiki Index\n")
    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


def _all_tool_calls(result) -> list[dict]:
    """Collect every tool call made during the agent run."""
    calls: list[dict] = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(tc)
    return calls


class TestIngestAgentContract:
    """Ingest agent tool contract: no legacy index tools, index regenerated."""

    def test_ingest_flow_regenerates_index_no_legacy_tools(self, empty_wiki):
        """submit_extraction -> match_page_tool -> create_page -> regenerate_index -> append_log."""
        wp = str(empty_wiki)
        init_shared_tools(wp)

        model = ScriptedChatModel(
            responses=[
                # Step 1: structured extraction of the source
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="submit_extraction",
                            args={
                                "entities": [
                                    {
                                        "name": "MLX",
                                        "type": "software",
                                        "summary": (
                                            "Machine learning framework by Apple "
                                            "for Apple Silicon."
                                        ),
                                        "sources": ["sample.md"],
                                    }
                                ],
                                "concepts": [
                                    {
                                        "name": "Tool Calling",
                                        "summary": (
                                            "Pattern where an LLM invokes "
                                            "registered tools."
                                        ),
                                        "related_entities": ["MLX"],
                                    }
                                ],
                                "contradictions": [],
                            },
                            id="tc-1",
                        )
                    ],
                ),
                # Step 2: match each extracted entity/concept (empty wiki -> none)
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="match_page_tool",
                            args={"name": "MLX", "page_type": "entity"},
                            id="tc-2",
                        )
                    ],
                ),
                # Step 3: create the new page
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="create_page",
                            args={
                                "slug": "entities/mlx",
                                "page_type": "entity",
                                "title": "MLX",
                                "content": (
                                    "# MLX\n\n"
                                    "MLX is a machine learning framework by "
                                    "Apple for Apple Silicon.\n\n"
                                    "## Related\n\n"
                                    "- [[Tool Calling]]"
                                ),
                                "sources": ["sample.md"],
                            },
                            id="tc-3",
                        )
                    ],
                ),
                # Step 4: regenerate the index from pages on disk
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="regenerate_index", args={}, id="tc-4")
                    ],
                ),
                # Step 5: log the ingestion
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="append_log",
                            args={"op": "ingest", "title": "sample.md"},
                            id="tc-5",
                        )
                    ],
                ),
                # Step 6: final answer
                AIMessage(content="Ingested sample.md."),
            ]
        )

        agent = create_agent(
            model=model,
            tools=INGEST_TOOLS,
            system_prompt=build_ingest_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "ingest sample.md"}]},
            config=config,
        )

        calls = _all_tool_calls(result)
        called_names = [c["name"] for c in calls]

        # No legacy index/read tools in the sequence
        for name in called_names:
            assert name not in LEGACY_TOOL_NAMES, (
                f"Legacy tool '{name}' was called by ingest agent"
            )

        # Expected order of the scripted flow
        assert called_names[0] == "submit_extraction"
        assert "match_page_tool" in called_names
        assert "create_page" in called_names
        assert "update_page" not in called_names
        assert "regenerate_index" in called_names
        assert called_names[-1] == "append_log"

        # Exactly one regenerate_index call
        assert called_names.count("regenerate_index") == 1

        # index.md was regenerated from the created page (derived view)
        index_text = (empty_wiki / "index.md").read_text(encoding="utf-8")
        assert "entities/mlx" in index_text or "MLX" in index_text

        # Page and log entry were written
        assert (empty_wiki / "entities" / "mlx.md").is_file()
        log_text = (empty_wiki / "log.md").read_text(encoding="utf-8")
        assert "[ingest]" in log_text or "ingest |" in log_text

    def test_ingest_prompt_has_no_legacy_tool_references(self):
        """The ingest prompt must never instruct using dropped/legacy tools.

        The rules section keeps the pinned rule 'NEVER call update_index'
        (task/spec both mandate that exact text), so update_index is allowed
        to appear ONLY inside that negation.
        """
        prompt = build_ingest_prompt("# Test schema")
        for banned in [
            "wiki_search",
            "read_index",
            "search_index",
            "find_relevant_pages",
        ]:
            assert banned not in prompt, (
                f"build_ingest_prompt must not reference '{banned}'"
            )
        # update_index may appear only in the 'NEVER call' rule, never as
        # an instruction to use it.
        assert prompt.count("update_index") == 1
        assert "NEVER call update_index" in prompt
        assert "call update_index" not in prompt.replace(
            "NEVER call update_index", ""
        )
        # And must reference the required new tools
        for required in [
            "submit_extraction",
            "match_page_tool",
            "wiki_read_page",
            "regenerate_index",
            "append_log",
        ]:
            assert required in prompt
