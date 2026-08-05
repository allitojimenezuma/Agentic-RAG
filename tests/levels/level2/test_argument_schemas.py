"""Level 2 — deterministic argument-schema + tool-error contracts (0 LLM).

Pins the "tool error semantics" convention from docs/spec.md: every tool
returns a string, never raises; errors are returned as ``"ERROR:"`` /
``"Error:"``-prefixed strings. Three layers are pinned here:

1. Direct tool error strings — call the ``@tool`` functions directly (after
   ``init_shared_tools``) and assert the EXACT error strings for the standard
   misuse cases (create on existing, update/delete/fix on missing, etc.).
2. Schema-level invalid args — when a scripted agent emits a ToolCall with
   WRONG/INCOMPLETE args (e.g. ``flag_contradiction`` missing
   ``proposed_resolution``), LangChain's tool schema catches it: the run must
   NOT raise, and the tool result must be a string carrying
   Error/validation text (never an exception).
3. Path-guard short-circuit shape — a scripted write-tool call with a
   traversal slug is short-circuited by ``path_guard_middleware`` and the
   tool result starts with ``"ERROR:"`` (matrix-tested in level2/T2; here we
   assert the SHAPE once through a real agent run).

Every run uses ScriptedChatModel + build_agent; no Settings, no network, no
real LLM. Each run gets its own thread_id so the in-memory checkpointer never
shares state across runs.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import (
    build_fix_prompt,
    build_ingest_prompt,
)
from agentic_rag.tools.fix_tools import (
    add_frontmatter,
    append_related_section,
    fix_link,
)
from agentic_rag.tools.ingest_tools import (
    create_page,
    delete_wiki_page,
    flag_contradiction,
    read_source,
    update_page,
)
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.match import match_page_tool
from tests.fixtures.fake_llm import ScriptedChatModel


def _run(agent, user_content: str, config: dict) -> dict:
    """Invoke a built agent with a plain user message."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]}, config=config
    )


# --- 1. Direct tool error strings (exact, orchestrator-verified) --------------
class TestDirectToolErrorStrings:
    """Calling the @tool functions directly returns exact error strings."""

    def test_create_page_on_existing_page(self, eval_wiki):
        """create_page on an existing slug errors with the pinned message."""
        init_shared_tools(str(eval_wiki))
        result = create_page.invoke(
            {
                "slug": "entities/mlx",
                "page_type": "entity",
                "title": "MLX",
                "content": "duplicate",
            }
        )
        assert isinstance(result, str)
        assert result == (
            "Error: Page 'entities/mlx' already exists. Use update_page to modify it."
        )

    def test_update_page_on_missing_page(self, eval_wiki):
        """update_page on a missing slug errors with the pinned message."""
        init_shared_tools(str(eval_wiki))
        result = update_page.invoke(
            {"slug": "entities/does-not-exist", "content": "body"}
        )
        assert isinstance(result, str)
        assert result == (
            "Error: Page 'entities/does-not-exist' does not exist. "
            "Use create_page first."
        )

    def test_delete_wiki_page_on_missing_page(self, eval_wiki):
        """delete_wiki_page on a missing slug errors with the pinned message."""
        init_shared_tools(str(eval_wiki))
        result = delete_wiki_page.invoke({"slug": "entities/does-not-exist"})
        assert isinstance(result, str)
        assert result == "Error: Wiki page not found: entities/does-not-exist"

    def test_fix_link_missing_page(self, eval_wiki):
        """fix_link on a missing page returns the unpinned-prefix not-found."""
        init_shared_tools(str(eval_wiki))
        result = fix_link.invoke(
            {"slug": "entities/does-not-exist", "old_target": "A", "new_target": "B"}
        )
        assert isinstance(result, str)
        assert result == "Page not found: entities/does-not-exist"

    def test_fix_link_absent_old_target(self, eval_wiki):
        """fix_link with an old_target not present reports zero replacements."""
        init_shared_tools(str(eval_wiki))
        result = fix_link.invoke(
            {"slug": "entities/mlx", "old_target": "NoSuchTarget", "new_target": "B"}
        )
        assert isinstance(result, str)
        assert result == "No links to 'NoSuchTarget' found in entities/mlx"

    def test_add_frontmatter_on_page_with_frontmatter(self, eval_wiki):
        """add_frontmatter refuses a page that already has frontmatter."""
        init_shared_tools(str(eval_wiki))
        result = add_frontmatter.invoke(
            {"slug": "entities/mlx", "title": "MLX", "page_type": "entity"}
        )
        assert isinstance(result, str)
        assert result == "Error: entities/mlx already has frontmatter"

    def test_append_related_section_missing_page(self, eval_wiki):
        """append_related_section on a missing page returns the not-found."""
        init_shared_tools(str(eval_wiki))
        result = append_related_section.invoke(
            {"slug": "entities/does-not-exist", "links": ["MLX"]}
        )
        assert isinstance(result, str)
        assert result == "Page not found: entities/does-not-exist"

    def test_read_source_nonexistent_path(self, eval_wiki):
        """read_source with a bad path returns an Error: string, never raises."""
        init_shared_tools(str(eval_wiki))
        result = read_source.invoke({"source_path": "/nonexistent/path.md"})
        assert isinstance(result, str)
        assert result.startswith("Error: could not read source '/nonexistent/path.md'")

    def test_match_page_tool_returns_decision_string(self, eval_wiki):
        """match_page_tool always returns a '<decision>: ...' string, never raises."""
        init_shared_tools(str(eval_wiki))
        exact = match_page_tool.invoke({"name": "MLX", "page_type": "entity"})
        assert isinstance(exact, str)
        assert exact.startswith("exact:")

        none = match_page_tool.invoke(
            {"name": "Brand New Entity", "page_type": "entity"}
        )
        assert isinstance(none, str)
        assert none.startswith("none:")

    def test_flag_contradiction_direct_returns_flag_string(self, eval_wiki):
        """flag_contradiction called directly returns the HITL flag string
        (the interrupt is added by the HITL middleware, not the tool)."""
        init_shared_tools(str(eval_wiki))
        result = flag_contradiction.invoke(
            {
                "page_slug": "entities/mlx",
                "existing_claim": "MLX is by Apple.",
                "new_claim": "MLX is by Google.",
                "proposed_resolution": "Update the page.",
            }
        )
        assert isinstance(result, str)
        assert result.startswith("CONTRADICTION FLAGGED (requires HITL):")


# --- 2. Schema-invalid args through an agent: never raise ---------------------
# Each case emits a ToolCall with WRONG/INCOMPLETE args. The tool schema
# (validated by LangChain when called through the agent) must reject it as an
# error STRING fed back into the run — never an exception.
SCHEMA_INVALID_CASES: list[tuple] = [
    (
        "flag_contradiction_missing_resolution",
        flag_contradiction,
        [flag_contradiction],
        build_ingest_prompt,
        {"page_slug": "entities/mlx"},  # missing existing_claim + proposed_resolution
    ),
    (
        "create_page_missing_content",
        create_page,
        [create_page],
        build_ingest_prompt,
        {"slug": "entities/fresh", "page_type": "entity", "title": "Fresh"},
    ),
    (
        "fix_link_missing_new_target",
        fix_link,
        [fix_link],
        build_fix_prompt,
        {"slug": "entities/mlx", "old_target": "Something"},
    ),
]


class TestSchemaInvalidArgsThroughAgent:
    """Wrong/incomplete tool args fail schema validation as strings, never raise."""

    @pytest.mark.parametrize(
        "name,tool,tools,prompt_builder,bad_args",
        SCHEMA_INVALID_CASES,
        ids=[c[0] for c in SCHEMA_INVALID_CASES],
    )
    def test_incomplete_args_return_error_string(
        self, name, tool, tools, prompt_builder, bad_args, eval_wiki
    ):
        """Run completes; the failing tool result is a string with validation text."""
        init_shared_tools(str(eval_wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name=tool.name, args=bad_args, id="tc-1")],
                ),
                AIMessage(content="recovered from the schema error"),
            ]
        )
        agent = build_agent(
            model=model,
            tools=tools,
            system_prompt=prompt_builder("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "proceed", config)

        # The run completed and produced the scripted final answer.
        assert result["messages"][-1].content == "recovered from the schema error"

        # The failing tool result is a STRING carrying Error/validation text.
        error_strings = [
            m.content
            for m in result["messages"]
            if isinstance(m.content, str)
            and any(tok in m.content.lower() for tok in ("error", "validation"))
        ]
        assert error_strings, "expected a string tool-error message, got none"
        assert all(isinstance(s, str) for s in error_strings)


# --- 3. Path-guard short-circuit shape (one agent run) -------------------------
class TestPathGuardShortCircuit:
    """Traversal slugs on write tools are short-circuited with an ERROR: string."""

    def test_traversal_slug_short_circuits_with_error_prefix(self, eval_wiki):
        """create_page with '../' slug → tool result starts with 'ERROR:'."""
        init_shared_tools(str(eval_wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="create_page",
                            args={
                                "slug": "../escape",
                                "page_type": "entity",
                                "title": "Evil",
                                "content": "x",
                            },
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(content="write blocked"),
            ]
        )
        agent = build_agent(
            model=model,
            tools=[create_page],
            system_prompt=build_ingest_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "write a page", config)

        # The short-circuit result is a string starting with "ERROR:".
        blocked = [
            m.content
            for m in result["messages"]
            if isinstance(m.content, str) and m.content.startswith("ERROR:")
        ]
        assert blocked
        assert result["messages"][-1].content == "write blocked"
        # Nothing escaped the wiki directory.
        assert not (eval_wiki.parent / "escape.md").exists()
