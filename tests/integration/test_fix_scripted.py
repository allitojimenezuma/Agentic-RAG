"""Integration tests for the fix agent + CLI — scripted fake model.

Verifies the Pass B contract: the fix agent consumes the structured issues from
the conversation (health_check output passed by the CLI), fixes via the pinned
kind→tool map, verifies with wiki_read_page, and NEVER calls legacy tools
(read_wiki_page / execute_command / remove_index_entry).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolCall
from typer.testing import CliRunner

from agentic_rag.agents.prompts import build_fix_prompt
from agentic_rag.cli import app
from agentic_rag.tools.fix_tools import (
    add_frontmatter,
    append_related_section,
    edit_wiki_page,
    fix_link,
)
from agentic_rag.tools.ingest_tools import delete_wiki_page
from agentic_rag.tools.nav import regenerate_index, wiki_read_page
from agentic_rag.tools.shared import init_shared_tools
from tests.fixtures.fake_llm import ScriptedChatModel

runner = CliRunner()

LEGACY_TOOL_NAMES = {
    "read_wiki_page",  # the old shared one — nav.wiki_read_page is expected
    "execute_command",
    "remove_index_entry",
    "read_index",
}

FIX_TOOLS = [
    wiki_read_page,
    edit_wiki_page,
    add_frontmatter,
    fix_link,
    append_related_section,
    regenerate_index,
    delete_wiki_page,
]


@pytest.fixture
def fixable_wiki(tmp_path: Path) -> Path:
    """A wiki with a page lacking YAML frontmatter (a fixable issue)."""
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    (wiki / "index.md").write_text("# Wiki Index\n")
    (wiki / "log.md").write_text("# Wiki Log\n")
    (wiki / "entities" / "mlx.md").write_text(
        "# MLX\n\n"
        "MLX is a machine learning framework by Apple for Apple Silicon.\n\n"
        "## Related\n\n"
        "- [[Tool Calling]]"
    )
    return wiki


def _all_tool_calls(result) -> list[dict]:
    """Collect every tool call made during the agent run."""
    calls: list[dict] = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(tc)
    return calls


class TestFixAgentContract:
    """Fix agent tool contract: structured issues -> kind tool -> wiki_read_page verify."""

    def test_missing_frontmatter_fixed_with_add_frontmatter_then_verified(self, fixable_wiki):
        """add_frontmatter -> wiki_read_page verification; no legacy tools."""
        wp = str(fixable_wiki)
        init_shared_tools(wp)

        model = ScriptedChatModel(
            responses=[
                # Step 1: fix the missing frontmatter
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="add_frontmatter",
                            args={
                                "slug": "entities/mlx",
                                "title": "MLX",
                                "page_type": "entity",
                            },
                            id="tc-1",
                        )
                    ],
                ),
                # Step 2: verify the fix by reading the page back
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
                # Step 3: final answer
                AIMessage(content="Fixed missing-frontmatter on entities/mlx."),
            ]
        )

        agent = create_agent(
            model=model,
            tools=FIX_TOOLS,
            system_prompt=build_fix_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Fix these lint issues:\n"
                            "- [missing-frontmatter] entities/mlx: Page lacks YAML frontmatter"
                        ),
                    }
                ]
            },
            config=config,
        )

        calls = _all_tool_calls(result)
        called_names = [c["name"] for c in calls]

        # No legacy tools anywhere in the sequence
        for name in called_names:
            assert name not in LEGACY_TOOL_NAMES, (
                f"Legacy tool '{name}' was called by fix agent"
            )

        # One issue -> one fix tool call -> one verification read
        assert called_names == ["add_frontmatter", "wiki_read_page"]

        # The fix actually landed on disk
        content = (fixable_wiki / "entities" / "mlx.md").read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "slug: entities/mlx" in content

    def test_broken_link_fixed_with_fix_link_then_verified(self, tmp_path):
        """fix_link -> wiki_read_page verification; no execute_command."""
        wiki = tmp_path / "wiki"
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text("# Wiki Index\n")
        (wiki / "log.md").write_text("# Wiki Log\n")
        (wiki / "concepts" / "ai.md").write_text(
            "# AI\n\n"
            "AI is the study of intelligent systems.\n\n"
            "## Related\n\n"
            "- [[MissingPage]]"
        )
        init_shared_tools(str(wiki))

        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="fix_link",
                            args={
                                "slug": "concepts/ai",
                                "old_target": "MissingPage",
                                "new_target": "MLX",
                            },
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="wiki_read_page",
                            args={"slug": "concepts/ai"},
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(content="Fixed the broken link on concepts/ai."),
            ]
        )

        agent = create_agent(
            model=model,
            tools=FIX_TOOLS,
            system_prompt=build_fix_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Fix these lint issues:\n"
                            "- [broken-link] concepts/ai: Unresolved link(s): [[MissingPage]]"
                        ),
                    }
                ]
            },
            config=config,
        )

        calls = _all_tool_calls(result)
        called_names = [c["name"] for c in calls]

        for name in called_names:
            assert name not in LEGACY_TOOL_NAMES, (
                f"Legacy tool '{name}' was called by fix agent"
            )

        assert called_names == ["fix_link", "wiki_read_page"]

        content = (wiki / "concepts" / "ai.md").read_text(encoding="utf-8")
        assert "[[MissingPage]]" not in content
        assert "[[MLX]]" in content

    def test_fix_prompt_has_no_legacy_tool_references(self):
        """The fix prompt must never instruct using dropped/legacy tools."""
        prompt = build_fix_prompt("# Test schema")
        for banned in ["execute_command", "remove_index_entry", "read_index"]:
            assert banned not in prompt, (
                f"build_fix_prompt must not reference '{banned}'"
            )
        # Issues are provided in the conversation, NOT read from the report
        assert "provided" in prompt
        assert "lint-report" in prompt
        # Pinned kind -> tool map and verification rule
        for required in [
            "missing-frontmatter",
            "broken-link",
            "missing-related",
            "missing-index",
            "orphan",
            "empty",
            "stale",
            "add_frontmatter",
            "fix_link",
            "append_related_section",
            "regenerate_index",
            "wiki_read_page",
        ]:
            assert required in prompt


class TestFixCliCommand:
    """CLI fix: runs health_check and passes structured issues to the agent."""

    def _capture_fake_build(self, captured: dict, final: str = "Fixed all issues."):
        """Return a fake build_fix_agent that records the user message sent."""

        def fake_build(settings):
            from langchain.agents import create_agent

            model = ScriptedChatModel(responses=[AIMessage(content=final)])
            agent = create_agent(
                model=model,
                tools=[],
                system_prompt="You are a test agent.",
            )

            def invoke(state, **kw):
                msg = state["messages"][-1]
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                captured["user"] = content
                return {"messages": [AIMessage(content=final)]}

            agent.invoke = invoke
            return agent

        return fake_build

    def _env(self, tmp_path: Path, wiki: Path) -> dict:
        return {
            "OPENAI_API_KEY": "sk-test-key",
            "WIKI_PATH": str(wiki),
            "AGENTS_MD_PATH": str(tmp_path / "AGENTS.md"),
        }

    def test_fix_latest_runs_health_check_and_passes_issues(self, tmp_path, fixable_wiki):
        """`fix latest` runs health_check and passes the serialized issues to the agent."""
        (tmp_path / "AGENTS.md").write_text("# Wiki Schema\n")
        captured: dict = {}
        with (
            patch.dict("os.environ", self._env(tmp_path, fixable_wiki), clear=False),
            patch(
                "agentic_rag.agents.fix.build_fix_agent",
                self._capture_fake_build(captured),
            ),
        ):
            result = runner.invoke(app, ["fix", "latest"])

        assert result.exit_code == 0
        assert "Fixed all issues." in result.output
        assert "Fix these lint issues:" in captured["user"]
        assert "[missing-frontmatter] entities/mlx" in captured["user"]

    def test_fix_issue_filter_keeps_matching_kind_only(self, tmp_path, fixable_wiki):
        """`fix <kind>` filters the health_check issues before sending."""
        (tmp_path / "AGENTS.md").write_text("# Wiki Schema\n")
        captured: dict = {}
        with (
            patch.dict("os.environ", self._env(tmp_path, fixable_wiki), clear=False),
            patch(
                "agentic_rag.agents.fix.build_fix_agent",
                self._capture_fake_build(captured),
            ),
        ):
            result = runner.invoke(app, ["fix", "missing-frontmatter"])

        assert result.exit_code == 0
        assert "[missing-frontmatter] entities/mlx" in captured["user"]
        assert "[missing-related]" not in captured["user"]
        assert "[orphan]" not in captured["user"]

    def test_fix_health_check_failure_falls_back_to_no_issues(self, tmp_path, fixable_wiki):
        """If health_check raises, the CLI degrades to 'No issues' instead of crashing."""
        (tmp_path / "AGENTS.md").write_text("# Wiki Schema\n")
        captured: dict = {}
        with (
            patch.dict("os.environ", self._env(tmp_path, fixable_wiki), clear=False),
            patch(
                "agentic_rag.agents.fix.build_fix_agent",
                self._capture_fake_build(captured, final="Nothing to do."),
            ),
            patch(
                "agentic_rag.lint.health.health_check",
                side_effect=RuntimeError("empty wiki"),
            ),
        ):
            result = runner.invoke(app, ["fix", "latest"])

        assert result.exit_code == 0
        assert "Nothing to do." in result.output
        assert captured["user"] == "No issues"

    def test_fix_unmatched_filter_passes_user_request_through(self, tmp_path, fixable_wiki):
        """A natural-language arg is kept as the instruction; issues become context."""
        (tmp_path / "AGENTS.md").write_text("# Wiki Schema\n")
        captured: dict = {}
        with (
            patch.dict("os.environ", self._env(tmp_path, fixable_wiki), clear=False),
            patch(
                "agentic_rag.agents.fix.build_fix_agent",
                self._capture_fake_build(captured),
            ),
        ):
            result = runner.invoke(
                app, ["fix", "deepseek flash is cheaper than glm5.2, fix that in glm5.2 page"]
            )

        assert result.exit_code == 0
        assert "Fixed all issues." in result.output
        # Warning echoed to the user
        assert "Warning: no issues matched" in result.output
        # The agent receives the USER'S ACTUAL WORDS as the instruction...
        assert captured["user"].startswith(
            "deepseek flash is cheaper than glm5.2, fix that in glm5.2 page"
        )
        # ...plus the deterministic issues as context (not a bare 'No issues')
        assert "health check found" in captured["user"]
        assert "[missing-frontmatter] entities/mlx" in captured["user"]
        assert captured["user"] != "No issues"

    def test_fix_no_issues_still_says_no_issues(self, tmp_path):
        """A clean wiki still sends 'No issues' (guard against regressions)."""
        wiki = tmp_path / "wiki"
        (wiki / "entities").mkdir(parents=True)
        (wiki / "index.md").write_text("# Wiki Index\n")
        (wiki / "log.md").write_text("# Wiki Log\n")
        (tmp_path / "AGENTS.md").write_text("# Wiki Schema\n")
        captured: dict = {}
        with (
            patch.dict("os.environ", self._env(tmp_path, wiki), clear=False),
            patch(
                "agentic_rag.agents.fix.build_fix_agent",
                self._capture_fake_build(captured, final="Nothing to do."),
            ),
        ):
            result = runner.invoke(app, ["fix", "latest"])

        assert result.exit_code == 0
        assert "Nothing to do." in result.output
        assert captured["user"] == "No issues"
