"""Grounded-flow tests for the query agent (cite-or-die) and CLI rendering.

Tests the nav-tools + auto-built answer pipeline end-to-end with the
``ScriptedChatModel`` harness: there is no finalization tool — the answer is
synthesized from the model's final message, a fabricated ``[[link]]`` citation
is dropped because the agent never navigated that page, and the CLI renders the
structured ``QueryAnswer`` (falling back to the raw final message for plain
agents without a NavCapture).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolCall
from typer.testing import CliRunner

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_query_prompt
from agentic_rag.tools.grounding import build_final_answer, new_nav_capture
from agentic_rag.tools.nav import wiki_command
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.cli import app
from tests.fixtures.fake_llm import ScriptedChatModel

runner = CliRunner()


@pytest.fixture
def grounded_wiki(tmp_path: Path) -> Path:
    """A wiki with an MLX entity page (slug entities/mlx) plus a page that
    exists but is never navigated this turn (not citable)."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "entities").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "comparisons").mkdir()

    (wiki / "entities" / "mlx.md").write_text(
        "---\nslug: entities/mlx\ntype: entity\ntitle: MLX\nsources:\n  - sample.md\n"
        "updated: 2025-01-01\ntags:\n  - ml\n  - apple\n---\n\n"
        "# MLX\n\nMLX is a machine learning framework by Apple for Apple Silicon.\n\n"
        "## Related\n\n- [[Tool Calling]]\n"
    )
    (wiki / "entities" / "navigated-only.md").write_text(
        "---\nslug: entities/navigated-only\ntype: entity\ntitle: Navigated Only\n"
        "sources: []\nupdated: 2025-01-01\ntags: []\n---\n\n"
        "# Navigated Only\n\nA page that exists on disk but is never navigated "
        "this turn, so it must not be citable.\n"
    )

    (wiki / "index.md").write_text(
        "# Wiki Index\n\n## Entities\n\n## Concepts\n\n## Sources\n\n## Comparisons\n"
    )
    (wiki / "log.md").write_text("# Wiki Log\n")
    return wiki


@pytest.fixture
def env_with_api_key(tmp_path: Path, grounded_wiki: Path):
    """Env vars for Settings, pointing wiki_path at the grounded fixture wiki."""
    env = {
        "OPENAI_API_KEY": "sk-test-key",
        "WIKI_PATH": str(grounded_wiki),
        "AGENTS_MD_PATH": str(tmp_path / "AGENTS.md"),
    }
    (tmp_path / "AGENTS.md").write_text("# Wiki Schema\n\nPage types: entity, concept.\n")
    with patch.dict("os.environ", env):
        yield env


def _build_grounded_agent(wiki: Path):
    """Grounded scripted agent: navigates mlx via wiki_command search, then
    writes a final answer citing BOTH the navigated slug and a fabricated one."""
    init_shared_tools(str(wiki))
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="wiki_command", args={"command": 'search "mlx"'}, id="tc-1")],
            ),
            AIMessage(
                content=(
                    "MLX is a machine learning framework by Apple "
                    "([[entities/mlx]], [[entities/fabricated]])."
                )
            ),
        ]
    )
    agent = build_agent(
        model=model,
        tools=[wiki_command],
        system_prompt=build_query_prompt("# Test schema"),
    )
    agent._nav_capture = new_nav_capture()
    return agent


class TestGroundedQueryAgent:
    """Agent-level grounded flow: fabricated citations are dropped (cite-or-die)."""

    def test_final_answer_drops_fabricated_citation(self, grounded_wiki):
        agent = _build_grounded_agent(grounded_wiki)
        config = {"configurable": {"thread_id": str(uuid4())}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What is MLX?"}]},
            config=config,
        )

        # wiki_search must have recorded the navigated slug into the capture.
        assert "entities/mlx" in agent._nav_capture.navigated

        qa = build_final_answer(result["messages"], agent._nav_capture.navigated)
        assert [c.slug for c in qa.citations] == ["entities/mlx"]
        assert all(c.slug != "entities/fabricated" for c in qa.citations)
        assert qa.confidence == "high"


class TestGroundedCli:
    """CLI-level rendering of the auto-built QueryAnswer + compat fallback."""

    def test_cli_renders_auto_built_answer(self, grounded_wiki, env_with_api_key):
        agent = _build_grounded_agent(grounded_wiki)
        with patch("agentic_rag.agents.query.build_query_agent", lambda settings: agent):
            result = runner.invoke(app, ["query", "What is MLX?"])

        assert result.exit_code == 0
        assert "Answer:" in result.output
        assert "Confidence: high" in result.output
        # The navigated [[link]] became a citation (with its frontmatter title).
        assert "entities/mlx - MLX" in result.output
        # The fabricated link is NOT promoted to a citation.
        assert "entities/fabricated -" not in result.output

    def test_cli_compat_fallback_plain_agent(self, env_with_api_key):
        """Plain agent (create_agent, no _nav_capture) falls back to raw content."""

        def fake_build(settings):
            from langchain.agents import create_agent

            model = ScriptedChatModel(responses=[AIMessage(content="plain answer ([[mlx]])")])
            return create_agent(
                model=model,
                tools=[],
                system_prompt="You are a test agent.",
            )

        with patch("agentic_rag.agents.query.build_query_agent", fake_build):
            result = runner.invoke(app, ["query", "What is MLX?"])

        assert result.exit_code == 0
        assert result.output.strip() == "plain answer ([[mlx]])"
