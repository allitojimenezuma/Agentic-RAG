"""Integration tests for the CLI — CliRunner with fixture wiki."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from agentic_rag.cli import app
from tests.fixtures.fake_llm import ScriptedChatModel

runner = CliRunner()


@pytest.fixture
def wiki_fixture(tmp_path: Path) -> Path:
    """Create a wiki directory with some content for status/log tests."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "entities").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "sources").mkdir()
    (wiki / "comparisons").mkdir()
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n"
        "## Entities\n\n"
        "- [[MLX]] - Machine learning framework | Sources: sample.md | Updated: 2025-01-15\n\n"
        "## Concepts\n\n"
        "- [[Tool Calling]] - LLM function invocation | Sources: sample.md | Updated: 2025-01-15\n\n"
        "## Sources\n\n"
        "- [Sample](sources/sample.md) - Ingested: 2025-01-15\n\n"
        "## Comparisons\n\n"
    )
    (wiki / "log.md").write_text(
        "# Wiki Log\n\n"
        "## [2025-01-15 10:30] ingest | sample.md\n"
        "- Created: [[MLX]], [[Tool Calling]]\n\n"
    )
    return wiki


@pytest.fixture
def env_with_api_key(tmp_path: Path, wiki_fixture: Path):
    """Provide env vars needed by Settings, pointing wiki_path to fixture."""
    env = {
        "OPENAI_API_KEY": "sk-test-key",
        "WIKI_PATH": str(wiki_fixture),
        "AGENTS_MD_PATH": str(tmp_path / "AGENTS.md"),
    }
    (tmp_path / "AGENTS.md").write_text("# Wiki Schema\n\nPage types: entity, concept.\n")
    with patch.dict("os.environ", env):
        yield env


def _make_fake_agent(responses):
    """Create a function that returns a fake agent with ScriptedChatModel."""

    def fake_build(settings):
        from langchain.agents import create_agent

        model = ScriptedChatModel(responses=responses)
        return create_agent(
            model=model,
            tools=[],
            system_prompt="You are a test agent.",
        )

    return fake_build


class _InterruptAgent:
    """Fake agent: first invoke returns a HITL interrupt, resumed invoke a plain result.

    Records every resume Command so tests can assert on the decisions sent.
    """

    def __init__(self, interrupt_value, resume_result):
        self._interrupt_value = interrupt_value
        self._resume_result = resume_result
        self.resume_commands = []
        self._calls = 0

    def invoke(self, input, config=None):
        self._calls += 1
        if self._calls == 1:
            return {
                "__interrupt__": [SimpleNamespace(value=self._interrupt_value)],
                "messages": [AIMessage(content="Contradiction flagged.")],
            }
        self.resume_commands.append(input)
        return self._resume_result


def _make_interrupting_build(interrupt_value, resume_result):
    """Return (fake_build, agent) — fake_build patches build_ingest_agent."""
    agent = _InterruptAgent(interrupt_value, resume_result)
    return lambda settings: agent, agent


def _contradiction_action(proposed_resolution="Merge both claims."):
    return {
        "name": "flag_contradiction",
        "args": {
            "page_slug": "entities/x",
            "existing_claim": "Python is interpreted",
            "new_claim": "Python can be compiled",
            "proposed_resolution": proposed_resolution,
        },
    }


def _resume_decisions(resume_command):
    """Extract the decisions dict from a recorded resume Command."""
    return getattr(resume_command, "resume")["decisions"]


class TestStatusCommand:
    """Test the status command."""

    def test_status_shows_page_count(self, env_with_api_key, wiki_fixture):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Wiki pages:" in result.output
        assert "Index entries:" in result.output

    def test_status_shows_last_log(self, env_with_api_key, wiki_fixture):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Last log:" in result.output
        assert "ingest" in result.output


class TestLogCommand:
    """Test the log command."""

    def test_log_shows_entries(self, env_with_api_key, wiki_fixture):
        result = runner.invoke(app, ["log"])
        assert result.exit_code == 0
        assert "ingest" in result.output
        assert "sample.md" in result.output

    def test_log_tail_limit(self, env_with_api_key, wiki_fixture):
        result = runner.invoke(app, ["log", "--tail", "1"])
        assert result.exit_code == 0
        assert "ingest" in result.output

    def test_log_empty_wiki(self, env_with_api_key, wiki_fixture):
        (wiki_fixture / "log.md").write_text("# Wiki Log\n")
        result = runner.invoke(app, ["log"])
        assert result.exit_code == 0


class TestIngestCommand:
    """Test the ingest command with a scripted agent."""

    def test_ingest_simple_flow(self, env_with_api_key, wiki_fixture, tmp_path):
        """Ingest a file — agent returns final answer without tool calls."""
        sample = tmp_path / "sample.md"
        sample.write_text("# Sample\n\nSome content.\n")

        fake_build = _make_fake_agent(
            [AIMessage(content="Ingestion complete. No new entities found.")]
        )

        with patch("agentic_rag.agents.ingest.build_ingest_agent", fake_build):
            result = runner.invoke(app, ["ingest", str(sample)])

        assert result.exit_code == 0
        assert "Ingestion complete" in result.output

    def test_ingest_with_hitl_approve(self, env_with_api_key, wiki_fixture, tmp_path):
        """Ingest with HITL interrupt — user approves."""
        sample = tmp_path / "sample.md"
        sample.write_text("# Sample\n\nSome content.\n")

        fake_build = _make_fake_agent([AIMessage(content="Done.")])

        with patch("agentic_rag.agents.ingest.build_ingest_agent", fake_build):
            result = runner.invoke(app, ["ingest", str(sample)])

        assert result.exit_code == 0

    def test_ingest_hitl_resume_approve(self, env_with_api_key, wiki_fixture, tmp_path):
        """Interrupt -> user approves -> resume sends approve decisions."""
        sample = tmp_path / "sample.md"
        sample.write_text("# Sample\n\nSome content.\n")

        interrupt_value = {
            "action_requests": [_contradiction_action()],
            "review_configs": [],
        }
        fake_build, agent = _make_interrupting_build(
            interrupt_value, {"messages": [AIMessage(content="Ingestion complete.")]}
        )

        with patch("agentic_rag.agents.ingest.build_ingest_agent", fake_build):
            result = runner.invoke(app, ["ingest", str(sample)], input="a\n")

        assert result.exit_code == 0
        assert "Ingestion complete" in result.output
        assert len(agent.resume_commands) == 1
        assert _resume_decisions(agent.resume_commands[0]) == [{"type": "approve"}]

    def test_ingest_hitl_resume_edit(self, env_with_api_key, wiki_fixture, tmp_path):
        """Interrupt -> user edits a resolution -> resume sends an edit decision."""
        sample = tmp_path / "sample.md"
        sample.write_text("# Sample\n\nSome content.\n")

        interrupt_value = {
            "action_requests": [
                _contradiction_action(proposed_resolution="Original resolution."),
                _contradiction_action(proposed_resolution="Second resolution."),
            ],
            "review_configs": [],
        }
        fake_build, agent = _make_interrupting_build(
            interrupt_value, {"messages": [AIMessage(content="Ingestion complete.")]}
        )

        with patch("agentic_rag.agents.ingest.build_ingest_agent", fake_build):
            result = runner.invoke(
                app, ["ingest", str(sample)], input="e\n1\nUse the edited resolution text\n"
            )

        assert result.exit_code == 0
        assert len(agent.resume_commands) == 1
        decisions = _resume_decisions(agent.resume_commands[0])
        assert len(decisions) == 2
        assert decisions[0]["type"] == "edit"
        assert decisions[0]["edited_action"]["name"] == "flag_contradiction"
        assert (
            decisions[0]["edited_action"]["args"]["proposed_resolution"]
            == "Use the edited resolution text"
        )
        # non-edited actions stay approved
        assert decisions[1] == {"type": "approve"}


class TestQueryCommand:
    """Test the query command with a scripted agent."""

    def test_query_returns_answer(self, env_with_api_key, wiki_fixture):
        fake_build = _make_fake_agent(
            [AIMessage(content="MLX is a machine learning framework by Apple ([[MLX]]).")]
        )

        with patch("agentic_rag.agents.query.build_query_agent", fake_build):
            result = runner.invoke(app, ["query", "What is MLX?"])

        assert result.exit_code == 0
        assert "MLX" in result.output


class TestLintCommand:
    """Test the lint command with a scripted agent."""

    def test_lint_returns_report(self, env_with_api_key, wiki_fixture):
        fake_build = _make_fake_agent(
            [AIMessage(content="Lint report: No issues found.")]
        )

        with patch("agentic_rag.agents.lint.build_lint_agent", fake_build):
            result = runner.invoke(app, ["lint"])

        assert result.exit_code == 0
        assert "Lint report" in result.output
