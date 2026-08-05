"""Unit tests for frontend/ui_common + frontend/builders.

Two layers, mirroring the spec's acceptance for ``test_ui_common.py``:

- Pure asserts: ``agent_config`` EXACTLY mirrors ``src/agentic_rag/cli.py``
  per-agent shapes (recursion limits for query/ingest, none for lint/fix),
  the ``AGENTS`` label map, and the unit-testable helpers
  (``action_summary``, ``make_pending``, ``render_final``).
- AppTest HITL flow via ``AppTest.from_string`` + ``ScriptedChatModel`` stub
  page: a duck-typed fake HITL agent (scripted interrupt turn, final on
  resume) driven through ``ui_common.run_turn``. Clicking "Approve all"
  produces ``[{"type": "approve"}]*N`` and resumes; "Reject all" carries the
  feedback; the edit path (flag_contradiction only) produces the
  ``edited_action`` shape. The LLM is never contacted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import builders
from frontend import ui_common
from frontend.agent_driver import FinalMessage
from frontend.history_store import HistoryStore

# --- Pure asserts ------------------------------------------------------------

DELETE_MLX = {"name": "delete_wiki_page", "args": {"page_slug": "entities/mlx"}}
DELETE_OTHER = {"name": "delete_wiki_page", "args": {"page_slug": "entities/other"}}
CONTRADICTION = {
    "name": "flag_contradiction",
    "args": {"slug": "entities/a", "proposed_resolution": "old text"},
}


class _FakeSettings:
    recursion_limit = 30
    ingest_recursion_limit = 200


class TestAgentConfig:
    def test_agent_config_mirrors_cli(self, monkeypatch):
        """Acceptance: exact cli.py shapes per agent."""
        monkeypatch.setattr(builders, "get_settings", lambda: _FakeSettings())
        assert builders.agent_config("query", "t-1") == {
            "configurable": {"thread_id": "t-1"},
            "recursion_limit": 30,
        }
        assert builders.agent_config("ingest", "t-1") == {
            "configurable": {"thread_id": "t-1"},
            "recursion_limit": 200,
        }
        assert builders.agent_config("lint", "t-1") == {
            "configurable": {"thread_id": "t-1"}
        }
        assert builders.agent_config("fix", "t-1") == {
            "configurable": {"thread_id": "t-1"}
        }

    def test_lint_and_fix_omit_recursion_limit(self, monkeypatch):
        """CLI omits recursion_limit for lint/fix — the key must not appear."""
        monkeypatch.setattr(builders, "get_settings", lambda: _FakeSettings())
        for key in ("lint", "fix"):
            assert "recursion_limit" not in builders.agent_config(key, "t-1")

    def test_unknown_agent_raises(self, monkeypatch):
        monkeypatch.setattr(builders, "get_settings", lambda: _FakeSettings())
        with pytest.raises(ValueError):
            builders.agent_config("nope", "t-1")


class TestPureHelpers:
    def test_agents_label_map(self):
        assert ui_common.AGENTS == {
            "query": "Wiki Q&A",
            "ingest": "Ingest",
            "lint": "Lint",
            "fix": "Fix",
        }
        assert set(ui_common.AGENT_ICONS) == set(ui_common.AGENTS)

    def test_action_summary_mirrors_cli_display(self):
        # cli.py shows [i] name(desc) with page_slug/slug first.
        assert ui_common.action_summary(DELETE_MLX) == "delete_wiki_page(entities/mlx)"
        assert (
            ui_common.action_summary(CONTRADICTION)
            == "flag_contradiction(entities/a, proposed_resolution='old text')"
        )
        # remaining args keep key=value; slug is dropped from the tail.
        assert (
            ui_common.action_summary(
                {"name": "wiki_search", "args": {"query": "hello", "top_k": 5}}
            )
            == "wiki_search(query='hello', top_k=5)"
        )

    def test_action_summary_truncates(self):
        long_args = {"slug": "x", "note": "y" * 300}
        summary = ui_common.action_summary({"name": "flag_contradiction", "args": long_args})
        assert summary.endswith("...")
        assert len(summary) <= 120

    def test_action_summary_tolerates_garbage(self):
        assert ui_common.action_summary(None) == "unknown()"
        assert ui_common.action_summary("junk") == "unknown()"
        assert ui_common.action_summary({"name": ""}) == "unknown()"
        assert (
            ui_common.action_summary({"name": "t", "args": "not-a-dict"})
            == "t()"
        )

    def test_make_pending_shape(self):
        pending = ui_common.make_pending([DELETE_MLX], "Checking…")
        assert pending == {"actions": [DELETE_MLX], "turn_text": "Checking…"}
        # decisions only appears once a widget button writes it (resume gate).
        assert "decisions" not in pending

    def test_render_final_returns_text(self):
        assert (
            ui_common.render_final("Ingest", FinalMessage(text="Done.")) == "Done."
        )


# --- AppTest HITL flow -------------------------------------------------------

# Stub page: a duck-typed fake HITL agent whose AIMessages come from a
# ScriptedChatModel (never the real LLM). Scripted via env vars:
#   STUB_ACTIONS    — JSON list of pending action_requests
#   STUB_RESUME_TEXT— the assistant text on the resumed turn
#   STUB_OUT_DIR    — tmp dir for the HistoryStore + an agent-call log
_HITL_STUB = r"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tests"))

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, ToolCall
from langgraph.types import Command

import frontend.builders as agents_mod
from frontend import ui_common
from frontend.history_store import HistoryStore
from tests.fixtures.fake_llm import ScriptedChatModel

OUT_DIR = Path(os.environ["STUB_OUT_DIR"])
LOG_FILE = OUT_DIR / "agent_calls.jsonl"
MODEL_META = {"langgraph_node": "model"}

ACTIONS = json.loads(os.environ["STUB_ACTIONS"])
RESUME_TEXT = os.environ["STUB_RESUME_TEXT"]


class FakeAgent:
    # Duck-typed HITL agent: call 1 interrupts, call 2 (resume) finishes.

    def __init__(self):
        self.calls = 0

    async def astream(self, inputs, config, stream_mode="messages"):
        self.calls += 1
        resume = inputs.resume if isinstance(inputs, Command) else None
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"call": self.calls, "resume": resume, "config": config}) + "\n")
        if self.calls == 1:
            yield "messages", (AIMessageChunk(content="Checking…"), MODEL_META)
            yield "messages", (model._generate([], None).generations[0].message, MODEL_META)
            yield "values", {
                "__interrupt__": [
                    SimpleNamespace(value={"action_requests": ACTIONS})
                ],
                "messages": [],
            }
        else:
            yield "messages", (model._generate([], None).generations[0].message, MODEL_META)
            yield "values", {"messages": []}

    def get_state(self, config):
        return SimpleNamespace(values={"messages": [AIMessage(content=RESUME_TEXT)]})


# The fake agent + scripted model must survive script re-runs (st.rerun() after
# a decision click re-executes this page): stash them in session_state.
if "_hitl_model" not in st.session_state:
    st.session_state["_hitl_model"] = ScriptedChatModel(responses=[
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name=a["name"], args=a["args"], id=f"tc-{i}")
                for i, a in enumerate(ACTIONS)
            ],
        ),
        AIMessage(content=RESUME_TEXT),
    ])
    st.session_state["_hitl_fake"] = FakeAgent()
model = st.session_state["_hitl_model"]
fake = st.session_state["_hitl_fake"]


agents_mod.get_settings = lambda: SimpleNamespace(recursion_limit=30, ingest_recursion_limit=200)
agents_mod.get_ingest_agent = lambda: st.session_state["_hitl_fake"]

tid, messages = ui_common.init_page("ingest")
store = HistoryStore(OUT_DIR / "history")
ui_common.sidebar("ingest", store)
ui_common.render_history(messages)

msg = st.text_input("Message", key="msg")
if st.button("Submit", key="submit") and msg:
    st.session_state["queued_message"] = msg

if st.session_state.get("queued_message"):
    ui_common.run_turn("ingest", "Ingest", st.session_state["queued_message"], store)
    st.session_state["queued_message"] = ""
elif st.session_state.get("ingest_pending") is not None:
    ui_common.run_turn("ingest", "Ingest", "", store)
"""


def _start_hitl(
    monkeypatch,
    tmp_path: Path,
    actions: list[dict],
    resume_text: str = "Approved and applied.",
    message: str = "delete entities/mlx",
) -> AppTest:
    """Boot the stub page and submit one fresh turn (lands on the interrupt)."""
    monkeypatch.setenv("STUB_ACTIONS", json.dumps(actions))
    monkeypatch.setenv("STUB_RESUME_TEXT", resume_text)
    monkeypatch.setenv("STUB_OUT_DIR", str(tmp_path))
    at = AppTest.from_string(_HITL_STUB)
    at.run()
    at.text_input(key="msg").set_value(message)
    at.button(key="submit").click().run()
    return at


def _call_log(tmp_path: Path) -> list[dict]:
    log = tmp_path / "agent_calls.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestHitlApproveAll:
    def test_approve_all_resumes_with_n_approvals(self, monkeypatch, tmp_path):
        """Acceptance: clicking "Approve all" produces [{"type":"approve"}]*N
        and resumes to a FinalMessage."""
        actions = [DELETE_MLX, DELETE_OTHER]
        at = _start_hitl(monkeypatch, tmp_path, actions, message="delete both")

        # No script exceptions anywhere so far.
        assert len(at.exception) == 0

        # Interrupt: pending marker (no decisions yet) + action cards rendered.
        assert at.session_state["ingest_pending"] == {
            "actions": actions,
            "turn_text": "Checking…",
        }
        rendered = " ".join(m.value for m in at.markdown)
        assert "delete_wiki_page(entities/mlx)" in rendered
        assert "delete_wiki_page(entities/other)" in rendered
        assert "✅ Approve all" in [b.label for b in at.button]
        # Edit path must NOT appear for delete_wiki_page actions.
        assert not any(t.label == "New proposed resolution" for t in at.text_input)

        # Click Approve all: the decision is written to the pending marker and
        # the rerun resumes the turn; one more run finishes the re-render.
        at.button(key="ingest_approve_all").click().run()

        # The click writes the decision, reruns, and the resume completes in
        # the same AppTest interaction (st.rerun() re-executes the page).
        # FinalMessage rendered, pending cleared, transcript persisted.
        assert len(at.exception) == 0
        assert "ingest_pending" not in at.session_state
        assert at.session_state["ingest_messages"] == [
            {"role": "user", "content": "delete both"},
            {"role": "assistant", "content": "Approved and applied."},
        ]
        tid = at.session_state["ingest_thread_id"]
        store = HistoryStore(tmp_path / "history")
        assert store.load("ingest", tid) == [
            {"role": "user", "content": "delete both"},
            {"role": "assistant", "content": "Checking…"},
            {"role": "assistant", "content": "Approved and applied."},
        ]

        # The resume drove Command(resume=...) with CLI-identical decisions.
        log = _call_log(tmp_path)
        assert [entry["call"] for entry in log] == [1, 2]
        assert log[0]["resume"] is None
        assert log[1]["resume"]["decisions"] == [{"type": "approve"}, {"type": "approve"}]
        assert log[1]["config"] == {
            "configurable": {"thread_id": tid},
            "recursion_limit": 200,
        }


class TestHitlRejectAll:
    def test_reject_all_carries_feedback(self, monkeypatch, tmp_path):
        """Acceptance: "Reject all" includes feedback in every decision."""
        actions = [DELETE_MLX]
        at = _start_hitl(monkeypatch, tmp_path, actions)

        at.text_input(key="ingest_reject_feedback").set_value("no good")
        at.button(key="ingest_reject_all").click().run()

        assert len(at.exception) == 0
        assert "ingest_pending" not in at.session_state
        log = _call_log(tmp_path)
        assert log[1]["resume"]["decisions"] == [{"type": "reject", "feedback": "no good"}]
        tid = at.session_state["ingest_thread_id"]
        assert HistoryStore(tmp_path / "history").load("ingest", tid)[-1] == {
            "role": "assistant",
            "content": "Approved and applied.",
        }


class TestHitlEditPath:
    def test_edit_path_produces_edited_action(self, monkeypatch, tmp_path):
        """Acceptance: the edit path (flag_contradiction only) produces the
        edited_action shape with the new proposed_resolution."""
        at = _start_hitl(
            monkeypatch, tmp_path, [CONTRADICTION], resume_text="Flagged."
        )

        # Edit widgets are offered for flag_contradiction.
        assert at.session_state["ingest_pending"]["actions"] == [CONTRADICTION]
        assert any(t.label == "New proposed resolution" for t in at.text_input)
        at.text_input(key="ingest_edit_resolution").set_value("new text")
        at.button(key="ingest_edit").click().run()

        assert len(at.exception) == 0
        assert "ingest_pending" not in at.session_state
        log = _call_log(tmp_path)
        assert log[1]["resume"]["decisions"][0]["type"] == "edit"
        assert log[1]["resume"]["decisions"][0]["edited_action"]["args"] == {
            "slug": "entities/a",
            "proposed_resolution": "new text",
        }
