"""Deterministic unit tests for the headless HITL helpers (T1).

Proves the decision shapes produced by ``auto_decide`` (approve / reject /
edit) mirror ``src/agentic_rag/cli.py`` exactly, that interrupt parsing is
tolerant of the real-world state shapes, and that ``resume_auto`` builds a
``Command(resume={"decisions": [...]})`` and never calls ``input()``.
Zero LLM calls, zero network.
"""

from __future__ import annotations

from langgraph.types import Command

from tests.fixtures.eval_hitl import auto_decide, resume_auto


class _InterruptLike:
    """Mimics ``langgraph.types.Interrupt``: value carried in ``.value``."""

    def __init__(self, value):
        self.value = value


def _state_with_actions(actions: list[dict]) -> dict:
    return {"__interrupt__": [_InterruptLike({"action_requests": actions, "review_configs": []})]}


def _contradiction_action(name: str = "flag_contradiction", resolution: str = "Merge both claims.") -> dict:
    return {
        "name": name,
        "args": {
            "page_slug": "entities/mlx",
            "existing_claim": "MLX is developed by Apple",
            "new_claim": "MLX is developed by Google",
            "proposed_resolution": resolution,
        },
    }


# --- approve ------------------------------------------------------------------


def test_approve_mirrors_cli_for_multiple_actions() -> None:
    state = _state_with_actions([_contradiction_action(), _contradiction_action()])
    assert auto_decide(state, "approve") == [{"type": "approve"}, {"type": "approve"}]


def test_approve_with_no_actions_yields_single_approve() -> None:
    """CLI shape: ``max(len(actions), 1)`` — never an empty decisions list."""
    assert auto_decide({}, "approve") == [{"type": "approve"}]
    assert auto_decide({"__interrupt__": []}, "approve") == [{"type": "approve"}]


# --- reject -------------------------------------------------------------------


def test_reject_carries_feedback_on_every_decision() -> None:
    state = _state_with_actions([_contradiction_action(), _contradiction_action()])
    decisions = auto_decide(state, "reject", feedback="Keep the original claim.")
    assert decisions == [
        {"type": "reject", "feedback": "Keep the original claim."},
        {"type": "reject", "feedback": "Keep the original claim."},
    ]


def test_reject_with_no_actions_yields_single_reject() -> None:
    assert auto_decide({}, "reject", feedback="n/a") == [{"type": "reject", "feedback": "n/a"}]


# --- edit ---------------------------------------------------------------------


def test_edit_replaces_one_decision_and_keeps_others_approved() -> None:
    actions = [_contradiction_action(resolution="Original."), _contradiction_action(resolution="Second.")]
    decisions = auto_decide(_state_with_actions(actions), "edit", index=1, new_resolution="Edited text.")
    assert decisions[0] == {"type": "approve"}
    assert decisions[1] == {
        "type": "edit",
        "edited_action": {
            "name": "flag_contradiction",
            "args": {
                "page_slug": "entities/mlx",
                "existing_claim": "MLX is developed by Apple",
                "new_claim": "MLX is developed by Google",
                "proposed_resolution": "Edited text.",
            },
        },
    }


def test_edit_default_index_targets_first_action() -> None:
    actions = [_contradiction_action(resolution="Original.")]
    decisions = auto_decide(_state_with_actions(actions), "edit", new_resolution="New.")
    assert decisions[0]["edited_action"]["args"]["proposed_resolution"] == "New."
    assert decisions[0]["edited_action"]["name"] == "flag_contradiction"


def test_unknown_choice_raises() -> None:
    try:
        auto_decide({}, "maybe")
    except ValueError as exc:
        assert "approve|reject|edit" in str(exc)
    else:  # pragma: no cover - guard against silent fallback
        raise AssertionError("unknown choice must raise, not fall back")


# --- tolerant interrupt parsing -----------------------------------------------


def test_parses_langgraph_interrupt_object() -> None:
    """The canonical shape: a list of interrupt objects with ``.value``."""
    actions = [_contradiction_action()]
    assert auto_decide(_state_with_actions(actions)) == [{"type": "approve"}]


def test_parses_plain_dict_interrupt() -> None:
    """CLI/agent-harness shape: the dict is used directly when no ``.value``."""
    state = {"__interrupt__": {"action_requests": [_contradiction_action()]}}
    assert auto_decide(state) == [{"type": "approve"}]


def test_garbage_interrupt_degrades_to_empty_actions() -> None:
    """Any exception during parsing must degrade to [] — never raise/hang."""
    for state in (
        {"__interrupt__": "garbage"},
        {"__interrupt__": 42},
        {"__interrupt__": [object()]},
        {"__interrupt__": {"action_requests": "not-a-list"}},
    ):
        assert auto_decide(state) == [{"type": "approve"}]


# --- resume_auto --------------------------------------------------------------


class _FakeAgent:
    """Minimal agent double: get_state(config).values + invoke(Command, config)."""

    def __init__(self, state: dict, result: dict):
        self._state = state
        self._result = result
        self.last_command: Command | None = None

    def get_state(self, config):
        return type("StateSnapshot", (), {"values": self._state})()

    def invoke(self, command, config=None):
        self.last_command = command
        return self._result


def test_resume_auto_builds_command_with_decisions() -> None:
    state = _state_with_actions([_contradiction_action(), _contradiction_action()])
    agent = _FakeAgent(state, {"messages": []})
    result = resume_auto(agent, {"configurable": {"thread_id": "t1"}}, choice="approve")

    assert result == {"messages": []}
    assert agent.last_command is not None
    decisions = getattr(agent.last_command, "resume")["decisions"]
    assert decisions == [{"type": "approve"}, {"type": "approve"}]


def test_resume_auto_edit_uses_new_resolution() -> None:
    state = _state_with_actions([_contradiction_action(resolution="Old.")])
    agent = _FakeAgent(state, {"messages": []})
    resume_auto(agent, {}, choice="edit", new_resolution="Approved-as-edited.")

    decisions = getattr(agent.last_command, "resume")["decisions"]
    assert decisions[0]["type"] == "edit"
    assert (
        decisions[0]["edited_action"]["args"]["proposed_resolution"] == "Approved-as-edited."
    )


def test_resume_auto_with_state_issues_edit_decision() -> None:
    """``state=result`` (interrupt-bearing invoke result) -> the CLI EDIT shape
    reaches invoke: name + args + human ``proposed_resolution``."""
    result = _state_with_actions([_contradiction_action(resolution="Old.")])
    agent = _FakeAgent({"messages": []}, {"messages": []})
    resume_auto(agent, {}, state=result, choice="edit", index=0, new_resolution="X")

    decisions = getattr(agent.last_command, "resume")["decisions"]
    assert decisions == [
        {
            "type": "edit",
            "edited_action": {
                "name": "flag_contradiction",
                "args": {
                    "page_slug": "entities/mlx",
                    "existing_claim": "MLX is developed by Apple",
                    "new_claim": "MLX is developed by Google",
                    "proposed_resolution": "X",
                },
            },
        }
    ]


def test_resume_auto_with_state_issues_reject_decision() -> None:
    """``state=result`` + ``choice="reject"`` -> every decision carries feedback."""
    result = _state_with_actions([_contradiction_action()])
    agent = _FakeAgent({"messages": []}, {"messages": []})
    resume_auto(agent, {}, state=result, choice="reject", feedback="Keep it.")

    decisions = getattr(agent.last_command, "resume")["decisions"]
    assert decisions == [{"type": "reject", "feedback": "Keep it."}]


def test_resume_auto_without_state_still_approves() -> None:
    """Backward compat: no ``state`` kwarg and no ``get_state`` -> the approve
    fallback shape, never an empty/None decisions list."""
    agent = _FakeAgent({"messages": []}, {"messages": []})
    agent.get_state = None  # type: ignore[method-assign]
    resume_auto(agent, {}, choice="approve")

    decisions = getattr(agent.last_command, "resume")["decisions"]
    assert decisions == [{"type": "approve"}]


def test_resume_auto_tolerates_missing_get_state() -> None:
    """Fakes without ``get_state`` degrade to an empty state, still resume."""
    agent = _FakeAgent({"__interrupt__": "no-actions"}, {"messages": []})
    agent.get_state = None  # type: ignore[method-assign]
    result = resume_auto(agent, {})
    assert result == {"messages": []}
