"""Headless HITL auto-responders (T1) — programmatic resume, never ``input()``.

Every test whose scripted/real flow triggers an interrupt (``flag_contradiction``,
``delete_wiki_page``) MUST resume via ``resume_auto`` building
``Command(resume={"decisions": [...]})`` with these helpers. The decision shapes
mirror ``src/agentic_rag/cli.py`` EXACTLY:

- approve: ``[{"type": "approve"}] * max(len(actions), 1)``
- reject:  ``[{"type": "reject", "feedback": feedback}] * max(len(actions), 1)``
- edit:    ``[{"type": "approve"}] * len(actions)`` with ``decisions[index]``
  replaced by ``{"type": "edit", "edited_action": {"name": ...,
  "args": {**args, "proposed_resolution": new_resolution}}}``

Never drive the CLI runner into an interactive decision and never
monkeypatch ``input()`` — tests must not hang in CI/headless environments.
"""

from __future__ import annotations

from langgraph.types import Command


def _extract_actions(state: dict) -> list[dict]:
    """Tolerantly parse ``state["__interrupt__"]`` into the pending actions.

    Handles the shapes produced in practice:
    - interrupt objects with a ``.value`` attribute (langgraph ``Interrupt``);
    - plain dicts carrying ``"action_requests"`` directly (CLI/agent harnesses);
    - anything else / any exception -> ``[]`` (never raise).
    """
    try:
        interrupts = state.get("__interrupt__", [])
    except (AttributeError, TypeError):
        return []
    if not interrupts:
        return []
    interrupt = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    try:
        raw = getattr(interrupt, "value", interrupt)
        actions = raw.get("action_requests", []) if hasattr(raw, "get") else []
        return actions if isinstance(actions, list) else []
    except Exception:
        return []


def auto_decide(
    state: dict,
    choice: str = "approve",
    *,
    feedback: str = "",
    index: int = 0,
    new_resolution: str = "",
) -> list[dict]:
    """Build the resume decisions for a pending interrupt.

    Args:
        state: interrupt-bearing result dict (or agent state values) holding
            ``"__interrupt__"``.
        choice: ``"approve"`` | ``"reject"`` | ``"edit"``.
        feedback: used by ``reject`` (all decisions get the same feedback).
        index: zero-based index of the action to edit (``edit`` only).
        new_resolution: replacement ``proposed_resolution`` (``edit`` only).
    """
    actions = _extract_actions(state)
    n = len(actions)

    if choice == "approve":
        return [{"type": "approve"}] * max(n, 1)
    if choice == "reject":
        return [{"type": "reject", "feedback": feedback}] * max(n, 1)
    if choice == "edit":
        decisions = [{"type": "approve"}] * max(n, 1)
        if actions:
            idx = max(0, min(index, n - 1))
            target = actions[idx] if isinstance(actions[idx], dict) else {}
            target_args = target.get("args", {}) if isinstance(target, dict) else {}
            decisions[idx] = {
                "type": "edit",
                "edited_action": {
                    "name": target.get("name", "flag_contradiction"),
                    "args": {**target_args, "proposed_resolution": new_resolution},
                },
            }
        return decisions
    raise ValueError(f"Unknown HITL choice: {choice!r} (expected approve|reject|edit)")


def resume_auto(
    agent,
    config,
    *,
    state: dict | None = None,
    choice: str = "approve",
    feedback: str = "",
    index: int = 0,
    new_resolution: str = "",
) -> dict:
    """Resume an interrupted agent run programmatically.

    Args:
        agent: the (mocked) agent/graph with ``invoke`` and (optionally)
            ``get_state``.
        config: run config forwarded to ``invoke``.
        state: OPTIONAL interrupt-bearing dict. Pass the pre-resume invoke
            RESULT here when the interrupt lives only in that result — which
            is exactly what langchain's ``HumanInTheLoopMiddleware`` does:
            ``agent.get_state(config).values`` carries ``messages`` but NO
            ``__interrupt__``, so without ``state`` the ``action_requests``
            would be invisible and edit/reject choices would silently degrade
            to ``[{"type": "approve"}]``. When omitted, falls back to
            ``agent.get_state(config).values`` (``{}`` on exception, e.g.
            fakes without ``get_state``).
        choice: ``"approve"`` | ``"reject"`` | ``"edit"``.
        feedback: used by ``reject``.
        index: zero-based index of the action to edit (``edit`` only).
        new_resolution: replacement ``proposed_resolution`` (``edit`` only).

    Builds the decisions with :func:`auto_decide` and invokes
    ``Command(resume={"decisions": [...]})``. Returns the resumed run result.
    """
    if state is None:
        try:
            state = agent.get_state(config).values
        except Exception:
            state = {}
    decisions = auto_decide(
        state, choice, feedback=feedback, index=index, new_resolution=new_resolution
    )
    return agent.invoke(Command(resume={"decisions": decisions}), config=config)
