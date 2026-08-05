"""Level 2 — pure DAG trajectory contract (T5).

The deterministic validator backing the L2 real-LLM acceptance tier.
Purely declarative: pydantic models + one pure function. No imports from
agentic_rag (pydantic only), no I/O, no LLM, no network. The DAG logic
itself is verified independently of any model by
``tests/levels/level2/test_trajectory_contract.py``.

Semantics mirror docs/spec.md "Interfaces → L2 trajectory contracts" verbatim:

- **forbidden**: every called tool must be in ``allowed`` (``forbidden`` is
  an explicit documentation pin — the allowed check already implies it).
- **prerequisites**: edge ``(A, B)`` requires the FIRST call to ``B`` to come
  after the FIRST call to ``A``; vacuous when ``B`` is never called (calling
  ``B`` while ``A`` was never called is a violation).
- **required**: each name must appear at least once.
- **required_any**: each group must contribute at least one called member.
- **terminal**: each ``terminal_after`` tool must appear at least once AFTER
  the LAST call to any ``write_tools`` member; vacuous when no write tool was
  called.
- **max_calls**: total tool calls must not exceed ``max_calls``.
- **ends_with_interrupt**: a contract that pins a HITL interrupt fails unless
  the run reported an interrupt (``interrupted=True``).

Each violation is one human-readable string in ``TrajectoryReport.violations``.
"""

from __future__ import annotations

from pydantic import BaseModel


class TrajectoryContract(BaseModel):
    """Declarative contract for one real-agent tool trajectory."""

    agent: str
    message: str
    allowed: list[str]                      # tools the agent may call; anything else -> violation
    prerequisites: list[tuple[str, str]] = []   # edge (A, B): first call to B must come after
                                                # the FIRST call to A (vacuous if B never called)
    required: list[str] = []                # each must appear >= 1
    required_any: list[tuple[str, ...]] = []    # each group: at least one member must appear
    write_tools: list[str] = []             # tools that arm the terminal invariant
    terminal_after: list[str] = []          # each must appear >= 1 AFTER the LAST call to any
                                            # write_tool (vacuous if no write tool was called)
    forbidden: list[str] = []               # explicit extra prohibitions (allowed already implies)
    max_calls: int = 30
    ends_with_interrupt: bool = False       # run must terminate in a HITL interrupt
    # REPORT-ONLY: expected first tool for the tool-selection accuracy metric
    # in the acceptance tier. Never asserted by any test.
    expected_first_tool: str | None = None


class TrajectoryReport(BaseModel):
    """Result of validating one trajectory against a contract."""

    passed: bool
    violations: list[str]                   # human-readable, one per failing rule


def validate_trajectory(
    contract: TrajectoryContract,
    tool_names: list[str],
    *,
    interrupted: bool = False,
) -> TrajectoryReport:
    """Validate an ordered tool-call list against ``contract``.

    Pure and deterministic. ``interrupted`` reports whether the agent run
    terminated in a HITL interrupt (``"__interrupt__" in result``); it is
    only consulted when ``contract.ends_with_interrupt`` is set — a contract
    that does NOT pin an interrupt passes regardless of the flag.

    The spec-pinned positional signature ``validate_trajectory(contract,
    tool_names)`` stays callable; ``interrupted`` is keyword-only.
    """
    violations: list[str] = []

    for i, name in enumerate(tool_names, start=1):
        if name not in contract.allowed:
            violations.append(f"forbidden tool '{name}' called at step {i}")

    for a, b in contract.prerequisites:
        ia = tool_names.index(a) if a in tool_names else None
        ib = tool_names.index(b) if b in tool_names else None
        if ib is None:
            continue  # B never called -> vacuous pass
        if ia is None or ib <= ia:
            first_a = ia + 1 if ia is not None else "never"
            violations.append(
                f"prerequisite violated: first call to '{b}' (step {ib + 1}) "
                f"must come after first call to '{a}' (step {first_a})"
            )

    for name in contract.required:
        if name not in tool_names:
            violations.append(f"required tool '{name}' never called")

    for group in contract.required_any:
        if not any(m in tool_names for m in group):
            members = ", ".join(repr(m) for m in group)
            violations.append(
                f"required_any group ({members}): no member was ever called"
            )

    # Terminal invariant: w = last index of any write_tool; each t in
    # terminal_after must appear at an index > w. No write tool called ->
    # vacuous pass.
    last_write = -1
    for i, name in enumerate(tool_names):
        if name in contract.write_tools:
            last_write = i
    if last_write >= 0:
        for t in contract.terminal_after:
            appears_after = any(
                i > last_write for i, n in enumerate(tool_names) if n == t
            )
            if not appears_after:
                violations.append(
                    f"terminal tool '{t}' must appear after the last write "
                    f"(step {last_write + 1})"
                )

    if len(tool_names) > contract.max_calls:
        violations.append(
            f"trajectory length {len(tool_names)} exceeds max_calls {contract.max_calls}"
        )

    if contract.ends_with_interrupt and not interrupted:
        violations.append("run did not end in a HITL interrupt")

    return TrajectoryReport(passed=not violations, violations=violations)
