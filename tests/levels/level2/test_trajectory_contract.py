"""Level 2 — deterministic unit tests for the trajectory validator (0 LLM).

Synthetic trajectories exercise every validator rule from docs/spec.md
"L2 trajectory contracts" — pass + each failure mode:

- happy-path passes in the pinned shapes (query / ingest / fix)
- forbidden tool hit (not in ``allowed``)
- required tool missing
- required_any group with no member called
- prerequisite order violation (including A never called + B called)
- prerequisite vacuous pass when B never called
- terminal-after-last-write violation
- terminal vacuous pass when no write tool was called
- max_calls breach
- ends_with_interrupt mismatch in both directions (True contract +
  interrupted=False -> fail; False contract + interrupted=True -> pass)

No imports from agentic_rag beyond the contract module; fully headless-safe
(runs with no env vars and no network).
"""

from __future__ import annotations

import pytest

from tests.levels.level2.trajectory_contract import (
    TrajectoryContract,
    TrajectoryReport,
    validate_trajectory,
)

# --- Pinned contract shapes (spec table) -------------------------------------
QUERY_CONTRACT = TrajectoryContract(
    agent="query",
    message="What is MLX?",
    allowed=["wiki_search", "wiki_read_page", "wiki_summary"],
    prerequisites=[("wiki_search", "wiki_read_page")],
    required=["wiki_read_page"],
    max_calls=5,
)

INGEST_CONTRACT = TrajectoryContract(
    agent="ingest",
    message="Ingest sample.md",
    allowed=[
        "read_source",
        "submit_extraction",
        "match_page_tool",
        "wiki_read_page",
        "wiki_scan",
        "wiki_link_graph",
        "create_page",
        "update_page",
        "flag_contradiction",
        "regenerate_index",
        "append_log",
        "delete_wiki_page",
    ],
    prerequisites=[
        ("read_source", "submit_extraction"),
        ("submit_extraction", "match_page_tool"),
        ("match_page_tool", "create_page"),
        ("match_page_tool", "update_page"),
    ],
    required=[
        "read_source",
        "submit_extraction",
        "match_page_tool",
        "regenerate_index",
        "append_log",
    ],
    required_any=[("create_page", "update_page")],
    write_tools=["create_page", "update_page"],
    terminal_after=["regenerate_index", "append_log"],
    max_calls=15,
)

FIX_CONTRACT = TrajectoryContract(
    agent="fix",
    message="Fix the missing-frontmatter issue on entities/broken-fm",
    allowed=[
        "wiki_read_page",
        "edit_wiki_page",
        "add_frontmatter",
        "fix_link",
        "append_related_section",
        "regenerate_index",
        "delete_wiki_page",
    ],
    required=["add_frontmatter"],
    forbidden=["fix_link", "append_related_section", "edit_wiki_page"],
    write_tools=["add_frontmatter", "fix_link", "append_related_section", "edit_wiki_page"],
    terminal_after=["regenerate_index"],
    max_calls=8,
)

CONTRADICTION_CONTRACT = TrajectoryContract(
    agent="ingest",
    message="Ingest contradiction-source.md",
    allowed=INGEST_CONTRACT.allowed,
    prerequisites=[
        ("read_source", "submit_extraction"),
        ("submit_extraction", "match_page_tool"),
    ],
    required=["flag_contradiction"],
    write_tools=[],
    ends_with_interrupt=True,
    max_calls=15,
)


def test_contract_defaults_match_spec():
    """Field defaults are exactly the spec pins (max_calls=30, no flags)."""
    c = TrajectoryContract(agent="a", message="m", allowed=["x"])
    assert c.prerequisites == []
    assert c.required == []
    assert c.required_any == []
    assert c.write_tools == []
    assert c.terminal_after == []
    assert c.forbidden == []
    assert c.max_calls == 30
    assert c.ends_with_interrupt is False
    assert c.expected_first_tool is None


def test_report_shape_on_pass():
    """A passing trajectory yields passed=True and an empty violation list."""
    report = validate_trajectory(QUERY_CONTRACT, ["wiki_search", "wiki_read_page"])
    assert isinstance(report, TrajectoryReport)
    assert report.passed is True
    assert report.violations == []


def test_pinned_positional_signature_stays_callable():
    """The spec signature validate_trajectory(contract, tool_names) works."""
    report = validate_trajectory(QUERY_CONTRACT, ["wiki_search", "wiki_read_page"])
    assert report.passed is True


# --- Happy-path passes --------------------------------------------------------
def test_query_happy_path_passes():
    """wiki_search -> wiki_read_page: order + required satisfied."""
    report = validate_trajectory(
        QUERY_CONTRACT, ["wiki_search", "wiki_read_page"]
    )
    assert report.passed is True
    assert report.violations == []


def test_query_happy_path_with_summary_passes():
    """wiki_search -> wiki_read_page -> wiki_summary is also fine."""
    report = validate_trajectory(
        QUERY_CONTRACT,
        ["wiki_search", "wiki_read_page", "wiki_summary"],
    )
    assert report.passed is True


def test_ingest_happy_path_passes():
    """read_source -> submit_extraction -> match -> create -> index -> log."""
    report = validate_trajectory(
        INGEST_CONTRACT,
        [
            "read_source",
            "submit_extraction",
            "match_page_tool",
            "create_page",
            "regenerate_index",
            "append_log",
        ],
    )
    assert report.passed is True
    assert report.violations == []


def test_fix_happy_path_passes():
    """wiki_read_page -> add_frontmatter -> regenerate_index."""
    report = validate_trajectory(
        FIX_CONTRACT,
        ["wiki_read_page", "add_frontmatter", "regenerate_index"],
    )
    assert report.passed is True
    assert report.violations == []


# --- Failure modes -------------------------------------------------------------
def test_forbidden_tool_hit():
    """A tool outside ``allowed`` is flagged at its 1-based step."""
    report = validate_trajectory(
        QUERY_CONTRACT, ["wiki_search", "delete_wiki_page"]
    )
    assert report.passed is False
    assert report.violations[0] == (
        "forbidden tool 'delete_wiki_page' called at step 2"
    )


def test_forbidden_field_pin_is_flagged_by_allowed_rule():
    """``forbidden`` is documentation (allowed already implies): a tool pinned
    in ``forbidden`` that is outside ``allowed`` is flagged by the allowed
    check."""
    contract = TrajectoryContract(
        agent="query",
        message="x",
        allowed=["wiki_search", "wiki_read_page"],
        forbidden=["delete_wiki_page"],
    )
    report = validate_trajectory(
        contract, ["wiki_search", "delete_wiki_page"]
    )
    assert report.passed is False
    assert "forbidden tool 'delete_wiki_page' called at step 2" in (
        report.violations
    )


def test_required_missing():
    """A required tool never called is reported."""
    report = validate_trajectory(QUERY_CONTRACT, ["wiki_search"])
    assert report.passed is False
    assert report.violations == ["required tool 'wiki_read_page' never called"]


def test_required_any_group_miss():
    """Neither create_page nor update_page called -> group violation."""
    report = validate_trajectory(
        INGEST_CONTRACT,
        [
            "read_source",
            "submit_extraction",
            "match_page_tool",
            "regenerate_index",
            "append_log",
        ],
    )
    assert report.passed is False
    assert report.violations == [
        "required_any group ('create_page', 'update_page'): no member was ever called"
    ]
    # No prerequisite violation in this trajectory (B never called -> vacuous).
    assert not any("prerequisite" in v for v in report.violations)


def test_prerequisite_order_violation():
    """B called before A is a prerequisite violation with step numbers."""
    report = validate_trajectory(
        QUERY_CONTRACT, ["wiki_read_page", "wiki_search"]
    )
    assert report.passed is False
    assert report.violations == [
        "prerequisite violated: first call to 'wiki_read_page' (step 1) "
        "must come after first call to 'wiki_search' (step 2)"
    ]


def test_prerequisite_a_never_called_b_called_violates():
    """B called while A never ran is a violation (not vacuous)."""
    report = validate_trajectory(
        INGEST_CONTRACT,
        ["match_page_tool", "create_page", "regenerate_index", "append_log"],
    )
    assert report.passed is False
    assert any(
        "first call to 'match_page_tool' (step 1) must come after "
        "first call to 'submit_extraction' (step never)" in v
        for v in report.violations
    )


def test_terminal_after_last_write_violation():
    """regenerate_index before the LAST write (create_page) is a violation."""
    report = validate_trajectory(
        INGEST_CONTRACT,
        [
            "read_source",
            "submit_extraction",
            "match_page_tool",
            "regenerate_index",
            "create_page",
            "append_log",
        ],
    )
    assert report.passed is False
    assert report.violations == [
        "terminal tool 'regenerate_index' must appear after the last write (step 5)"
    ]


def test_terminal_missing_tool_violates_when_write_happened():
    """terminal_after tool never called at all, but a write did happen."""
    report = validate_trajectory(
        FIX_CONTRACT, ["wiki_read_page", "add_frontmatter"]
    )
    assert report.passed is False
    assert any(
        "terminal tool 'regenerate_index' must appear after the last write (step 2)"
        in v
        for v in report.violations
    )


def test_terminal_vacuous_when_no_write_tool_called():
    """No write tool called -> terminal_after is vacuously satisfied."""
    contract = TrajectoryContract(
        agent="query",
        message="read-only",
        allowed=["wiki_search", "wiki_read_page"],
        write_tools=["create_page"],
        terminal_after=["regenerate_index"],
    )
    report = validate_trajectory(contract, ["wiki_search", "wiki_read_page"])
    assert report.passed is True
    assert report.violations == []


def test_max_calls_breach():
    """More calls than max_calls is a violation."""
    report = validate_trajectory(
        QUERY_CONTRACT,
        ["wiki_search", "wiki_read_page", "wiki_summary", "wiki_search", "wiki_read_page", "wiki_summary"],
    )
    assert report.passed is False
    assert report.violations == [
        "trajectory length 6 exceeds max_calls 5"
    ]


def test_ends_with_interrupt_true_contract_without_interrupt_fails():
    """ends_with_interrupt=True but run not interrupted -> violation."""
    report = validate_trajectory(
        CONTRADICTION_CONTRACT,
        ["read_source", "submit_extraction", "match_page_tool", "flag_contradiction"],
        interrupted=False,
    )
    assert report.passed is False
    assert report.violations == ["run did not end in a HITL interrupt"]


def test_ends_with_interrupt_true_contract_with_interrupt_passes():
    """ends_with_interrupt=True and run interrupted -> passes."""
    report = validate_trajectory(
        CONTRADICTION_CONTRACT,
        ["read_source", "submit_extraction", "match_page_tool", "flag_contradiction"],
        interrupted=True,
    )
    assert report.passed is True
    assert report.violations == []


def test_ends_with_interrupt_false_contract_with_interrupt_passes():
    """A contract that does NOT pin an interrupt passes even if interrupted."""
    report = validate_trajectory(
        INGEST_CONTRACT,
        [
            "read_source",
            "submit_extraction",
            "match_page_tool",
            "create_page",
            "regenerate_index",
            "append_log",
        ],
        interrupted=True,
    )
    assert report.passed is True
    assert report.violations == []


def test_multiple_violations_accumulate():
    """Several failing rules all land in the violation list, in rule order."""
    report = validate_trajectory(
        QUERY_CONTRACT,
        ["wiki_read_page", "delete_wiki_page", "wiki_search"],
    )
    assert report.passed is False
    assert len(report.violations) == 2  # forbidden + prerequisite (required ok)
    assert report.violations[0] == "forbidden tool 'delete_wiki_page' called at step 2"
    assert "prerequisite violated" in report.violations[1]
