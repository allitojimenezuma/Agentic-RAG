"""Level 2 — real-LLM DAG trajectory acceptance tier (T5).

Runs the 8 pinned ``TRAJECTORY_TASKS`` contracts from docs/spec.md "L2
real-LLM tier" against the REAL agents (``build_query_agent`` /
``build_ingest_agent`` / ``build_lint_agent`` / ``build_fix_agent``) over tmp
copies of the committed eval corpus and the broken-wiki fixture. Tool names
are recorded in call order from ``result["messages"]``; each run is validated
with ``validate_trajectory``; on failure the FULL trajectory (step index,
tool name, args dict) plus the per-task contract and violations are printed
for human review.

- Aggregate acceptance: ``pass_rate = passed / len(TRAJECTORY_TASKS) >= 0.8``
  (spec "≥ 8/10"; with 8 tasks that means ≥ 7 of 8). If fewer pass, the test
  legitimately FAILS and prints every trajectory + contract + violations —
  the threshold is never loosened.
- Tool-selection accuracy (correct-first-tool rate) is reported via the
  report-only ``expected_first_tool`` field — never asserted.
- The ingest-contradiction task ends in a HITL interrupt; it is asserted
  directly as ``ends_with_interrupt`` (names recorded only up to the
  interrupt — no resume, per spec "asserted directly as ends_with_interrupt").
- Never assert model wording: contracts describe causal order + invariants
  only.

Real-model outcomes are not guaranteed by design — this is an acceptance
test. A sub-0.8 aggregate is the measured result, not something to hack.

Per-run isolation: fresh ``thread_id`` (uuid4), a fresh wiki/raw tmp copy per
task, and each agent is built right before its own run (its ``init_shared_tools``
call targets that run's wiki). Skipped without ``OPENAI_API_KEY``
(``requires_llm`` marker from ``tests/levels/conftest.py``).
"""

from __future__ import annotations

import traceback
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from agentic_rag.agents.fix import build_fix_agent
from agentic_rag.agents.ingest import build_ingest_agent
from agentic_rag.agents.lint import build_lint_agent
from agentic_rag.agents.query import build_query_agent
from agentic_rag.config import Settings
from agentic_rag.io.log import tail_log
from agentic_rag.tools.grounding import build_final_answer
from agentic_rag.wiki.health import health_check
from tests.fixtures.eval_corpus import copy_broken_wiki, copy_eval_raw, copy_eval_wiki
from tests.levels.conftest import requires_llm
from tests.levels.level2.trajectory_contract import (
    TrajectoryContract,
    TrajectoryReport,
    validate_trajectory,
)

# --- Pinned tool inventories (docs/spec.md "L2 real-LLM tier") ----------------
QUERY_TOOLS = ["wiki_command"]

INGEST_TOOLS = [
    "read_source",
    "submit_extraction",
    "wiki_command",
    "create_page",
    "update_page",
    "flag_contradiction",
    "regenerate_index",
    "append_log",
    "delete_wiki_page",
]

LINT_TOOLS = ["wiki_command", "write_lint_report"]

FIX_TOOLS = [
    "wiki_command",
    "edit_wiki_page",
    "add_frontmatter",
    "fix_link",
    "append_related_section",
    "regenerate_index",
    "delete_wiki_page",
]

# "write_tools=fix writes" from the spec table — every fix tool that mutates
# the wiki (they arm the terminal invariant; regenerate_index is the terminal).
FIX_WRITES = [
    "add_frontmatter",
    "fix_link",
    "append_related_section",
    "edit_wiki_page",
    "delete_wiki_page",
]

INGEST_PREREQS = [
    ("read_source", "submit_extraction"),
    ("submit_extraction", "wiki_command"),
    ("wiki_command", "create_page"),
    ("wiki_command", "update_page"),
]

# {RAW} is a per-run placeholder for the ABSOLUTE raw-source path — read_source
# resolves its argument from CWD (repo root has no raw/), so the ingest task
# messages must use the absolute tmp path (orchestrator-verified fact).
TRAJECTORY_TASKS: list[TrajectoryContract] = [
    TrajectoryContract(
        agent="query",
        message="What is MLX?",
        allowed=QUERY_TOOLS,
        prerequisites=[("wiki_command", "wiki_command")],
        required=["wiki_command"],
        max_calls=5,
        expected_first_tool="wiki_command",
    ),
    TrajectoryContract(
        agent="query",
        message="How does MLX relate to Apple Silicon?",
        allowed=QUERY_TOOLS,
        prerequisites=[("wiki_command", "wiki_command")],
        required=["wiki_command"],
        max_calls=5,
        expected_first_tool="wiki_command",
    ),
    TrajectoryContract(
        agent="ingest",
        message="Ingest {RAW}/sample.md",
        allowed=INGEST_TOOLS,
        prerequisites=INGEST_PREREQS,
        required=[
            "read_source",
            "submit_extraction",
            "wiki_command",
            "regenerate_index",
            "append_log",
        ],
        required_any=[("create_page", "update_page")],
        write_tools=["create_page", "update_page"],
        terminal_after=["regenerate_index", "append_log"],
        max_calls=15,
        expected_first_tool="read_source",
    ),
    TrajectoryContract(
        agent="ingest",
        message="Ingest {RAW}/contradiction-source.md",
        allowed=INGEST_TOOLS,
        prerequisites=INGEST_PREREQS[:2],
        required=["flag_contradiction"],
        write_tools=[],
        ends_with_interrupt=True,
        max_calls=15,
        expected_first_tool="read_source",
    ),
    TrajectoryContract(
        agent="lint",
        message=(
            "Run a full wiki health check. Report orphans, contradictions, "
            "missing links, and data gaps."
        ),
        allowed=LINT_TOOLS,
        prerequisites=[("wiki_command", "write_lint_report")],
        required=["wiki_command"],
        max_calls=7,
        expected_first_tool="wiki_command",
    ),
    TrajectoryContract(
        agent="fix",
        message="Fix the missing-frontmatter issue on entities/broken-fm",
        allowed=FIX_TOOLS,
        required=["add_frontmatter"],
        forbidden=["fix_link", "append_related_section", "edit_wiki_page"],
        write_tools=FIX_WRITES,
        terminal_after=["regenerate_index"],
        max_calls=8,
        expected_first_tool="wiki_command",
    ),
    TrajectoryContract(
        agent="fix",
        message="Fix broken links",
        allowed=FIX_TOOLS,
        required=["fix_link"],
        forbidden=["add_frontmatter", "append_related_section", "edit_wiki_page"],
        write_tools=FIX_WRITES,
        terminal_after=["regenerate_index"],
        max_calls=8,
        expected_first_tool="wiki_command",
    ),
    TrajectoryContract(
        agent="fix",
        message="Fix missing-related on entities/lonely",
        allowed=FIX_TOOLS,
        required=["append_related_section"],
        forbidden=["add_frontmatter", "fix_link", "edit_wiki_page"],
        write_tools=FIX_WRITES,
        terminal_after=["regenerate_index"],
        max_calls=8,
        expected_first_tool="wiki_command",
    ),
]

BUILDERS = {
    "query": build_query_agent,
    "ingest": build_ingest_agent,
    "lint": build_lint_agent,
    "fix": build_fix_agent,
}


def _record_calls(result: dict) -> list[dict]:
    """Every tool call made during the run, in order, as {name, args} dicts.

    Iterates ``result["messages"]`` and extends with one entry per
    ``tool_calls`` element (AIMessage.tool_calls + ToolMessage per step).
    For an interrupted run the messages stop at the interrupt, so names are
    naturally recorded only up to it.
    """
    calls: list[dict] = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append({"name": tc["name"], "args": tc.get("args", {})})
    return calls


def _make_env(tmp_path: Path, tag: str, agent: str) -> dict:
    """Fresh, isolated wiki (+ raw when needed) copy for one task run."""
    root = tmp_path / tag
    # Fix tasks run against eval_broken_wiki (pinned defects: broken-fm,
    # linker, lonely) — NEVER the clean eval_wiki (spec table "(broken-wiki
    # copy)"; T1 handoff: "eval_wiki is NEVER used by fix evals"). Query and
    # lint run against the clean corpus; ingest gets a raw copy too.
    if agent == "fix":
        env = {"wiki": copy_broken_wiki(root / "wiki")}
    else:
        env = {"wiki": copy_eval_wiki(root / "wiki")}
    if agent == "ingest":
        env["raw"] = copy_eval_raw(root / "raw")
    return env


def _run_task(task: TrajectoryContract, env: dict) -> dict:
    """Build the real agent right before its run and invoke it once.

    Settings init args win over .env; recursion limit per agent kind
    (ingest uses settings.ingest_recursion_limit). Returns result + recorded
    calls + resolved message + interrupt flag.
    """
    settings = Settings(wiki_path=env["wiki"], agents_md_path=Path("AGENTS.md"))
    agent = BUILDERS[task.agent](settings)
    recursion_limit = (
        settings.ingest_recursion_limit
        if task.agent == "ingest"
        else settings.recursion_limit
    )
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": recursion_limit,
    }
    message = task.message
    if env.get("raw") is not None:
        message = message.replace("{RAW}", str(env["raw"]))
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]}, config=config
    )
    nav_capture = getattr(agent, "_nav_capture", None)
    return {
        "result": result,
        "message": message,
        "calls": _record_calls(result),
        "interrupted": "__interrupt__" in result,
        "nav_slugs": set(nav_capture.navigated) if nav_capture is not None else set(),
    }


def _print_trajectory(message: str, calls: list[dict]) -> None:
    """Full trajectory dump (step index, tool name, args) for human review."""
    print(f"  trajectory ({len(calls)} calls):")
    for i, call in enumerate(calls, start=1):
        print(f"    step {i}: {call['name']} args={call['args']!r}")


def _check_outcome(task: TrajectoryContract, run: dict, env: dict) -> list[str]:
    """On-disk / output assertions that turn 'tool order ok' into 'work done'.

    Contract validation proves the model CALLED the right tools in the right
    order; these checks prove the run actually CHANGED the wiki / produced a
    usable answer. Real-model outcomes are measured results — a violation
    here is a genuine model gap to report, never something to loosen.
    """
    violations: list[str] = []
    if task.agent == "query":
        qa = build_final_answer(run["result"]["messages"], run["nav_slugs"])
        if not qa.answer.strip():
            violations.append("query: final answer is empty")
        if not qa.citations:
            violations.append("query: final answer has zero citations")
        for citation in qa.citations:
            if citation.slug not in run["nav_slugs"]:
                violations.append(
                    f"query: citation {citation.slug!r} not in navigated set "
                    f"{sorted(run['nav_slugs'])}"
                )
        return violations
    if task.agent == "ingest" and task.message.endswith("sample.md"):
        # The created/updated page exists on disk with valid frontmatter,
        # the index picked it up, the log gained an ingest entry, and the
        # copy is still health-clean (no orphans / broken links introduced).
        writes = [c for c in run["calls"] if c["name"] in ("create_page", "update_page")]
        if not writes:
            violations.append("ingest: no create_page/update_page call recorded")
        else:
            for w in writes:
                slug = w["args"].get("slug")
                if not slug:
                    continue
                page_file = Path(env["wiki"]) / f"{slug}.md"
                if not page_file.is_file():
                    violations.append(f"ingest: page file missing on disk: {slug}")
                elif not page_file.read_text(encoding="utf-8").startswith("---"):
                    violations.append(f"ingest: page {slug} lacks YAML frontmatter")
        index_text = (Path(env["wiki"]) / "index.md").read_text(encoding="utf-8")
        if not any(
            w["args"].get("slug") and w["args"]["slug"] in index_text
            for w in writes
        ):
            violations.append("ingest: created/updated slug absent from index.md")
        if not any(e.op == "ingest" for e in tail_log(Path(env["wiki"]))):
            violations.append("ingest: log.md gained no ingest entry")
        report = health_check(Path(env["wiki"]))
        if report.issues:
            kinds = sorted({i.kind for i in report.issues})
            violations.append(f"ingest: health_check reports issues after run: {kinds}")
        return violations
    if task.agent == "lint":
        report_file = Path(env["wiki"]) / f"lint-report-{date.today().isoformat()}.md"
        if not report_file.is_file():
            violations.append("lint: lint-report-YYYY-MM-DD.md not written to wiki/")
        else:
            content = report_file.read_text(encoding="utf-8")
            if len(content.strip()) < 20:
                violations.append("lint: written report is empty/trivial")
        return violations
    if task.agent == "fix":
        # Each fix task must actually clear its seeded defect on disk.
        expected_gone: dict[str, str] = {
            "add_frontmatter": "missing-frontmatter",
            "fix_link": "broken-link",
            "append_related_section": "missing-related",
        }
        for call in run["calls"]:
            kind = expected_gone.get(call["name"])
            if not kind:
                continue
            report = health_check(Path(env["wiki"]))
            if any(i.kind == kind for i in report.issues):
                violations.append(
                    f"fix: {kind} still present in health_check after fixing"
                )
            break
        return violations
    return violations


@requires_llm
def test_real_llm_trajectory_acceptance(tmp_path: Path) -> None:
    """Run all 8 pinned contracts against real agents; aggregate >= 0.8.

    Per task: invoke the real builder over a fresh tmp copy, record tool
    names in order, validate against the contract (``interrupted`` set from
    ``"__interrupt__" in result``), then check the ON-DISK / OUTPUT outcome
    (``_check_outcome``): query answers are non-empty + grounded (citations
    resolve to navigated pages), ingest leaves a valid page + index + log
    entry + health-clean copy, lint writes a real report file, and each fix
    task actually clears its seeded defect. Print the full trajectory +
    violations on failure.
    Acceptance: pass_rate >= 0.8 (>= 7 of 8). Tool-selection accuracy
    (correct-first-tool rate) is reported, never asserted. A sub-0.8
    aggregate is a legitimate, measured failure — the threshold is pinned.
    """
    outcomes: list[dict] = []

    for idx, task in enumerate(TRAJECTORY_TASKS, start=1):
        env = _make_env(tmp_path, f"run{idx}", task.agent)
        print(f"\n[{idx}/{len(TRAJECTORY_TASKS)}] {task.agent}: {task.message}")
        try:
            run = _run_task(task, env)
        except Exception as exc:  # keep the harness alive; count as a failure
            print(f"  RUN EXCEPTION: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            outcomes.append(
                {
                    "task": task,
                    "message": task.message,
                    "passed": False,
                    "violations": [f"run raised {type(exc).__name__}: {exc}"],
                    "calls": [],
                    "interrupted": False,
                }
            )
            continue

        names = [c["name"] for c in run["calls"]]
        report: TrajectoryReport = validate_trajectory(
            task, names, interrupted=run["interrupted"]
        )
        outcome_violations = _check_outcome(task, run, env)
        all_violations = list(report.violations) + outcome_violations
        outcomes.append(
            {
                "task": task,
                "message": run["message"],
                "passed": report.passed and not outcome_violations,
                "violations": all_violations,
                "calls": run["calls"],
                "interrupted": run["interrupted"],
            }
        )
        if report.passed and not outcome_violations:
            print(f"  PASS ({len(names)} calls, interrupted={run['interrupted']})")
        else:
            print(f"  FAIL ({len(names)} calls, interrupted={run['interrupted']})")
            _print_trajectory(run["message"], run["calls"])
            print(f"  contract: {task.model_dump(exclude_none=True)!r}")
            for v in all_violations:
                print(f"    violation: {v}")

    passed = sum(1 for o in outcomes if o["passed"])
    pass_rate = passed / len(TRAJECTORY_TASKS)

    # Tool-selection accuracy — REPORT ONLY, never asserted.
    first_tool_correct = 0
    first_tool_count = 0
    for o in outcomes:
        expected = o["task"].expected_first_tool
        if expected is None:
            continue
        first_tool_count += 1
        if o["calls"] and o["calls"][0]["name"] == expected:
            first_tool_correct += 1
    first_tool_rate = (
        first_tool_correct / first_tool_count if first_tool_count else 0.0
    )

    print("\n" + "=" * 70)
    print(f"AGGREGATE: pass_rate = {passed}/{len(TRAJECTORY_TASKS)} = {pass_rate:.2f}")
    print(f"TOOL-SELECTION ACCURACY (correct-first-tool): {first_tool_correct}/{first_tool_count} = {first_tool_rate:.2f}")
    for o in outcomes:
        status = "PASS" if o["passed"] else "FAIL"
        print(f"  {status}  {o['task'].agent:6s} | {o['message']}")
        if not o["passed"]:
            _print_trajectory(o["message"], o["calls"])
            for v in o["violations"]:
                print(f"    violation: {v}")
    print("=" * 70)

    assert pass_rate >= 0.8, (
        f"aggregate trajectory pass rate {pass_rate:.2f} < 0.8 "
        f"({passed}/{len(TRAJECTORY_TASKS)} passed) — acceptance floor not met"
    )
