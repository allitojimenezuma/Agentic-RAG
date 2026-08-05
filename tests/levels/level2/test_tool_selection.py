"""Level 2 — deterministic tool-selection invariants (0 LLM, fully headless).

Pins the per-agent tool whitelists, causal ordering, the fix kind→tool map,
and the HITL resume contract from docs/spec.md "Interfaces" + "Conventions":

- query:  scripted runs may only call {wiki_search, wiki_read_page,
  wiki_summary} — NEVER write tools.
- ingest: happy path must be read_source -> submit_extraction (exactly once)
  -> match_page_tool -> create_page -> regenerate_index -> append_log, with
  regenerate_index + append_log as the LAST two tool calls.
- lint:   run_health_check exactly once and BEFORE write_lint_report; never a
  foreign write tool from the write-tools pin.
- fix:    kind→tool map is pinned (missing-frontmatter->add_frontmatter,
  broken-link->fix_link, missing-related->append_related_section,
  missing-index/orphan/empty/stale -> NO fix tool: a final answer with zero
  tool calls).
- HITL:   flag_contradiction / delete_wiki_page runs interrupt and MUST be
  resumed via resume_auto (approve) — never input(), never patch.

Every run uses ScriptedChatModel + build_agent (pattern from the integration
suite); no Settings, no network, no real LLM. Each run gets its own
thread_id so the in-memory checkpointer never shares state across runs.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import (
    build_fix_prompt,
    build_ingest_prompt,
    build_lint_prompt,
    build_query_prompt,
)
from agentic_rag.tools.fix_tools import (
    add_frontmatter,
    append_related_section,
    edit_wiki_page,
    fix_link,
)
from agentic_rag.tools.ingest_grounding import submit_extraction
from agentic_rag.tools.ingest_tools import (
    append_log,
    create_page,
    delete_wiki_page,
    flag_contradiction,
    read_source,
    update_page,
)
from agentic_rag.tools.lint_tools import run_health_check, write_lint_report
from agentic_rag.tools.nav import (
    regenerate_index,
    wiki_link_graph,
    wiki_read_page,
    wiki_scan,
    wiki_search,
    wiki_summary,
)
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.match import match_page_tool
from tests.fixtures.eval_corpus import copy_broken_wiki
from tests.fixtures.eval_hitl import resume_auto
from tests.fixtures.fake_llm import ScriptedChatModel

# --- Pinned tool inventories (docs/spec.md "L2 real-LLM tier") ---------------
QUERY_TOOL_NAMES = {"wiki_search", "wiki_read_page", "wiki_summary"}

INGEST_TOOL_NAMES = {
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
}

LINT_TOOL_NAMES = {"run_health_check", "wiki_link_graph", "wiki_read_page", "write_lint_report"}

FIX_TOOL_NAMES = {
    "wiki_read_page",
    "edit_wiki_page",
    "add_frontmatter",
    "fix_link",
    "append_related_section",
    "regenerate_index",
    "delete_wiki_page",
}

# Write-tools set (path guard) — no agent may cross its boundary.
WRITE_TOOL_NAMES = {
    "create_page",
    "update_page",
    "delete_wiki_page",
    "write_lint_report",
    "add_frontmatter",
    "fix_link",
    "append_related_section",
}

INGEST_TOOLS = [
    read_source,
    submit_extraction,
    match_page_tool,
    wiki_read_page,
    wiki_scan,
    wiki_link_graph,
    create_page,
    update_page,
    flag_contradiction,
    regenerate_index,
    append_log,
    delete_wiki_page,
]

LINT_TOOLS = [run_health_check, wiki_link_graph, wiki_read_page, write_lint_report]

FIX_TOOLS = [
    wiki_read_page,
    edit_wiki_page,
    add_frontmatter,
    fix_link,
    append_related_section,
    regenerate_index,
    delete_wiki_page,
]


# --- HITL middleware, mirrored from src/agentic_rag/agents/{ingest,fix}.py ---
def _ingest_middleware() -> list:
    """Ingest-agent middleware: HITL on delete and contradictions."""
    return [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_wiki_page": {"allowed_decisions": ["approve", "reject"]},
                "flag_contradiction": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
            }
        )
    ]


def _fix_middleware() -> list:
    """Fix-agent middleware: HITL on delete only (mirrors fix.py)."""
    return [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_wiki_page": {"allowed_decisions": ["approve", "reject"]},
            }
        )
    ]


def _all_tool_calls(result) -> list[dict]:
    """Collect every tool call made during the agent run, in order."""
    calls: list[dict] = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(tc)
    return calls


def _run(agent, user_content: str, config: dict) -> dict:
    """Invoke a built agent with a plain user message."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]}, config=config
    )


# --- query: read-only whitelist ------------------------------------------------
class TestQueryToolWhitelist:
    """Query agent may ONLY navigate (wiki_search -> wiki_read_page), never write."""

    def test_scripted_run_calls_only_nav_tools(self, eval_wiki):
        """wiki_search -> wiki_read_page -> final answer; nothing else."""
        init_shared_tools(str(eval_wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="wiki_search",
                            args={"query": "MLX"},
                            id="tc-1",
                        )
                    ],
                ),
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
                AIMessage(content="MLX is a machine learning framework by Apple."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=[wiki_search, wiki_read_page, wiki_summary],
            system_prompt=build_query_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "What is MLX?", config)

        names = [c["name"] for c in _all_tool_calls(result)]
        assert set(names) <= QUERY_TOOL_NAMES
        # never a write tool from the write-tools pin
        assert set(names).isdisjoint(WRITE_TOOL_NAMES)
        # search precedes the page read
        assert names[0] == "wiki_search"
        assert names.index("wiki_search") < names.index("wiki_read_page")


# --- ingest: causal order + terminal invariants --------------------------------
class TestIngestToolInvariants:
    """Ingest happy path: read_source -> submit_extraction -> match_page_tool
    -> create_page -> regenerate_index -> append_log, with terminal writes last."""

    def test_happy_path_order_and_last_two_tools(self, eval_env):
        """Order of first occurrences is pinned; append_log closes the run."""
        wiki_path, raw_path = eval_env
        init_shared_tools(str(wiki_path))

        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_source",
                            args={"source_path": str(raw_path / "sample.md")},
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="submit_extraction",
                            args={
                                "entities": [
                                    {
                                        "name": "Samplecorp",
                                        "type": "organization",
                                        "summary": (
                                            "A sample organization mentioned "
                                            "in the source."
                                        ),
                                        "sources": ["sample.md"],
                                    }
                                ],
                                "concepts": [],
                                "contradictions": [],
                            },
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="match_page_tool",
                            args={"name": "Samplecorp", "page_type": "entity"},
                            id="tc-3",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="create_page",
                            args={
                                "slug": "entities/samplecorp",
                                "page_type": "entity",
                                "title": "Samplecorp",
                                "content": (
                                    "# Samplecorp\n\n"
                                    "A sample organization from sample.md.\n\n"
                                    "## Related\n\n"
                                    "- [[MLX]]"
                                ),
                                "sources": ["sample.md"],
                            },
                            id="tc-4",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="regenerate_index", args={}, id="tc-5")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="append_log",
                            args={"op": "ingest", "title": "sample.md"},
                            id="tc-6",
                        )
                    ],
                ),
                AIMessage(content="Ingested sample.md."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=INGEST_TOOLS,
            system_prompt=build_ingest_prompt("# Test schema"),
            middleware=_ingest_middleware(),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "Ingest raw/sample.md", config)

        names = [c["name"] for c in _all_tool_calls(result)]
        assert set(names) <= INGEST_TOOL_NAMES

        # Causal order of FIRST occurrences.
        for earlier, later in [
            ("read_source", "submit_extraction"),
            ("submit_extraction", "match_page_tool"),
            ("match_page_tool", "create_page"),
            ("create_page", "regenerate_index"),
            ("regenerate_index", "append_log"),
        ]:
            assert names.index(earlier) < names.index(later), (
                f"{earlier} must precede {later}; got {names}"
            )

        # submit_extraction exactly once.
        assert names.count("submit_extraction") == 1

        # Terminal invariant: regenerate_index + append_log close the run.
        assert names[-2:] == ["regenerate_index", "append_log"]

        # Disk effects via the deterministic engine: page, index, log.
        # (index.md lists display titles — compare case-insensitively)
        assert (wiki_path / "entities" / "samplecorp.md").is_file()
        index_text = (wiki_path / "index.md").read_text(encoding="utf-8")
        assert "samplecorp" in index_text.lower()
        log_text = (wiki_path / "log.md").read_text(encoding="utf-8")
        assert "ingest |" in log_text and "sample.md" in log_text


# --- lint: run_health_check first, report last, no foreign writes --------------
class TestLintToolInvariants:
    """Lint agent: run_health_check exactly once, before write_lint_report."""

    def test_scripted_run_order_and_no_foreign_writes(self, eval_wiki):
        """run_health_check -> write_lint_report; lint never touches foreign writes."""
        init_shared_tools(str(eval_wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="run_health_check", args={}, id="tc-1")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="write_lint_report",
                            args={"report": "# Lint Report\n\nNo issues found.\n"},
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(content="Lint complete."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=LINT_TOOLS,
            system_prompt=build_lint_prompt("# Test schema"),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "Run a full health check.", config)

        names = [c["name"] for c in _all_tool_calls(result)]
        assert set(names) <= LINT_TOOL_NAMES
        assert names.count("run_health_check") == 1
        assert names.index("run_health_check") < names.index("write_lint_report")
        # Never a write tool from the path-guard pin other than lint's own report.
        assert set(names).isdisjoint(WRITE_TOOL_NAMES - {"write_lint_report"})


# --- fix: kind -> tool map ------------------------------------------------------
FIX_KIND_EXPECTATIONS: list[tuple] = [
    (
        "missing-frontmatter",
        "add_frontmatter",
        {"fix_link", "append_related_section", "edit_wiki_page"},
        {
            "slug": "entities/broken-fm",
            "title": "Broken FM",
            "page_type": "entity",
        },
        "entities/broken-fm.md",
    ),
    (
        "broken-link",
        "fix_link",
        {"add_frontmatter", "append_related_section", "edit_wiki_page"},
        {
            "slug": "entities/linker",
            "old_target": "Nonexistent Page",
            "new_target": "Broken FM",
        },
        "entities/linker.md",
    ),
    (
        "missing-related",
        "append_related_section",
        {"add_frontmatter", "fix_link", "edit_wiki_page"},
        {"slug": "entities/lonely", "links": ["Broken FM", "Linker"]},
        "entities/lonely.md",
    ),
]


def _assert_fix_landed(expected_tool: str, wiki: Path) -> None:
    """Verify the pinned fix actually changed the broken-wiki copy on disk."""
    if expected_tool == "add_frontmatter":
        assert (wiki / "entities" / "broken-fm.md").read_text(
            encoding="utf-8"
        ).startswith("---")
    elif expected_tool == "fix_link":
        assert "[[Nonexistent Page]]" not in (
            wiki / "entities" / "linker.md"
        ).read_text(encoding="utf-8")
    elif expected_tool == "append_related_section":
        assert "## Related" in (wiki / "entities" / "lonely.md").read_text(
            encoding="utf-8"
        )


class TestFixKindToolMap:
    """Pinned kind -> tool map: the right tool runs, forbidden ones never do."""

    @pytest.mark.parametrize(
        "kind,expected_tool,forbidden,args,relpath",
        FIX_KIND_EXPECTATIONS,
        ids=[e[0] for e in FIX_KIND_EXPECTATIONS],
    )
    def test_kind_uses_its_pinned_tool_only(
        self, kind, expected_tool, forbidden, args, relpath, tmp_path
    ):
        """One kind -> exactly its pinned fix tool, never a sibling/forbidden one."""
        wiki = copy_broken_wiki(tmp_path)
        init_shared_tools(str(wiki))

        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name=expected_tool, args=args, id="tc-1")
                    ],
                ),
                AIMessage(content=f"Fixed {kind} on {relpath}."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=FIX_TOOLS,
            system_prompt=build_fix_prompt("# Test schema"),
            middleware=_fix_middleware(),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(
            agent,
            f"Fix these lint issues:\n- [{kind}] {relpath}",
            config,
        )

        names = [c["name"] for c in _all_tool_calls(result)]
        assert names == [expected_tool]
        assert set(names) <= FIX_TOOL_NAMES
        assert set(names).isdisjoint(forbidden)
        assert result["messages"][-1].content == f"Fixed {kind} on {relpath}."
        _assert_fix_landed(expected_tool, wiki)

    @pytest.mark.parametrize("kind", ["missing-index", "orphan", "empty", "stale"])
    def test_no_fix_tool_kinds_finish_with_final_answer(self, kind, tmp_path):
        """These kinds have NO fix tool: the run ends with a plain answer."""
        wiki = copy_broken_wiki(tmp_path)
        init_shared_tools(str(wiki))

        final = f"{kind} requires human review; no automated fix."
        model = ScriptedChatModel(responses=[AIMessage(content=final)])
        agent = build_agent(
            model=model,
            tools=FIX_TOOLS,
            system_prompt=build_fix_prompt("# Test schema"),
            middleware=_fix_middleware(),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, f"Fix issues:\n- [{kind}] entities/x", config)

        assert _all_tool_calls(result) == []
        assert result["messages"][-1].content == final
        # run terminated normally and wrote nothing
        assert not (wiki / "entities" / "x.md").exists()


# --- HITL: interrupt + resume_auto ----------------------------------------------
class TestHitlResume:
    """flag_contradiction / delete_wiki_page interrupt; resume_auto approves."""

    def test_contradiction_interrupts_then_resume_approve_appends_log(self, eval_env):
        """read_source -> submit_extraction -> match_page_tool ->
        flag_contradiction (INTERRUPT) -> resume approve -> regenerate_index +
        append_log; the page is kept and the log gains an entry."""
        wiki_path, raw_path = eval_env
        init_shared_tools(str(wiki_path))

        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="read_source",
                            args={
                                "source_path": str(raw_path / "contradiction-source.md")
                            },
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="submit_extraction",
                            args={
                                "entities": [
                                    {
                                        "name": "MLX",
                                        "type": "software",
                                        "summary": (
                                            "MLX is developed by Google for "
                                            "Google TPU clusters."
                                        ),
                                        "sources": ["contradiction-source.md"],
                                    }
                                ],
                                "concepts": [],
                                "contradictions": [
                                    {
                                        "page_slug": "entities/mlx",
                                        "existing_claim": (
                                            "MLX was developed by Apple for "
                                            "Apple Silicon."
                                        ),
                                        "new_claim": (
                                            "MLX was developed by Google for "
                                            "Google TPU clusters."
                                        ),
                                        "proposed_resolution": (
                                            "Update entities/mlx to the "
                                            "Google/TPU claim."
                                        ),
                                    }
                                ],
                            },
                            id="tc-2",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="match_page_tool",
                            args={"name": "MLX", "page_type": "entity"},
                            id="tc-3",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="flag_contradiction",
                            args={
                                "page_slug": "entities/mlx",
                                "existing_claim": (
                                    "MLX was developed by Apple for Apple Silicon."
                                ),
                                "new_claim": (
                                    "MLX was developed by Google for Google TPU clusters."
                                ),
                                "proposed_resolution": (
                                    "Update entities/mlx to the Google/TPU claim."
                                ),
                            },
                            id="tc-4",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(name="regenerate_index", args={}, id="tc-5")
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="append_log",
                            args={
                                "op": "ingest",
                                "title": "contradiction-source.md",
                            },
                            id="tc-6",
                        )
                    ],
                ),
                AIMessage(content="Contradiction handled."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=INGEST_TOOLS,
            system_prompt=build_ingest_prompt("# Test schema"),
            middleware=_ingest_middleware(),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "Ingest raw/contradiction-source.md", config)

        # The run stopped at the HITL interrupt, flag_contradiction requested.
        assert "__interrupt__" in result
        interrupted_names = [c["name"] for c in _all_tool_calls(result)]
        assert "flag_contradiction" in interrupted_names
        assert "regenerate_index" not in interrupted_names

        # Programmatic approve resume — never input().
        resumed = resume_auto(agent, config, choice="approve")
        assert resumed["messages"][-1].content == "Contradiction handled."

        resumed_names = [c["name"] for c in _all_tool_calls(resumed)]
        assert set(resumed_names) <= INGEST_TOOL_NAMES
        # Approve lets the flow finish with the terminal pair.
        assert resumed_names[-2:] == ["regenerate_index", "append_log"]

        # Page kept, index consistent, log gained the ingest entry.
        assert (wiki_path / "entities" / "mlx.md").is_file()
        index_text = (wiki_path / "index.md").read_text(encoding="utf-8")
        assert "mlx" in index_text.lower()
        log_text = (wiki_path / "log.md").read_text(encoding="utf-8")
        assert "ingest |" in log_text and "contradiction-source.md" in log_text

    def test_delete_interrupts_then_resume_approve_deletes_page(self, eval_wiki):
        """delete_wiki_page -> INTERRUPT -> resume approve -> page file gone."""
        init_shared_tools(str(eval_wiki))

        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="delete_wiki_page",
                            args={"slug": "entities/azure"},
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(content="Deleted entities/azure."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=INGEST_TOOLS,
            system_prompt=build_ingest_prompt("# Test schema"),
            middleware=_ingest_middleware(),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "Delete entities/azure", config)

        assert "__interrupt__" in result
        assert "delete_wiki_page" in [
            c["name"] for c in _all_tool_calls(result)
        ]
        # Still on disk until the human approves.
        assert (eval_wiki / "entities" / "azure.md").is_file()

        resumed = resume_auto(agent, config, choice="approve")
        assert resumed["messages"][-1].content == "Deleted entities/azure."
        assert not (eval_wiki / "entities" / "azure.md").exists()
