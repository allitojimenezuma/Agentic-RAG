"""Level 3 — end-to-end contradiction HITL handling (T7).

Scripted ingest flows over the committed ``contradiction-source.md`` fixture
run the FULL headless loop: read_source -> submit_extraction ->
match_page_tool -> flag_contradiction -> INTERRUPT -> ``resume_auto``
(approve / reject / edit) -> terminal writes (regenerate_index +
append_log). Mirrors the T3 ``TestHitlResume`` pattern from
``tests/levels/level2/test_tool_selection.py`` exactly: the same
``INGEST_TOOLS`` whitelist, the same ``_ingest_middleware()`` HITL shape, the same scripted style. All
three resume variants pass the pre-resume invoke RESULT as ``state=result``
because langchain's ``HumanInTheLoopMiddleware`` keeps ``__interrupt__`` (with
``action_requests``) in the invoke result, while ``agent.get_state(config)
.values`` carries only ``messages``. Never ``input()``, never
``patch("builtins.input")`` (headless HITL hard rule).

Resume variants (one fresh ``eval_env`` per test):
- approve: the flagged page is updated to the new claim, the index is
  regenerated, and the log gains an ``ingest`` entry.
- reject:  page bytes are UNCHANGED vs before the resume; the run still
  regenerates the index and logs the outcome.
- edit:    the edited ``flag_contradiction`` ToolCall re-executes with the
  human's new ``proposed_resolution`` (visible in the resulting
  ToolMessage) and the page adopts the merged resolution.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, ToolCall, ToolMessage

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_ingest_prompt
from agentic_rag.tools.extraction import submit_extraction
from agentic_rag.tools.ingest_tools import (
    append_log,
    create_page,
    delete_wiki_page,
    flag_contradiction,
    read_source,
    update_page,
)
from agentic_rag.tools.nav import (
    regenerate_index,
    wiki_link_graph,
    wiki_read_page,
    wiki_scan,
)
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.match import match_page_tool
from tests.fixtures.eval_hitl import resume_auto
from tests.fixtures.fake_llm import ScriptedChatModel
from tests.levels.conftest import eval_env

# --- Pinned tool inventory + HITL middleware (mirror level2 T3 exactly) --------
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


def _all_tool_calls(result) -> list[dict]:
    """Collect every tool call made during the agent run, in order."""
    calls: list[dict] = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(tc)
    return calls


# --- Scripted contradiction flow ----------------------------------------------
EXISTING_CLAIM = "MLX was developed by Apple for Apple Silicon."
NEW_CLAIM = "MLX was developed by Google for Google TPU clusters."
PROPOSED_RESOLUTION = "Update entities/mlx to the Google/TPU claim."

SUBMIT_ARGS = {
    "entities": [
        {
            "name": "MLX",
            "type": "software",
            "summary": "MLX is developed by Google for Google TPU clusters.",
            "sources": ["contradiction-source.md"],
        }
    ],
    "concepts": [],
    "contradictions": [
        {
            "page_slug": "entities/mlx",
            "existing_claim": EXISTING_CLAIM,
            "new_claim": NEW_CLAIM,
            "proposed_resolution": PROPOSED_RESOLUTION,
        }
    ],
}

FLAG_ARGS = {
    "page_slug": "entities/mlx",
    "existing_claim": EXISTING_CLAIM,
    "new_claim": NEW_CLAIM,
    "proposed_resolution": PROPOSED_RESOLUTION,
}


def _pre_resume_responses(raw_path: Path) -> list[AIMessage]:
    """The scripted pre-interrupt sequence (4 tool-call turns)."""
    return [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="read_source",
                    args={"source_path": str(raw_path / "contradiction-source.md")},
                    id="tc-1",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="submit_extraction", args=SUBMIT_ARGS, id="tc-2")
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
                ToolCall(name="flag_contradiction", args=FLAG_ARGS, id="tc-4")
            ],
        ),
    ]


def _run_to_interrupt(
    wiki_path: Path, raw_path: Path, post_resume: list[AIMessage]
) -> tuple[object, dict, dict]:
    """Build the ingest agent and run the scripted flow up to the interrupt."""
    init_shared_tools(str(wiki_path))
    model = ScriptedChatModel(
        responses=[*_pre_resume_responses(raw_path), *post_resume]
    )
    agent = build_agent(
        model=model,
        tools=INGEST_TOOLS,
        system_prompt=build_ingest_prompt("# Test schema"),
        middleware=_ingest_middleware(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Ingest raw/contradiction-source.md"}]},
        config=config,
    )
    return agent, config, result


def _assert_interrupt(result: dict) -> None:
    """The run stopped at the HITL interrupt, flag_contradiction requested."""
    assert "__interrupt__" in result
    pre_resume_names = [c["name"] for c in _all_tool_calls(result)]
    assert "flag_contradiction" in pre_resume_names
    assert "regenerate_index" not in pre_resume_names


def _assert_terminal_state(wiki_path: Path) -> None:
    """Post-resume invariants: log gained the ingest entry, index consistent."""
    log_text = (wiki_path / "log.md").read_text(encoding="utf-8")
    assert "ingest |" in log_text and "contradiction-source.md" in log_text
    index_text = (wiki_path / "index.md").read_text(encoding="utf-8")
    assert "mlx" in index_text.lower()


# --- resume variant: approve ----------------------------------------------------


def test_approve_applies_new_claim_and_keeps_page(eval_env) -> None:
    """Approve -> update_page applies the resolution; page kept, log + index updated."""
    wiki_path, raw_path = eval_env
    agent, config, result = _run_to_interrupt(
        wiki_path,
        raw_path,
        post_resume=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="update_page",
                        args={
                            "slug": "entities/mlx",
                            "content": (
                                "# MLX\n\n"
                                "MLX is a machine learning framework developed "
                                "by Google for Google TPU clusters.\n\n"
                                "## Related\n\n"
                                "- [[Apple Silicon]]"
                            ),
                        },
                        id="tc-5",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="regenerate_index", args={}, id="tc-6")
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="append_log",
                        args={"op": "ingest", "title": "contradiction-source.md"},
                        id="tc-7",
                    )
                ],
            ),
            AIMessage(content="Contradiction resolved: approved."),
        ],
    )
    _assert_interrupt(result)

    resumed = resume_auto(agent, config, state=result, choice="approve")
    assert resumed["messages"][-1].content == "Contradiction resolved: approved."

    # Page kept on disk and carrying the approved (new) claim.
    page = wiki_path / "entities" / "mlx.md"
    assert page.is_file()
    assert "Google" in page.read_text(encoding="utf-8")
    _assert_terminal_state(wiki_path)


# --- resume variant: reject -----------------------------------------------------


def test_reject_leaves_page_unchanged(eval_env) -> None:
    """Reject -> no update_page; page bytes identical, run still logs + regenerates."""
    wiki_path, raw_path = eval_env
    page = wiki_path / "entities" / "mlx.md"
    before = page.read_text(encoding="utf-8")

    agent, config, result = _run_to_interrupt(
        wiki_path,
        raw_path,
        post_resume=[
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
                        args={"op": "ingest", "title": "contradiction-source.md"},
                        id="tc-6",
                    )
                ],
            ),
            AIMessage(content="Contradiction rejected; page unchanged."),
        ],
    )
    _assert_interrupt(result)

    resumed = resume_auto(
        agent,
        config,
        state=result,
        choice="reject",
        feedback="Keep the existing claim.",
    )
    assert resumed["messages"][-1].content == "Contradiction rejected; page unchanged."

    # The page was NOT touched: byte-identical, old claim still present.
    after = page.read_text(encoding="utf-8")
    assert after == before
    assert "Google" not in after
    _assert_terminal_state(wiki_path)


# --- resume variant: edit -------------------------------------------------------


def test_edit_re_executes_flag_with_edited_resolution(eval_env) -> None:
    """Edit -> flag_contradiction re-executes with the human's resolution (visible
    in the ToolMessage) and the page adopts the merged wording."""
    wiki_path, raw_path = eval_env
    new_resolution = (
        "Merge both claims: Apple developed MLX for Apple Silicon; "
        "the Google/TPU claim is unsupported."
    )
    merged_wording = "Apple developed MLX for Apple Silicon"

    agent, config, result = _run_to_interrupt(
        wiki_path,
        raw_path,
        post_resume=[
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="update_page",
                        args={
                            "slug": "entities/mlx",
                            "content": (
                                "# MLX\n\n"
                                f"{merged_wording}.\n\n"
                                "## Related\n\n"
                                "- [[Apple Silicon]]"
                            ),
                        },
                        id="tc-5",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="regenerate_index", args={}, id="tc-6")
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="append_log",
                        args={"op": "ingest", "title": "contradiction-source.md"},
                        id="tc-7",
                    )
                ],
            ),
            AIMessage(content="Contradiction resolved: edited."),
        ],
    )
    _assert_interrupt(result)

    # Headless edit resume. ``state=result`` gives ``resume_auto`` the
    # interrupt-bearing invoke RESULT (langchain's HumanInTheLoopMiddleware
    # keeps ``__interrupt__``/``action_requests`` there, NOT in
    # ``agent.get_state(config).values``), so the CLI edit shape
    # (cli.py ~L119-129) reaches the middleware genuinely. Never ``input()``.
    resumed = resume_auto(
        agent,
        config,
        state=result,
        choice="edit",
        index=0,
        new_resolution=new_resolution,
    )
    assert resumed["messages"][-1].content == "Contradiction resolved: edited."

    # The edited flag_contradiction re-executed with the edited args — the
    # re-executed ToolMessage renders the new resolution into the flag string.
    edited_tool_messages = [
        m
        for m in resumed["messages"]
        if isinstance(m, ToolMessage) and new_resolution in m.content
    ]
    assert edited_tool_messages, (
        "edited flag_contradiction ToolMessage with the new resolution not "
        "found in the resumed run"
    )

    # Page carries the merged resolution.
    page_text = (wiki_path / "entities" / "mlx.md").read_text(encoding="utf-8")
    assert merged_wording in page_text
    _assert_terminal_state(wiki_path)
