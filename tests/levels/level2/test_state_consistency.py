"""Level 2 — post-ingest state consistency (0 LLM, fully headless).

Proves that the deterministic ingest flow leaves the wiki in a fully
consistent state on a tmp copy of the eval corpus (``eval_env``):

(a) the ingest writes a page (direct engine calls and, separately, one
    scripted agent run end-to-end);
(b) ``regenerate_index`` is idempotent — two consecutive runs produce an
    IDENTICAL ``index.md`` — and the created page's slug is present in the
    index (asserted through the deterministic ``read_index`` engine);
(c) ``log.md`` gained a new entry (parsed via ``tail_log``);
(d) ``health_check`` reports zero issues on the copy — no orphans introduced,
    index complete;
(e) the written page's raw content starts with ``---`` and ``load_wiki``
    parses it with valid frontmatter fields.

Assertions go through the deterministic engine (``load_wiki``, ``health_check``,
``regenerate_index``, ``read_index``, ``tail_log``) — never ad-hoc file parsing.
The created page is linked FROM an existing corpus page so it is not an
orphan; the whole wiki must stay zero-issue after the ingest.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_ingest_prompt
from agentic_rag.io.index_manager import read_index
from agentic_rag.io.log_manager import tail_log
from agentic_rag.io.wiki_io import read_page_with_frontmatter
from agentic_rag.lint.health import health_check
from agentic_rag.tools.ingest_grounding import submit_extraction
from agentic_rag.tools.ingest_tools import (
    append_log,
    create_page,
    read_source,
    update_page,
)
from agentic_rag.tools.nav import regenerate_index
from agentic_rag.tools.shared import init_shared_tools
from agentic_rag.wiki.dedupe_index import regenerate_index as regenerate_index_engine
from agentic_rag.wiki.match import match_page_tool
from agentic_rag.wiki.model import load_wiki
from tests.fixtures.fake_llm import ScriptedChatModel

NEWCORP_SLUG = "entities/newcorp"
NEWCORP_CONTENT = (
    "# Newcorp\n\n"
    "Newcorp is a fictional organization introduced by the sample source to "
    "exercise the deterministic ingest pipeline. It provides cloud "
    "infrastructure services and sponsors open-source machine learning "
    "research, with offices in Málaga and Madrid. The organization collaborates "
    "with local universities on distributed training systems and contributes "
    "reference implementations to the MLX ecosystem.\n\n"
    "## Related\n\n"
    "- [[MLX]]"
)


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


def _run(agent, user_content: str, config: dict) -> dict:
    """Invoke a built agent with a plain user message."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]}, config=config
    )


def _link_newcorp(wiki_path: Path) -> None:
    """Link the existing entities/mlx page to the new page (no orphans)."""
    _fm, body = read_page_with_frontmatter(wiki_path, "entities/mlx")
    update_page.invoke(
        {"slug": "entities/mlx", "content": body.rstrip() + "\n\n- [[Newcorp]]\n"}
    )


def _assert_index_state(wiki_path: Path) -> None:
    """Index is idempotent, complete, and carries the new page's slug."""
    regenerate_index_engine(wiki_path)
    first = (wiki_path / "index.md").read_text(encoding="utf-8")
    regenerate_index_engine(wiki_path)
    second = (wiki_path / "index.md").read_text(encoding="utf-8")
    # (b) idempotent: two regenerations produce identical content.
    assert first == second
    # The created page is present (display name in text, slug via the engine).
    assert "[[Newcorp]]" in second
    index = read_index(wiki_path)
    entry_slugs = {e.slug for entries in index.categories.values() for e in entries}
    assert "newcorp" in entry_slugs


def _assert_page_state(wiki_path: Path) -> None:
    """(e) raw page starts with --- and load_wiki parses valid frontmatter."""
    raw = (wiki_path / NEWCORP_SLUG).with_suffix(".md").read_text(encoding="utf-8")
    assert raw.startswith("---")
    page = load_wiki(wiki_path).by_slug[NEWCORP_SLUG]
    assert page.fm.type == "entity"
    assert page.fm.title == "Newcorp"
    assert page.fm.sources == ["sample.md"]
    assert page.fm.updated is not None


class TestStateConsistency:
    """Ingest writes leave a zero-issue, idempotent, fully indexed wiki."""

    def test_direct_engine_ingest_leaves_consistent_state(self, eval_env):
        """Direct engine calls: create_page -> regenerate_index -> append_log
        (+ link step) yield an idempotent index, a log entry, zero health
        issues and a valid parsed page."""
        wiki_path, raw_path = eval_env
        init_shared_tools(str(wiki_path))

        # (a) direct engine writes.
        create_page.invoke(
            {
                "slug": NEWCORP_SLUG,
                "page_type": "entity",
                "title": "Newcorp",
                "content": NEWCORP_CONTENT,
                "sources": ["sample.md"],
            }
        )
        regenerate_index_engine(wiki_path)
        append_log.invoke(
            {
                "op": "ingest",
                "title": "sample.md",
                "details": "Created entities/newcorp from sample.md",
            }
        )
        # Link the new page from an existing one so it is not an orphan.
        _link_newcorp(wiki_path)

        # (b) idempotent index + new slug present.
        _assert_index_state(wiki_path)

        # (c) log.md gained a new entry (parsed via tail_log).
        entries = tail_log(wiki_path, n=1)
        assert entries
        assert entries[-1].op == "ingest"
        assert entries[-1].title == "sample.md"

        # (d) zero issues on the copy — no orphans, index complete.
        report = health_check(wiki_path)
        assert report.issues == []

        # (e) page shape + parseable frontmatter.
        _assert_page_state(wiki_path)

    def test_scripted_ingest_run_leaves_consistent_state(self, eval_env):
        """One scripted agent run (read_source -> submit_extraction ->
        match_page_tool -> create_page -> regenerate_index -> append_log)
        proves the end-to-end post-ingest state on disk."""
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
                                        "name": "Newcorp",
                                        "type": "organization",
                                        "summary": (
                                            "A fictional organization from "
                                            "sample.md."
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
                            args={"name": "Newcorp", "page_type": "entity"},
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
                                "slug": NEWCORP_SLUG,
                                "page_type": "entity",
                                "title": "Newcorp",
                                "content": NEWCORP_CONTENT,
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
            tools=[
                read_source,
                submit_extraction,
                match_page_tool,
                create_page,
                update_page,
                regenerate_index,
                append_log,
            ],
            system_prompt=build_ingest_prompt("# Test schema"),
            middleware=_ingest_middleware(),
        )

        config = {"configurable": {"thread_id": str(uuid4())}}
        result = _run(agent, "Ingest raw/sample.md", config)

        # The scripted happy path ran to completion (6 tool calls + final answer).
        names = [c["name"] for c in _all_tool_calls(result)]
        assert names[-2:] == ["regenerate_index", "append_log"]
        assert result["messages"][-1].content == "Ingested sample.md."

        # (a) the run wrote the page.
        page_path = wiki_path / NEWCORP_SLUG
        assert page_path.with_suffix(".md").is_file()

        # Finalize: link the page and regenerate the index (still zero issues).
        _link_newcorp(wiki_path)
        _assert_index_state(wiki_path)

        # (c) log.md gained the ingest entry.
        entries = tail_log(wiki_path, n=1)
        assert entries
        assert entries[-1].op == "ingest"
        assert entries[-1].title == "sample.md"

        # (d) zero issues on the copy after the end-to-end run.
        report = health_check(wiki_path)
        assert report.issues == []

        # (e) page shape + parseable frontmatter.
        _assert_page_state(wiki_path)
