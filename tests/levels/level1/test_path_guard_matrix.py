"""Level 1 — path-guard matrix (0 LLM, deterministic).

Exhaustively pins the write-path protection of
``agentic_rag.middleware.guardrails._path_guard_error`` — a pure function the
production ``PathGuardMiddleware`` runs on every tool call:

1. EVERY write tool (all 7: create_page, update_page, delete_wiki_page,
   write_lint_report, add_frontmatter, fix_link, append_related_section)
   rejects every path-shaped attack value with an ``ERROR:``-prefixed string
   (never None, never a raised exception).
2. ``read_source`` — a READ tool whose ``source_path`` legitimately points
   into raw/ — is ALLOWED (returns None).
3. Valid in-wiki slugs pass the guard on write tools (returns None).
4. The guard scans all four arg keys (slug/path/file_path/source_path).
5. End-to-end proof: a real ``build_agent`` run with a scripted model calling
   a write tool with a bad slug short-circuits with the ``ERROR:`` string as
   the tool RESULT and the run completes without raising.

Observed-behavior pin (documented gap, NOT to be "fixed" here — src/ is a
hard no-touch gate): the bare value ``./raw`` (no trailing slash) does NOT
trip the guard — the middleware matches ``"raw/" in val`` which misses it,
and ``./raw`` neither starts with ``"/"`` nor contains ``".."``. The matrix
therefore uses ``./raw/`` and ``./raw/sample.md`` to represent the ``./raw``
row, and the gap itself is asserted as observed behavior.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolCall

from agentic_rag.agents.factory import build_agent
from agentic_rag.middleware.guardrails import _path_guard_error
from agentic_rag.tools.ingest_tools import create_page
from agentic_rag.tools.shared import init_shared_tools
from tests.fixtures.fake_llm import ScriptedChatModel

WRITE_TOOLS = [
    "create_page",
    "update_page",
    "delete_wiki_page",
    "write_lint_report",
    "add_frontmatter",
    "fix_link",
    "append_related_section",
]

# (value, label) — every value must trip the guard on every write tool.
# `./raw` is represented by `./raw/` and `./raw/sample.md` (see module docstring).
REJECTED_VALUES = [
    ("raw/foo.md", "raw/-prefixed path"),
    ("/abs/path", "absolute path"),
    ("a/../b", "directory traversal"),
    ("./raw/", "./raw with trailing slash"),
    ("./raw/sample.md", "./raw/ with filename"),
]

# Keys the guard scans on write tools (guard is schema-agnostic: it does not
# care which key the tool actually declares).
SCANNED_ARG_KEYS = ["slug", "path", "file_path", "source_path"]

VALID_IN_WIKI_SLUG = "entities/mlx"


def _request(tool_name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(tool_call={"name": tool_name, "args": args})


class TestRejectionMatrix:
    @pytest.mark.parametrize("tool_name", WRITE_TOOLS)
    @pytest.mark.parametrize(
        "value,label",
        REJECTED_VALUES,
        ids=[f"{v[1]}" for v in REJECTED_VALUES],
    )
    def test_every_write_tool_rejects_every_attack_value(self, tool_name, value, label):
        """All 7 write tools x 5 attack values -> ERROR string, never None,
        never raised."""
        result = _path_guard_error(_request(tool_name, {"slug": value}))
        assert result is not None, f"{tool_name}(slug={value!r}) was NOT blocked"
        assert isinstance(result, str)
        assert result.startswith("ERROR:"), f"{tool_name}({value!r}) -> {result!r}"

    def test_bare_dot_raw_does_not_trip_guard_observed_gap(self):
        """Pinned observed behavior: bare ``./raw`` slips through (middleware
        matches ``"raw/"``, absolute prefix, and ``..`` — none hit)."""
        result = _path_guard_error(
            _request("create_page", {"slug": "./raw"})
        )
        assert result is None

    def test_bare_raw_slash_variants_do_trip(self):
        """The trailing-slash variants of the gap DO trip (via ``raw/``)."""
        for value in ("./raw/", "./raw/sample.md"):
            result = _path_guard_error(_request("create_page", {"slug": value}))
            assert result is not None and result.startswith("ERROR:")

    def test_all_scanned_arg_keys_are_guarded(self):
        """The guard scans slug/path/file_path/source_path, not just slug."""
        for key in SCANNED_ARG_KEYS:
            result = _path_guard_error(_request("create_page", {key: "raw/foo.md"}))
            assert result is not None and result.startswith("ERROR:"), (
                f"key {key!r} was not guarded"
            )

    def test_write_source_path_guard(self):
        """Write tools carrying a raw/ source_path are blocked (even when the
        slug itself is clean)."""
        result = _path_guard_error(
            _request(
                "create_page",
                {"slug": "entities/ok", "source_path": "raw/cv.pdf"},
            )
        )
        assert result is not None and result.startswith("ERROR:")


class TestAllowedCalls:
    def test_read_source_allows_raw_paths(self):
        """read_source is a READ tool — raw/ source_paths are legitimate."""
        for value in ("raw/sample.md", "./raw/sample.md"):
            result = _path_guard_error(
                _request("read_source", {"source_path": value})
            )
            assert result is None, f"read_source({value!r}) was blocked: {result!r}"

    @pytest.mark.parametrize("tool_name", WRITE_TOOLS)
    def test_valid_in_wiki_slug_allowed_on_write_tools(self, tool_name):
        """In-wiki slugs pass the guard on every write tool."""
        result = _path_guard_error(
            _request(tool_name, {"slug": VALID_IN_WIKI_SLUG})
        )
        assert result is None, f"{tool_name} blocked {VALID_IN_WIKI_SLUG!r}: {result!r}"


class TestEndToEndShortCircuit:
    def test_error_string_short_circuits_real_agent_run(self, eval_wiki):
        """A scripted agent calling create_page with a raw/ slug gets the
        ERROR string as the tool RESULT; the run completes without raising."""
        init_shared_tools(str(eval_wiki))
        model = ScriptedChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="create_page",
                            args={"slug": "raw/evil.md"},
                            id="tc-1",
                        )
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
        agent = build_agent(
            model=model,
            tools=[create_page],
            system_prompt="You are a test ingest agent.",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "create raw/evil.md"}]},
            config={"configurable": {"thread_id": str(uuid4())}},
        )

        # The middleware short-circuits the tool BEFORE its handler runs, so
        # langchain injects the ERROR string as a plain result message (not a
        # ToolMessage — the tool never executed). The contract we pin: the run
        # completes without raising and the ERROR-prefixed string is the tool's
        # result content.
        error_results = [
            m
            for m in result["messages"]
            if isinstance(m.content, str) and m.content.startswith("ERROR:")
        ]
        assert len(error_results) == 1
        assert "raw/evil.md" in error_results[0].content
        # The blocked page must NOT exist on disk (the handler never ran).
        assert not (eval_wiki / "raw" / "evil.md").exists()
