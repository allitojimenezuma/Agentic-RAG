"""Unit tests for path_guard_middleware (registered in build_agent)."""

from __future__ import annotations

from types import SimpleNamespace

from agentic_rag.agents import factory
from agentic_rag.middleware.guardrails import path_guard_middleware


class TestPathGuardMiddleware:
    def test_blocks_write_outside_wiki(self):
        """A write-tool arg containing raw/ is rejected; handler NOT called."""
        calls: list = []

        def handler(request):
            calls.append(request)
            return "handled"

        request = SimpleNamespace(
            tool_call={"name": "write_lint_report", "args": {"slug": "raw/foo"}}
        )
        result = path_guard_middleware.wrap_tool_call(request, handler)
        assert "ERROR" in result
        assert calls == []

    def test_blocks_absolute_path(self):
        calls: list = []

        def handler(request):
            calls.append(request)
            return "handled"

        request = SimpleNamespace(
            tool_call={"name": "create_page", "args": {"slug": "/etc/passwd"}}
        )
        result = path_guard_middleware.wrap_tool_call(request, handler)
        assert "ERROR" in result
        assert calls == []

    def test_allows_read_tool(self):
        """Read tools pass through — handler IS called."""
        calls: list = []

        def handler(request):
            calls.append(request)
            return "handled"

        request = SimpleNamespace(
            tool_call={"name": "wiki_read_page", "args": {"slug": "entities/python"}}
        )
        result = path_guard_middleware.wrap_tool_call(request, handler)
        assert result == "handled"
        assert len(calls) == 1

    def test_read_source_allows_raw_paths(self):
        """read_source (ingest's primary READ tool) must accept raw/ and absolute paths."""
        for p in ("raw/cv.pdf", "./raw/cv.pdf", "/Users/x/Proyectos/LangChain-RAG/raw/cv.pdf"):
            calls: list = []

            def handler(request):
                calls.append(request)
                return "handled"

            request = SimpleNamespace(
                tool_call={"name": "read_source", "args": {"source_path": p}}
            )
            result = path_guard_middleware.wrap_tool_call(request, handler)
            assert result == "handled", f"read_source({p!r}) was blocked: {result}"
            assert len(calls) == 1

    def test_blocks_new_fix_write_tools_with_bad_slug(self):
        """The new fix write-tools reject raw/, absolute, and '..' slugs."""
        for tool_name in ("add_frontmatter", "fix_link", "append_related_section"):
            for arg in ("raw/foo", "/etc/passwd", "entities/../secret"):
                calls: list = []

                def handler(request):
                    calls.append(request)
                    return "handled"

                request = SimpleNamespace(
                    tool_call={"name": tool_name, "args": {"slug": arg}}
                )
                result = path_guard_middleware.wrap_tool_call(request, handler)
                assert "ERROR" in result, f"{tool_name}({arg!r}) was not blocked: {result}"
                assert calls == [], f"handler called for blocked {tool_name}({arg!r})"

    def test_allows_new_fix_write_tools_with_in_wiki_slug(self):
        """In-wiki slugs pass the guardrail for the new fix write-tools."""
        for tool_name in ("add_frontmatter", "fix_link", "append_related_section"):
            calls: list = []

            def handler(request):
                calls.append(request)
                return "handled"

            request = SimpleNamespace(
                tool_call={"name": tool_name, "args": {"slug": "entities/python"}}
            )
            result = path_guard_middleware.wrap_tool_call(request, handler)
            assert result == "handled", f"{tool_name} was blocked: {result}"
            assert len(calls) == 1

    def test_registered_in_build_agent(self, monkeypatch):
        """The middleware is wired into build_agent's middleware chain."""
        captured: dict = {}

        def fake_create_agent(**kwargs):
            captured["middleware"] = kwargs.get("middleware", [])
            return SimpleNamespace()

        monkeypatch.setattr(factory, "create_agent", fake_create_agent)
        factory.build_agent(
            model=object(), tools=[], system_prompt="p", model_name="m"
        )
        assert path_guard_middleware in captured["middleware"]
