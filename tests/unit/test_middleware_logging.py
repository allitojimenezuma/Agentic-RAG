"""Tests for the audit-logging middleware and per-agent token capture."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agentic_rag.middleware.logging import (
    audit_logging_middleware,
    make_token_capture,
)
from agentic_rag.token_tracker import TokenTracker


class TestAuditLoggingMiddleware:
    """Tests for audit_logging_middleware."""

    def test_middleware_is_registered(self):
        """The shared audit middleware exposes both sync and async hooks."""
        assert audit_logging_middleware is not None
        assert hasattr(audit_logging_middleware, "wrap_tool_call")
        assert hasattr(audit_logging_middleware, "awrap_tool_call")


class TestTokenCapture:
    """Tests for make_token_capture — per-agent binding of LLM usage."""

    def _state_with_usage(self, usage: dict) -> dict:
        return {"messages": [AIMessage(content="answer", response_metadata={"token_usage": usage})]}

    def test_captures_token_usage_from_state(self):
        tracker = TokenTracker("gpt-4.1-mini")
        middleware = make_token_capture(tracker)

        middleware.after_model(
            self._state_with_usage({"prompt_tokens": 100, "completion_tokens": 50}),
            runtime=None,
        )

        assert tracker.call_count == 1
        summary = tracker.get_summary()
        assert summary.input_tokens == 100
        assert summary.output_tokens == 50

    def test_reads_usage_under_alternate_key(self):
        tracker = TokenTracker("gpt-4.1-mini")
        middleware = make_token_capture(tracker)

        middleware.after_model(
            {"messages": [AIMessage(content="answer", response_metadata={"usage": {"prompt_tokens": 7, "completion_tokens": 3}})]},
            runtime=None,
        )

        assert tracker.call_count == 1

    def test_noop_when_no_usage_present(self):
        tracker = TokenTracker("gpt-4.1-mini")
        middleware = make_token_capture(tracker)

        middleware.after_model({"messages": [AIMessage(content="x")]}, runtime=None)

        assert tracker.call_count == 0

    def test_noop_without_messages(self):
        tracker = TokenTracker("gpt-4.1-mini")
        middleware = make_token_capture(tracker)

        middleware.after_model({"messages": []}, runtime=None)

        assert tracker.call_count == 0

    def test_each_agent_gets_its_own_tracker(self):
        """Two middleware instances never write to each other's tracker."""
        tracker_a = TokenTracker("gpt-4.1-mini")
        tracker_b = TokenTracker("gpt-4.1-mini")
        middleware_a = make_token_capture(tracker_a)
        middleware_b = make_token_capture(tracker_b)

        middleware_a.after_model(
            self._state_with_usage({"prompt_tokens": 1, "completion_tokens": 1}),
            runtime=None,
        )

        assert tracker_a.call_count == 1
        assert tracker_b.call_count == 0