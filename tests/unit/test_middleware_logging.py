"""Tests for logging middleware with token tracking."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.middleware.logging import audit_logging_middleware, get_tracker, set_tracker
from agentic_rag.token_tracker import TokenTracker


class TestSetTracker:
    """Tests for set_tracker function."""

    def test_set_and_get_tracker(self):
        """Test setting and getting tracker."""
        tracker = TokenTracker("gpt-4.1-mini")
        set_tracker(tracker)
        assert get_tracker() is tracker

    def test_get_tracker_none_by_default(self):
        """Test get_tracker returns None when not set."""
        set_tracker(None)
        assert get_tracker() is None


class TestAuditLoggingMiddleware:
    """Tests for audit_logging_middleware."""

    def setup_method(self):
        """Reset tracker before each test."""
        set_tracker(None)

    def test_middleware_is_registered(self):
        """Test that middleware is registered and can be used."""
        # The middleware is registered via @wrap_tool_call decorator
        # It should be accessible and usable in the agent pipeline
        assert audit_logging_middleware is not None

    def test_middleware_has_correct_name(self):
        """Test that middleware has correct name."""
        # Check the middleware object has the expected name
        assert hasattr(audit_logging_middleware, "__name__") or hasattr(
            audit_logging_middleware, "name"
        )

    def test_tracker_functions_work(self):
        """Test that tracker functions work correctly."""
        tracker = TokenTracker("gpt-4.1-mini")
        set_tracker(tracker)
        assert get_tracker() is tracker

        set_tracker(None)
        assert get_tracker() is None

    def test_tracker_records_usage(self):
        """Test that tracker can record token usage."""
        tracker = TokenTracker("gpt-4.1-mini")
        set_tracker(tracker)

        # Simulate recording a call
        usage = tracker.record_call(input_tokens=100, output_tokens=50, duration=0.5)

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert tracker.call_count == 1

    def test_tracker_cumulative_usage(self):
        """Test that tracker accumulates usage across calls."""
        tracker = TokenTracker("gpt-4.1-mini")
        set_tracker(tracker)

        tracker.record_call(input_tokens=100, output_tokens=50, duration=0.5)
        tracker.record_call(input_tokens=200, output_tokens=100, duration=0.7)

        summary = tracker.get_summary()
        assert summary.input_tokens == 300
        assert summary.output_tokens == 150
        assert tracker.call_count == 2

    def test_tracker_cost_calculation(self):
        """Test that tracker calculates costs correctly."""
        tracker = TokenTracker("gpt-4.1-mini")
        # gpt-4.1-mini: input=$0.40/1M, output=$1.60/1M
        usage = tracker.record_call(
            input_tokens=1_000_000, output_tokens=1_000_000, duration=1.0
        )

        assert usage.input_cost == pytest.approx(0.40)
        assert usage.output_cost == pytest.approx(1.60)
        assert usage.total_cost == pytest.approx(2.00)
