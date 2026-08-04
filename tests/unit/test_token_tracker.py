"""Tests for token usage and cost tracking."""

import logging

import pytest

from agentic_rag.token_tracker import MODEL_PRICING, TokenTracker, TokenUsage


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        """Test default values are zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.input_cost == 0.0
        assert usage.output_cost == 0.0
        assert usage.total_cost == 0.0
        assert usage.duration_seconds == 0.0

    def test_custom_values(self):
        """Test custom values are set correctly."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
            duration_seconds=1.5,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.input_cost == 0.001
        assert usage.output_cost == 0.002
        assert usage.total_cost == 0.003
        assert usage.duration_seconds == 1.5


class TestTokenTracker:
    """Tests for TokenTracker class."""

    def test_initialization(self):
        """Test tracker initializes correctly."""
        tracker = TokenTracker("gpt-4.1-mini")
        assert tracker.model_name == "gpt-4.1-mini"
        assert tracker.pricing == MODEL_PRICING["gpt-4.1-mini"]
        assert tracker.call_count == 0

    def test_initialization_unknown_model(self):
        """Test tracker with unknown model defaults to zero pricing."""
        tracker = TokenTracker("unknown-model")
        assert tracker.pricing == {"input": 0.0, "output": 0.0}

    def test_record_call(self):
        """Test recording a single call."""
        tracker = TokenTracker("gpt-4.1-mini")
        usage = tracker.record_call(input_tokens=1000, output_tokens=500, duration=1.0)

        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_tokens == 1500
        assert usage.duration_seconds == 1.0
        assert usage.total_cost > 0
        assert tracker.call_count == 1

    def test_record_call_cost_calculation(self):
        """Test cost calculation is correct."""
        tracker = TokenTracker("gpt-4.1-mini")
        # gpt-4.1-mini: input=$0.40/1M, output=$1.60/1M
        usage = tracker.record_call(input_tokens=1_000_000, output_tokens=1_000_000, duration=1.0)

        assert usage.input_cost == pytest.approx(0.40)
        assert usage.output_cost == pytest.approx(1.60)
        assert usage.total_cost == pytest.approx(2.00)

    def test_cumulative_tracking(self):
        """Test cumulative tracking across multiple calls."""
        tracker = TokenTracker("gpt-4.1-mini")

        tracker.record_call(input_tokens=100, output_tokens=50, duration=0.5)
        tracker.record_call(input_tokens=200, output_tokens=100, duration=0.7)

        summary = tracker.get_summary()
        assert summary.input_tokens == 300
        assert summary.output_tokens == 150
        assert summary.total_tokens == 450
        assert summary.duration_seconds == pytest.approx(1.2)
        assert tracker.call_count == 2

    def test_get_summary(self):
        """Test get_summary returns cumulative usage."""
        tracker = TokenTracker("gpt-4.1-mini")
        tracker.record_call(input_tokens=100, output_tokens=50, duration=0.5)

        summary = tracker.get_summary()
        assert isinstance(summary, TokenUsage)
        assert summary.input_tokens == 100

    def test_log_summary(self, caplog):
        """Test log_summary outputs to logger."""
        tracker = TokenTracker("gpt-4.1-mini")
        tracker.record_call(input_tokens=1000, output_tokens=500, duration=1.0)

        with caplog.at_level(logging.INFO, logger="agentic_rag.tokens"):
            tracker.log_summary()

        assert "SESSION SUMMARY" in caplog.text
        assert "1 LLM calls" in caplog.text
        assert "tokens:" in caplog.text

    def test_record_call_logs_info(self, caplog):
        """Test record_call logs token usage at INFO level."""
        tracker = TokenTracker("gpt-4.1-mini")

        with caplog.at_level(logging.INFO, logger="agentic_rag.tokens"):
            tracker.record_call(input_tokens=100, output_tokens=50, duration=0.5)

        assert "TOKENS:" in caplog.text
        assert "in=100" in caplog.text
        assert "out=50" in caplog.text

    def test_record_call_logs_cumulative_at_debug(self, caplog):
        """Test record_call logs cumulative at DEBUG level."""
        tracker = TokenTracker("gpt-4.1-mini")

        with caplog.at_level(logging.DEBUG, logger="agentic_rag.tokens"):
            tracker.record_call(input_tokens=100, output_tokens=50, duration=0.5)

        assert "CUMULATIVE:" in caplog.text


class TestModelPricing:
    """Tests for MODEL_PRICING configuration."""

    def test_known_models_have_pricing(self):
        """Test known models have pricing defined."""
        known_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "glm-5.2", "deepseek-v4-pro", "deepseek-v4-flash"]
        for model in known_models:
            assert model in MODEL_PRICING
            assert "input" in MODEL_PRICING[model]
            assert "output" in MODEL_PRICING[model]
            assert MODEL_PRICING[model]["input"] >= 0
            assert MODEL_PRICING[model]["output"] >= 0

    def test_pricing_values_are_positive(self):
        """Test all pricing values are non-negative."""
        for model, pricing in MODEL_PRICING.items():
            assert pricing["input"] >= 0, f"{model} input price is negative"
            assert pricing["output"] >= 0, f"{model} output price is negative"
