"""Token usage and cost tracking for LLM calls."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("agentic_rag.tokens")

# Pricing per model (USD per 1M tokens) — update as needed
MODEL_PRICING = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "glm-5.2": {"input": 1.40, "output": 4.40},
    "deepseek-v4-pro": {"input": 0.44, "output": 0.87},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    # Add more models as needed
}


@dataclass
class TokenUsage:
    """Token usage statistics for a single call or cumulative session."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    duration_seconds: float = 0.0


class TokenTracker:
    """Tracks token usage and cost across LLM calls."""

    def __init__(self, model_name: str):
        """Initialize token tracker for a specific model.

        Args:
            model_name: Name of the model to look up pricing for.
        """
        self.model_name = model_name
        self.pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        self._cumulative = TokenUsage()
        self._call_count = 0

    def record_call(
        self, input_tokens: int, output_tokens: int, duration: float
    ) -> TokenUsage:
        """Record a single LLM call and return its usage.

        Args:
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
            duration: Duration of the call in seconds.

        Returns:
            TokenUsage for this specific call.
        """
        input_cost = (input_tokens / 1_000_000) * self.pricing["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing["output"]
        total_cost = input_cost + output_cost

        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            duration_seconds=duration,
        )

        # Update cumulative
        self._cumulative.input_tokens += input_tokens
        self._cumulative.output_tokens += output_tokens
        self._cumulative.total_tokens += input_tokens + output_tokens
        self._cumulative.input_cost += input_cost
        self._cumulative.output_cost += output_cost
        self._cumulative.total_cost += total_cost
        self._cumulative.duration_seconds += duration
        self._call_count += 1

        logger.info(
            f"TOKENS: in={input_tokens} out={output_tokens} "
            f"cost=${total_cost:.6f} (${input_cost:.6f}+${output_cost:.6f}) "
            f"duration={duration:.3f}s"
        )
        logger.debug(
            f"CUMULATIVE: {self._call_count} calls, "
            f"in={self._cumulative.input_tokens} out={self._cumulative.output_tokens}, "
            f"total=${self._cumulative.total_cost:.6f}"
        )

        return usage

    def get_summary(self) -> TokenUsage:
        """Get cumulative usage summary.

        Returns:
            TokenUsage with cumulative statistics.
        """
        return self._cumulative

    def log_summary(self) -> None:
        """Log the final session summary."""
        s = self._cumulative
        logger.info(
            f"SESSION SUMMARY: {self._call_count} LLM calls, "
            f"tokens: in={s.input_tokens} out={s.output_tokens} total={s.total_tokens}, "
            f"cost: ${s.total_cost:.6f} (${s.input_cost:.6f}+${s.output_cost:.6f}), "
            f"time: {s.duration_seconds:.3f}s"
        )

    @property
    def call_count(self) -> int:
        """Number of calls recorded."""
        return self._call_count
