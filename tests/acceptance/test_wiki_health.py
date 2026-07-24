"""Acceptance test: run lint agent over real wiki/."""

import os
import pytest
from agentic_rag.config import Settings
from agentic_rag.agents.lint import build_lint_agent


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY")
def test_lint_real_wiki():
    """Run lint agent over real wiki/ — must complete without exceptions."""
    settings = Settings()
    agent = build_lint_agent(settings)
    config = {
        "configurable": {"thread_id": "acceptance-lint"},
        "recursion_limit": settings.recursion_limit,
    }

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Run a full wiki health check."}]},
        config=config,
    )

    # Should complete without exceptions
    assert result["messages"][-1].content
    assert len(result["messages"]) > 1
