"""Level-scoped fixtures for the tests/levels/ suite (T1).

Two jobs:

1. Load the repo ``.env`` into ``os.environ`` at MODULE import time, so the
   ``requires_llm`` skip-if-no-key check sees the user's key. ``override=False``
   keeps any already-exported env vars; a missing ``.env`` is a no-op, so the
   suite stays headless-safe.
2. Re-export the corpus/HITL helpers from ``tests/fixtures`` so level tests
   import them from one place: ``eval_wiki`` / ``eval_env`` fixtures,
   ``copy_eval_wiki`` / ``copy_broken_wiki``, and the ``requires_llm``
   decorator used by every real-LLM tier.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root: tests/levels/<this file> -> parents[2].
_ENV_PATH: Path = Path(__file__).parents[2] / ".env"
if _ENV_PATH.is_file():
    load_dotenv(_ENV_PATH, override=False)

import pytest  # noqa: E402  (import after env bootstrap is deliberate)

from tests.fixtures.eval_corpus import (  # noqa: E402
    copy_broken_wiki,
    copy_eval_wiki,
    eval_env,
    eval_wiki,
)

__all__ = [
    "copy_broken_wiki",
    "copy_eval_wiki",
    "eval_env",
    "eval_wiki",
    "requires_llm",
]

def requires_llm(func):
    """Mark a test as a real-LLM tier test.

    Adds two markers:

    - ``requires_llm`` — selectable via ``pytest -m requires_llm``; the
      suite's default ``addopts`` (``-m 'not requires_llm'``) deselects it, so
      a plain ``pytest`` stays offline and finishes in seconds.
    - ``skipif`` — without ``OPENAI_API_KEY`` the test is skipped even when
      explicitly selected, so headless/CI runs never fail on a missing key.

    Decorate only plain test functions (no ``@pytest.mark.parametrize`` above
    it), exactly as the five real-LLM tests do today.
    """
    func = pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="No OPENAI_API_KEY configured — real-LLM tier skipped",
    )(func)
    return pytest.mark.requires_llm(func)
