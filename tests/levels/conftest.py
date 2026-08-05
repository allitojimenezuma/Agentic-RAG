"""Level-scoped fixtures for the tests/levels/ suite (T1).

Two jobs:

1. Load the repo ``.env`` into ``os.environ`` at MODULE import time. The real
   LLM tiers gate on ``os.getenv("OPENAI_API_KEY")`` in ``@pytest.mark.skipif``
   markers, but ``pydantic-settings`` loads ``.env`` into ``Settings``, NOT
   into ``os.environ`` — so without this hook the skip markers would never see
   the user's key. ``override=False`` keeps any already-exported env vars; a
   missing ``.env`` is a no-op, so the suite stays headless-safe.
2. Re-export the corpus/HITL helpers from ``tests/fixtures`` so level tests
   import them from one place: ``eval_wiki`` / ``eval_env`` fixtures,
   ``copy_eval_wiki`` / ``copy_broken_wiki``, and the ``requires_llm`` marker
   used by every real-LLM tier.
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

# Judge/trajectory tiers: skip without a key. os.environ now includes .env
# (loaded above), so with the user's .env present these tiers actually run.
requires_llm = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="No OPENAI_API_KEY configured — real-LLM tier skipped",
)
