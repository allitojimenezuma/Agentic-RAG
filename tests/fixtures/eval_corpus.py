"""Committed eval corpus for the levels suite (T1).

Neutral fixtures — NEVER modified by tests. Tests always operate on tmp copies
(``copy_eval_wiki`` / ``eval_wiki`` / ``eval_env`` / ``copy_broken_wiki``).

- ``EVAL_WIKI_SRC``: 21 clean, AGENTS.md-conformant pages seeded from the repo's
  ``wiki copy/`` tree; ``health_check`` reports ZERO issues on it.
- ``EVAL_RAW_SRC``: ``sample.md`` (normal source) + ``contradiction-source.md``
  (claims that conflict with the corpus page ``entities/mlx``).
- ``EVAL_BROKEN_WIKI_SRC``: tiny wiki with three pinned defects
  (missing-frontmatter on ``entities/broken-fm``, broken-link on
  ``entities/linker``, missing-related on ``entities/lonely``) used by the
  fix-agent evals; the clean corpus is never used for fix evals.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# --- Committed fixture roots (read-only sources of truth) ---------------------
EVAL_WIKI_SRC: Path = Path(__file__).parent / "eval_wiki"
EVAL_RAW_SRC: Path = Path(__file__).parent / "eval_raw"
EVAL_BROKEN_WIKI_SRC: Path = Path(__file__).parent / "eval_broken_wiki"

# --- Retrieval calibration ----------------------------------------------------
# The 15 curated (query, ground-truth-slug) pairs pinned in the old
# tests/eval/test_search_recall.py, reused verbatim. Every ground-truth slug
# exists in EVAL_WIKI_SRC (guarded by the corpus self-check).
CURATED_QUERIES: list[tuple[str, str]] = [
    ("how do neural networks learn from examples", "concepts/machine-learning"),
    ("Apple's matrix math framework for its chips", "entities/mlx"),
    ("city in southern Spain with a university", "entities/málaga"),
    ("quantized fine tuning of language models", "concepts/llm-fine-tuning-with-qlora"),
    ("calling functions and tools from an LLM", "concepts/tool-calling"),
    ("real estate ownership on a blockchain", "concepts/real-estate-tokenization"),
    (
        "design philosophy integrating safety guardrails into AI systems from the outset",
        "concepts/safe-by-design-ai",
    ),
    ("high-performance ARM processors by Apple", "entities/apple-silicon"),
    ("Microsoft cloud platform", "entities/azure"),
    ("ethereum sidechain for digital assets", "entities/polygon-network"),
    ("general purpose interpreted programming language", "entities/python"),
    ("automatic workflows driven by AI", "concepts/ai-workflow-automation"),
    ("system that continuously improves itself", "concepts/continuous-improvement-system"),
    ("gaussian splatting 3D scene rendering", "concepts/3d-gaussian-splatting"),
    ("distilled transformer for classification", "entities/modernbert"),
]

# Harder variants — typos / synonyms / cross-type paraphrases that never mirror
# the page title verbatim. Measured recall@8 over EVAL_WIKI_SRC: 6/6 (1.00).
HARD_QUERIES: list[tuple[str, str]] = [
    ("learn how neurl nets traing on examples", "concepts/machine-learning"),
    ("metal-accelerated tensor kit from cupertino", "entities/mlx"),
    ("compress a big model and add small adapters", "concepts/llm-fine-tuning-with-qlora"),
    ("owning shares of buildings through tokens", "concepts/real-estate-tokenization"),
    ("microsft cloud servcies", "entities/azure"),
    ("snake-named general purpose scripting language", "entities/python"),
]

RECALL_K: int = 8
CURATED_RECALL_FLOOR: float = 0.80  # spec-pinned minimum; measured 15/15 (1.00)
HARD_RECALL_FLOOR: float = 0.60  # spec-pinned minimum; measured 6/6 (1.00)


def copy_eval_wiki(tmp_path: Path) -> Path:
    """Copy the committed clean corpus to ``tmp_path/eval_wiki`` and return it.

    Tests never touch ``EVAL_WIKI_SRC`` itself — always operate on the copy.
    """
    dest = tmp_path / "eval_wiki"
    shutil.copytree(EVAL_WIKI_SRC, dest)
    return dest


def copy_broken_wiki(tmp_path: Path) -> Path:
    """Copy the committed broken-wiki fixture to ``tmp_path/broken_wiki``."""
    dest = tmp_path / "broken_wiki"
    shutil.copytree(EVAL_BROKEN_WIKI_SRC, dest)
    return dest


@pytest.fixture
def eval_wiki(tmp_path: Path) -> Path:
    """A writable tmp copy of the clean eval corpus."""
    return copy_eval_wiki(tmp_path)


@pytest.fixture
def eval_env(tmp_path: Path) -> tuple[Path, Path]:
    """Copies of BOTH the clean corpus and the raw sources; returns
    ``(wiki_path, raw_path)``. Used by ingest/query evals that need a full
    environment (pages + sources to ingest).
    """
    return copy_eval_wiki(tmp_path), copy_eval_raw(tmp_path)


def copy_eval_raw(tmp_path: Path) -> Path:
    """Copy the committed raw-source fixtures to ``tmp_path/eval_raw``."""
    dest = tmp_path / "eval_raw"
    shutil.copytree(EVAL_RAW_SRC, dest)
    return dest
