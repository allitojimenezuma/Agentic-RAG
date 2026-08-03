"""Eval: recall@8 of ``wiki.search`` against the LIVE wiki.

Pure deterministic eval (no LLM, no network, no timers): a curated ~15-query
set with ground-truth slugs drawn from the live ``wiki/`` pages. Asserts every
query's ground-truth page appears in the top-8 results and that overall
recall@8 is >= 0.8 (Pass B Acceptance).
"""

from __future__ import annotations

from pathlib import Path

from agentic_rag.wiki.model import load_wiki
from agentic_rag.wiki.search import search

# Repo root: tests/eval/<this file> -> parents[2]. Live wiki data dir.
WIKI_DIR = Path(__file__).parents[2] / "wiki"

# Curated queries — natural-language phrasings (NOT the page titles) so the
# eval actually measures retrieval, each with its expected page slug.
CURATED_QUERIES: list[tuple[str, str]] = [
    ("how do neural networks learn from examples", "concepts/machine-learning"),
    ("Apple's matrix math framework for its chips", "entities/mlx"),
    ("city in southern Spain with a university", "entities/málaga"),
    ("quantized fine tuning of language models", "concepts/llm-fine-tuning-with-qlora"),
    ("calling functions and tools from an LLM", "concepts/tool-calling"),
    ("real estate ownership on a blockchain", "concepts/real-estate-tokenization"),
    ("designing agents that call tools safely", "concepts/safe-by-design-ai"),
    ("high-performance ARM processors by Apple", "entities/apple-silicon"),
    ("Microsoft cloud platform", "entities/azure"),
    ("ethereum sidechain for digital assets", "entities/polygon-network"),
    ("general purpose interpreted programming language", "entities/python"),
    ("automatic workflows driven by AI", "concepts/ai-workflow-automation"),
    ("system that continuously improves itself", "concepts/continuous-improvement-system"),
    ("gaussian splatting 3D scene rendering", "concepts/3d-gaussian-splatting"),
    ("distilled transformer for classification", "entities/modernbert"),
]

RECALL_THRESHOLD = 0.8


def test_ground_truth_pages_exist_in_live_wiki() -> None:
    """Every expected slug must resolve in the live wiki (guard against drift)."""
    wiki = load_wiki(WIKI_DIR)
    assert len(wiki.pages) >= len(CURATED_QUERIES)
    missing = [slug for _, slug in CURATED_QUERIES if slug not in wiki.by_slug]
    assert not missing, f"ground-truth slugs missing from live wiki: {missing}"


def test_recall_at_8_on_curated_queries() -> None:
    """Each curated query must surface its ground-truth page within top-8 hits."""
    wiki = load_wiki(WIKI_DIR)
    for query, expected_slug in CURATED_QUERIES:
        hits = search(wiki, query, k=8)
        hit_slugs = {h.slug for h in hits}
        assert expected_slug in hit_slugs, (
            f"recall miss: {query!r} -> expected {expected_slug}, "
            f"got top-8 {[h.slug for h in hits]}"
        )


def test_overall_recall_meets_threshold() -> None:
    """Aggregate recall@8 over the curated set must be >= 0.8 (spec)."""
    wiki = load_wiki(WIKI_DIR)
    hits_count = sum(
        expected_slug in {h.slug for h in search(wiki, query, k=8)}
        for query, expected_slug in CURATED_QUERIES
    )
    recall = hits_count / len(CURATED_QUERIES)
    assert recall >= RECALL_THRESHOLD, (
        f"recall@8 = {recall:.2f} ({hits_count}/{len(CURATED_QUERIES)}) "
        f"below threshold {RECALL_THRESHOLD}"
    )
