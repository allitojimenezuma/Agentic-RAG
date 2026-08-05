"""Level 3 — Context Recall@K on the neutral committed corpus (0 LLM, headless).

Per-query and aggregate recall@8 over the committed ``EVAL_WIKI_SRC`` corpus,
always operating on a tmp copy (``eval_wiki`` fixture) — never the committed
tree. The corpus is neutral and immutable: recall thresholds are calibrated to
the retriever (never the corpus to the thresholds), no query phrasing or
threshold is ever adjusted to make a test pass, and this suite proves it never
mutates the corpus by hashing every file under the copy before and after a
full search pass.

Measured recall@8 on the committed corpus (re-measured by this suite):
- curated 15 queries: 15/15 = 1.00  (pinned floor CURATED_RECALL_FLOOR = 0.80)
- hard    6 queries:   6/6  = 1.00  (pinned floor HARD_RECALL_FLOOR = 0.60)

The floors comfortably pass; a failure here means corpus drift, and the corpus
must never be edited to chase a threshold — report the miss instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agentic_rag.wiki.model import load_wiki
from agentic_rag.wiki.search import search
from tests.fixtures.eval_corpus import (
    CURATED_QUERIES,
    CURATED_RECALL_FLOOR,
    HARD_QUERIES,
    HARD_RECALL_FLOOR,
    RECALL_K,
)


def _tree_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root`` (stable key: rel path).

    Mirrors ``tests/levels/test_corpus_selfcheck.py._tree_hashes`` so both
    suites measure corpus neutrality the same way.
    """
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _top_k_slugs(wiki, query: str) -> list[str]:
    """The top-``RECALL_K`` slugs for ``query`` in retrieval order."""
    return [h.slug for h in search(wiki, query, k=RECALL_K)]


# --- per-query recall: every ground-truth slug must be in the top-k -----------


@pytest.mark.parametrize("query,expected_slug", CURATED_QUERIES)
def test_curated_query_recalls_ground_truth(
    query: str, expected_slug: str, eval_wiki: Path
) -> None:
    """Curated query must surface its ground-truth page in the top-8 hits."""
    top_k = _top_k_slugs(load_wiki(eval_wiki), query)
    assert expected_slug in top_k, (
        f"curated query {query!r} MISSED ground truth {expected_slug!r}; "
        f"actual top-{RECALL_K} slugs: {top_k}"
    )


@pytest.mark.parametrize("query,expected_slug", HARD_QUERIES)
def test_hard_query_recalls_ground_truth(
    query: str, expected_slug: str, eval_wiki: Path
) -> None:
    """Hard (typo/synonym/cross-type) query must surface its ground truth."""
    top_k = _top_k_slugs(load_wiki(eval_wiki), query)
    assert expected_slug in top_k, (
        f"hard query {query!r} MISSED ground truth {expected_slug!r}; "
        f"actual top-{RECALL_K} slugs: {top_k}"
    )


# --- aggregate recall@k floors -------------------------------------------------


def test_curated_aggregate_recall_meets_floor(eval_wiki: Path) -> None:
    """Curated recall@8 >= CURATED_RECALL_FLOOR (0.80).

    Measured 2026-08-05: 15/15 = 1.00.
    """
    wiki = load_wiki(eval_wiki)
    hits = sum(
        expected in {h.slug for h in search(wiki, query, k=RECALL_K)}
        for query, expected in CURATED_QUERIES
    )
    recall = hits / len(CURATED_QUERIES)
    assert recall >= CURATED_RECALL_FLOOR, (
        f"curated recall@8 = {recall:.2f} ({hits}/{len(CURATED_QUERIES)}) "
        f"below pinned floor {CURATED_RECALL_FLOOR}"
    )


def test_hard_aggregate_recall_meets_floor(eval_wiki: Path) -> None:
    """Hard recall@8 >= HARD_RECALL_FLOOR (0.60).

    Measured 2026-08-05: 6/6 = 1.00.
    """
    wiki = load_wiki(eval_wiki)
    hits = sum(
        expected in {h.slug for h in search(wiki, query, k=RECALL_K)}
        for query, expected in HARD_QUERIES
    )
    recall = hits / len(HARD_QUERIES)
    assert recall >= HARD_RECALL_FLOOR, (
        f"hard recall@8 = {recall:.2f} ({hits}/{len(HARD_QUERIES)}) "
        f"below pinned floor {HARD_RECALL_FLOOR}"
    )


# --- neutrality: recall tests never mutate the corpus --------------------------


def test_full_search_pass_does_not_mutate_tmp_copy(eval_wiki: Path) -> None:
    """Hashing every file before/after a full search pass proves no writes."""
    before = _tree_hashes(eval_wiki)

    wiki = load_wiki(eval_wiki)
    for query, _expected in CURATED_QUERIES + HARD_QUERIES:
        search(wiki, query, k=RECALL_K)

    after = _tree_hashes(eval_wiki)
    mutated = [path for path in sorted(before) if before[path] != after.get(path)]
    assert after == before, f"search pass mutated corpus files: {mutated}"
