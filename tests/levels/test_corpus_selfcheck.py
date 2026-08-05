"""Corpus self-checks (T1) — the committed fixtures are valid and neutral.

Deterministic tier: zero LLM calls, zero network. Asserts:

1. ``health_check`` reports ZERO issues on the committed ``eval_wiki/`` corpus
   (and on a tmp copy, proving the copy path preserves conformance).
2. Every ``CURATED_QUERIES`` / ``HARD_QUERIES`` ground-truth slug resolves in
   the corpus.
3. ``eval_broken_wiki/`` reports EXACTLY its three seeded defects
   (missing-frontmatter / broken-link / missing-related), nothing else.
4. The pinned recall floors are reachable on the neutral corpus — measured
   recall@8: curated 15/15 = 1.00 (floor 0.80), hard 6/6 = 1.00 (floor 0.60).
5. Neutrality: searching over tmp copies never mutates the committed trees
   (file hashes unchanged across a full search pass).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agentic_rag.lint.health import health_check
from agentic_rag.wiki.model import load_wiki
from agentic_rag.wiki.search import search
from tests.fixtures.eval_corpus import (
    CURATED_QUERIES,
    CURATED_RECALL_FLOOR,
    EVAL_BROKEN_WIKI_SRC,
    EVAL_RAW_SRC,
    EVAL_WIKI_SRC,
    HARD_QUERIES,
    HARD_RECALL_FLOOR,
    RECALL_K,
    copy_broken_wiki,
    copy_eval_wiki,
)

# Pinned defect set of the broken-wiki fixture (kind, slug) — the self-check
# asserts the fixture reports exactly these and no other issues.
EXPECTED_BROKEN_DEFECTS: set[tuple[str, str]] = {
    ("missing-frontmatter", "entities/broken-fm"),
    ("broken-link", "entities/linker"),
    ("missing-related", "entities/lonely"),
}


def _tree_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every file under ``root`` (stable key: rel path)."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --- eval_wiki: clean + conformant -------------------------------------------


def test_eval_wiki_health_check_zero_issues() -> None:
    """Committed clean corpus must be lint-clean (21 pages, zero issues)."""
    report = health_check(EVAL_WIKI_SRC)
    assert report.pages_audited == 21
    assert report.issues == [], [f"{i.kind}:{i.slug} ({i.detail})" for i in report.issues]
    assert report.counts == {}


def test_eval_wiki_copy_is_health_clean(tmp_path: Path) -> None:
    """The tmp-copy path used by every test preserves zero-issue conformance."""
    assert health_check(copy_eval_wiki(tmp_path)).issues == []


def test_eval_wiki_has_required_layout() -> None:
    """Corpus contains entities + concepts pages plus index.md and log.md."""
    wiki = load_wiki(EVAL_WIKI_SRC)
    assert len(wiki.pages) >= 20
    assert (EVAL_WIKI_SRC / "index.md").is_file()
    assert (EVAL_WIKI_SRC / "log.md").is_file()
    assert any(p.slug.startswith("entities/") for p in wiki.pages)
    assert any(p.slug.startswith("concepts/") for p in wiki.pages)


# --- ground-truth slugs exist ------------------------------------------------


@pytest.mark.parametrize("query,expected_slug", CURATED_QUERIES)
def test_curated_ground_truth_slug_exists(query: str, expected_slug: str) -> None:
    """Every curated query's ground-truth slug must resolve in the corpus."""
    assert expected_slug in load_wiki(EVAL_WIKI_SRC).by_slug, (
        f"curated ground-truth slug missing: {expected_slug!r} (query: {query!r})"
    )


@pytest.mark.parametrize("query,expected_slug", HARD_QUERIES)
def test_hard_ground_truth_slug_exists(query: str, expected_slug: str) -> None:
    """Every hard query's ground-truth slug must resolve in the corpus."""
    assert expected_slug in load_wiki(EVAL_WIKI_SRC).by_slug, (
        f"hard ground-truth slug missing: {expected_slug!r} (query: {query!r})"
    )


# --- broken-wiki fixture: exactly the seeded defects --------------------------


def test_broken_wiki_reports_exact_seeded_defects(tmp_path: Path) -> None:
    """broken-wiki must report exactly the three pinned (kind, slug) defects."""
    report = health_check(copy_broken_wiki(tmp_path))
    actual = {(i.kind, i.slug) for i in report.issues}
    assert actual == EXPECTED_BROKEN_DEFECTS, (
        f"broken-wiki defect drift: expected {EXPECTED_BROKEN_DEFECTS}, got {actual}"
    )


# --- recall floors are reachable on the neutral corpus ------------------------


def test_curated_recall_meets_floor(tmp_path: Path) -> None:
    """Curated recall@8 on the committed corpus.

    Measured 2026-08-05: 15/15 = 1.00 >= CURATED_RECALL_FLOOR (0.80).
    """
    wiki = load_wiki(copy_eval_wiki(tmp_path))
    hits = sum(
        expected in {h.slug for h in search(wiki, query, k=RECALL_K)}
        for query, expected in CURATED_QUERIES
    )
    recall = hits / len(CURATED_QUERIES)
    assert recall >= CURATED_RECALL_FLOOR, (
        f"curated recall@8 = {recall:.2f} ({hits}/{len(CURATED_QUERIES)}) "
        f"below floor {CURATED_RECALL_FLOOR}"
    )


def test_hard_recall_meets_floor(tmp_path: Path) -> None:
    """Hard (typo/synonym/cross-type) recall@8 on the committed corpus.

    Measured 2026-08-05: 6/6 = 1.00 >= HARD_RECALL_FLOOR (0.60).
    """
    wiki = load_wiki(copy_eval_wiki(tmp_path))
    hits = sum(
        expected in {h.slug for h in search(wiki, query, k=RECALL_K)}
        for query, expected in HARD_QUERIES
    )
    recall = hits / len(HARD_QUERIES)
    assert recall >= HARD_RECALL_FLOOR, (
        f"hard recall@8 = {recall:.2f} ({hits}/{len(HARD_QUERIES)}) "
        f"below floor {HARD_RECALL_FLOOR}"
    )


# --- neutrality ---------------------------------------------------------------


def test_recall_searching_does_not_mutate_committed_corpus(tmp_path: Path) -> None:
    """A full search pass over tmp copies leaves the committed trees untouched."""
    before = {
        root: _tree_hashes(root)
        for root in (EVAL_WIKI_SRC, EVAL_RAW_SRC, EVAL_BROKEN_WIKI_SRC)
    }

    wiki = load_wiki(copy_eval_wiki(tmp_path))
    for query, _expected in CURATED_QUERIES + HARD_QUERIES:
        search(wiki, query, k=RECALL_K)

    assert _tree_hashes(EVAL_WIKI_SRC) == before[EVAL_WIKI_SRC]
    assert _tree_hashes(EVAL_RAW_SRC) == before[EVAL_RAW_SRC]
    assert _tree_hashes(EVAL_BROKEN_WIKI_SRC) == before[EVAL_BROKEN_WIKI_SRC]
