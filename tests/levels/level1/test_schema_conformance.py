"""Level 1 — schema conformance over the eval corpus (0 LLM, deterministic).

Every page of the committed clean corpus (``eval_wiki`` fixture, a tmp copy of
``tests/fixtures/eval_wiki/``) must conform to the AGENTS.md wiki schema:

1. Raw YAML frontmatter carries ALL required keys — ``slug``, ``type``,
   ``title``, ``updated`` (the ``Frontmatter`` model's required fields) plus
   ``sources`` and ``tags`` (defaulted by ``Frontmatter`` but required by
   AGENTS.md to be present in the file).
2. The frontmatter ``slug`` matches the file path (``slug == <relpath>
   without ``.md````).
3. The frontmatter ``type`` matches the directory the page lives in
   (entity -> entities/, concept -> concepts/, source -> sources/,
   comparison -> comparisons/, overview -> wiki root).
4. Every page has a ``## Related`` section (heading lower() == "related"),
   the AGENTS.md cross-reference requirement.

Three corpus pages carry DOCUMENTED deviations from the strict schema. They
were inherited faithfully from the repo's ``wiki copy/`` tree when the corpus
was seeded (T1); the corpus is a frozen fixture and is NEVER "fixed" here.
Crucially, ``fm.slug`` is cosmetic in the engine: ``load_wiki`` derives the
canonical ``page.slug`` from the filename, so recall, link resolution and
health checks are unaffected by the short ``fm.slug`` values. The exact
deviation set is pinned in ``EXPECTED_DEVIATIONS`` and re-verified by
``test_pinned_deviations_match_reality`` — if the corpus is ever cleaned up
upstream, that guard fails loudly instead of silently passing.

No LLM, no network, no ``Settings`` — pure file + deterministic-engine
(``load_wiki`` + ``markdown_parser``) assertions on a tmp copy.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agentic_rag.io.markdown_parser import parse_frontmatter
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.wiki.model import load_wiki

# AGENTS.md-required frontmatter keys. ``sources``/``tags`` are defaulted by
# the Frontmatter model, but the wiki schema demands them in the raw file.
REQUIRED_FRONTMATTER_KEYS = {"slug", "type", "title", "updated", "sources", "tags"}

# type -> directory the page file must live in ("" = wiki root for overview).
_TYPE_DIR = {
    "entity": "entities",
    "concept": "concepts",
    "source": "sources",
    "comparison": "comparisons",
    "overview": "",
}

# Pinned, documented schema deviations in the committed corpus (inherited from
# ``wiki copy/``; corpus is frozen — see module docstring). Keys are the page's
# relpath; values explain the deviation. Never grows, never shrinks silently.
EXPECTED_DEVIATIONS = {
    "concepts/machine-learning.md": (
        'frontmatter slug "machine-learning" lacks the "concepts/" prefix '
        '(fm.slug is cosmetic; page.slug == "concepts/machine-learning")'
    ),
    "entities/python.md": (
        'frontmatter slug "python" lacks the "entities/" prefix '
        '(fm.slug is cosmetic; page.slug == "entities/python")'
    ),
    "entities/alvaro-jimenez-martinez.md": (
        'frontmatter type "concept" but the file lives in entities/'
    ),
}


def _raw_frontmatter(content: str) -> dict:
    """Return the RAW frontmatter mapping (no defaults filled in)."""
    parts = content.split("---", 2)
    assert len(parts) >= 3, "page must open with a --- ... --- frontmatter block"
    data = yaml.safe_load(parts[1])
    assert isinstance(data, dict), "frontmatter must be a YAML mapping"
    return data


def _pages(eval_wiki: Path):
    """Yield (page, relpath_str, raw_content, raw_fm) for every corpus page."""
    wiki = load_wiki(eval_wiki)
    for page in wiki.pages:
        raw = page.rel_path.read_text(encoding="utf-8")
        rel = str(page.rel_path.relative_to(eval_wiki))
        yield page, rel, raw, _raw_frontmatter(raw)


def _slug_mismatch(rel: str, raw_fm: dict) -> bool:
    """Does this page fail the slug<->filename conformance check?"""
    expected = rel.removesuffix(".md")
    return raw_fm["slug"] != expected


def _type_dir_mismatch(rel: str, raw_fm: dict) -> bool:
    """Does this page fail the type<->directory conformance check?"""
    parent = Path(rel).parent
    parent_str = str(parent) if str(parent) != "." else ""
    return _TYPE_DIR.get(raw_fm["type"]) != parent_str


def test_required_frontmatter_keys_present_in_raw(eval_wiki):
    """AGENTS.md required keys (slug/type/title/updated/sources/tags) all
    appear in the raw frontmatter of every page — no exceptions."""
    for _page, rel, _raw, raw_fm in _pages(eval_wiki):
        missing = REQUIRED_FRONTMATTER_KEYS - set(raw_fm)
        assert not missing, f"{rel}: raw frontmatter missing keys {sorted(missing)}"


def test_frontmatter_parses_into_model(eval_wiki):
    """The raw frontmatter round-trips through the Frontmatter model without
    defaulting away any required field."""
    for _page, rel, raw, _raw_fm in _pages(eval_wiki):
        fm = parse_frontmatter(raw)
        assert isinstance(fm, Frontmatter), f"{rel}: frontmatter did not parse"


def test_slug_matches_filename(eval_wiki):
    """frontmatter slug == file path relative to wiki root, minus .md — for
    every page except the pinned EXPECTED_DEVIATIONS (short fm.slug pages)."""
    for page, rel, _raw, raw_fm in _pages(eval_wiki):
        if rel in EXPECTED_DEVIATIONS:
            continue
        expected = rel.removesuffix(".md")
        assert raw_fm["slug"] == expected, (
            f"{rel}: frontmatter slug {raw_fm['slug']!r} != filename slug {expected!r}"
        )
        # load_wiki must derive the same canonical slug (engine source of truth).
        assert page.slug == expected


def test_type_matches_directory(eval_wiki):
    """frontmatter type implies the page's directory (entity->entities/, ...,
    overview -> wiki root) — for every page except the pinned deviation."""
    for _page, rel, _raw, raw_fm in _pages(eval_wiki):
        if rel in EXPECTED_DEVIATIONS:
            continue
        parent = str(Path(rel).parent) if str(Path(rel).parent) != "." else ""
        expected_dir = _TYPE_DIR.get(raw_fm["type"])
        assert expected_dir is not None, (
            f"{rel}: unknown page type {raw_fm['type']!r}"
        )
        assert parent == expected_dir, (
            f"{rel}: type {raw_fm['type']!r} expects dir {expected_dir!r}, "
            f"found {parent!r}"
        )


def test_every_page_has_related_section(eval_wiki):
    """AGENTS.md: every page has a ``## Related`` section — no exceptions."""
    for page, rel, _raw, _raw_fm in _pages(eval_wiki):
        assert any(s.heading.lower() == "related" for s in page.sections), (
            f"{rel}: missing '## Related' section"
        )


def test_corpus_has_no_lint_report_pages(eval_wiki):
    """Guard: the conformance loop would otherwise sweep derived artifacts."""
    for page, _rel, _raw, _raw_fm in _pages(eval_wiki):
        assert not page.rel_path.name.startswith("lint-report-")


def test_pinned_deviations_match_reality(eval_wiki):
    """The set of pages failing slug<->filename OR type<->directory must be
    EXACTLY EXPECTED_DEVIATIONS — if the corpus is ever fixed upstream, this
    fails loudly (guarding against silent drift in either direction)."""
    failing = {
        rel
        for _page, rel, _raw, raw_fm in _pages(eval_wiki)
        if _slug_mismatch(rel, raw_fm) or _type_dir_mismatch(rel, raw_fm)
    }
    assert failing == set(EXPECTED_DEVIATIONS), (
        f"Corpus deviation set drifted: expected {sorted(EXPECTED_DEVIATIONS)}, "
        f"found {sorted(failing)}"
    )
