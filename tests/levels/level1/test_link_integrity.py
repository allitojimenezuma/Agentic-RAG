"""Level 1 — link integrity over the eval corpus (0 LLM, deterministic).

On a tmp copy of the committed clean corpus (``eval_wiki`` fixture):

1. ``health_check`` re-asserts ZERO issues on the copy (it already passes on
   the committed tree; the copy proves the copy path preserves integrity).
2. EVERY raw ``[[target]]`` extracted from every page's raw text (via
   ``markdown_parser.extract_links``) resolves against the wiki's slug set —
   exact slug match in ``wiki.by_slug`` or, failing that, the ``_resolve_link``
   semantics (slugified short-name, then unicode-preserving lowercase/hyphen).
   This goes FURTHER than ``health_check``: we walk the full raw link set,
   not just the flagged subset.
3. Every page has at least one INBOUND link from another page (no orphans) —
   asserted via the ``Page.outbound_links`` map (resolved, self-links already
   dropped by the model), mirroring ``health_check``'s orphan computation.

No LLM, no network, no ``Settings`` — pure deterministic engine + raw-text
parsing on a tmp copy.
"""

from __future__ import annotations

from pathlib import Path

from agentic_rag.io.markdown_parser import extract_links
from agentic_rag.wiki.health import health_check
from agentic_rag.wiki.model import _resolve_link, load_wiki


def test_health_check_reports_zero_issues_on_copy(eval_wiki: Path):
    """The tmp copy of the clean corpus is structurally clean (re-assert on
    the copy, per corpus discipline)."""
    report = health_check(eval_wiki)
    assert report.issues == [], (
        f"health_check found issues on the clean corpus copy: "
        f"{[(i.kind, i.slug) for i in report.issues]}"
    )
    assert report.pages_audited >= 20  # sanity: the corpus was actually loaded


def test_every_raw_link_resolves(eval_wiki: Path):
    """FURTHER than health_check: every [[target]] in every page's raw text
    resolves to an existing slug (exact match or _resolve_link semantics)."""
    wiki = load_wiki(eval_wiki)
    all_slugs = set(wiki.by_slug)
    assert all_slugs, "corpus loaded zero pages"

    for page in wiki.pages:
        raw = page.rel_path.read_text(encoding="utf-8")
        for link in extract_links(raw):
            if link.target.startswith("lint-report"):
                continue  # derived artifacts, mirrors health_check
            assert _resolve_link(link.target, all_slugs) is not None, (
                f"{page.slug}: raw link [[{link.target}]] does not resolve"
            )


def test_exact_slug_links_resolve_by_slug(eval_wiki: Path):
    """Slugs that appear verbatim in the wiki resolve through by_slug (the
    primary resolution path)."""
    wiki = load_wiki(eval_wiki)
    for page in wiki.pages:
        raw = page.rel_path.read_text(encoding="utf-8")
        for link in extract_links(raw):
            if link.target in wiki.by_slug:
                continue
            # non-exact targets must still resolve via _resolve_link's
            # display-name semantics (slugify / unicode-preserving).
            assert _resolve_link(link.target, set(wiki.by_slug)) is not None, (
                f"{page.slug}: display-name link [[{link.target}]] does not resolve"
            )


def test_every_page_has_inbound_link(eval_wiki: Path):
    """No orphans: every content page is linked to from at least one other
    page, computed from the model's resolved outbound_links map."""
    wiki = load_wiki(eval_wiki)
    inbound: dict[str, list[str]] = {slug: [] for slug in wiki.by_slug}
    for page in wiki.pages:
        if page.rel_path.name.startswith("lint-report-"):
            continue  # derived artifacts never count as content (health_check)
        for target in page.outbound_links:
            if target in inbound:
                inbound[target].append(page.slug)

    for slug, sources in inbound.items():
        assert sources, (
            f"{slug} is an orphan: no inbound links from any other page"
        )


def test_outbound_links_are_resolved_slugs(eval_wiki: Path):
    """Every entry in a page's outbound_links map is a real slug in the wiki
    (the model never emits unresolvable or self targets)."""
    wiki = load_wiki(eval_wiki)
    all_slugs = set(wiki.by_slug)
    for page in wiki.pages:
        for target in page.outbound_links:
            assert target in all_slugs, (
                f"{page.slug}: outbound_links contains non-slug {target!r}"
            )
            assert target != page.slug, (
                f"{page.slug}: outbound_links contains a self-link"
            )
