"""Deterministic wiki health check — 0 LLM calls.

Computes the structural lint issues (orphan, missing-index, broken-link,
missing-frontmatter, missing-related, empty, stale) from the source-of-truth
:class:`~agentic_rag.wiki.model.Wiki` model + the regenerated index. The one
thing this cannot do is semantic judgment (e.g. duplicate coverage), which is
left to the lint agent's LLM pass.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from agentic_rag.io.index_manager import read_index
from agentic_rag.io.markdown_parser import extract_links, slugify
from agentic_rag.schemas.lint import Issue, LintReport
from agentic_rag.wiki.model import _resolve_link, load_wiki

logger = logging.getLogger(__name__)

_STALE_DAYS = 90
_EMPTY_WORDS = 50

# Deterministic per-kind severity (lint severity rules: broken schema
# compliance = critical; orphan/missing-index/broken-link/empty = high;
# missing-related/stale = medium).
_KIND_SEVERITY = {
    "orphan": "high",
    "missing-index": "high",
    "broken-link": "high",
    "missing-frontmatter": "critical",
    "missing-related": "medium",
    "empty": "high",
    "stale": "medium",
}

_KIND_ORDER = ("orphan", "missing-index", "broken-link", "missing-frontmatter", "missing-related", "empty", "stale")


def health_check(wiki_path: Path) -> LintReport:
    """Audit a wiki directory and return a structured :class:`LintReport`.

    - ``lint-report-*.md`` pages are treated as non-content: excluded from
      ``pages_audited``, from inbound/orphan computation, and never flagged.
    - Deterministic: issues sorted by ``(kind, slug)``; ``counts`` holds only
      kinds that occurred.
    """
    wiki = load_wiki(wiki_path)
    content_pages = [p for p in wiki.pages if not _is_lint_report(p)]
    content_slugs = {p.slug for p in content_pages}
    all_slugs = set(wiki.by_slug)

    # Inbound links: from content pages' RESOLVED outbound links only.
    inbound: dict[str, set[str]] = defaultdict(set)
    for page in content_pages:
        for target in page.outbound_links:
            if target in content_slugs:
                inbound[target].add(page.slug)

    # Most recent fm.updated across content pages (stale reference point).
    latest = max((p.fm.updated for p in content_pages), default=None)

    # Index slugs (display-name-derived; compared via short slug / title slug).
    index = read_index(wiki_path)
    index_slugs = {e.slug for entries in index.categories.values() for e in entries}

    issues: list[Issue] = []
    for page in sorted(content_pages, key=lambda p: p.slug):
        slug = page.slug
        raw = page.rel_path.read_text(encoding="utf-8")

        if not inbound.get(slug):
            issues.append(_issue(slug, "orphan", "No inbound links from other pages",
                                 "Add a [[link]] from a related page"))

        if not _page_in_index(page, index_slugs):
            issues.append(_issue(slug, "missing-index", "Page not listed in index.md",
                                 "Add an entry to index.md"))

        # Broken links need RAW targets (model outbound_links are resolved slugs).
        broken = [
            f"[[{link.target}]]"
            for link in extract_links(raw)
            if link.target != slug
            and not link.target.startswith("lint-report")
            and _resolve_link(link.target, all_slugs) is None
        ]
        if broken:
            issues.append(_issue(slug, "broken-link",
                                 "Unresolved link(s): " + ", ".join(broken),
                                 "Fix or remove the [[link]]"))

        if not raw.startswith("---"):
            issues.append(_issue(slug, "missing-frontmatter", "Page lacks YAML frontmatter",
                                 "Add frontmatter with slug/type/title/updated"))

        if not any(s.heading.lower() == "related" for s in page.sections):
            issues.append(_issue(slug, "missing-related", "No 'Related' section",
                                 "Add a ## Related section with cross-links"))

        if page.word_count < _EMPTY_WORDS:
            issues.append(_issue(slug, "empty",
                                 f"Only {page.word_count} words of content (<{_EMPTY_WORDS})",
                                 "Add substantive content"))

        if latest is not None and page.fm.updated < latest - timedelta(days=_STALE_DAYS):
            age = (latest - page.fm.updated).days
            issues.append(_issue(slug, "stale",
                                 f"Updated {age} days before the most recent page (>{_STALE_DAYS} days)",
                                 "Review and update content"))

    issues.sort(key=lambda i: (i.kind, i.slug))
    counts = {kind: n for kind, n in Counter(i.kind for i in issues).items() if n}
    report = LintReport(pages_audited=len(content_pages), issues=issues, counts=counts)
    logger.info(
        "Health check: %d pages audited, %d issues (%s)",
        len(content_pages), len(issues), counts,
    )
    return report


def _is_lint_report(page) -> bool:
    """Lint-report pages are derived artifacts, not wiki content."""
    return page.rel_path.name.startswith("lint-report-")


def _issue(slug: str, kind: str, detail: str, action: str) -> Issue:
    return Issue(
        slug=slug,
        kind=kind,
        severity=_KIND_SEVERITY[kind],
        detail=detail,
        action=action,
    )


def _page_in_index(page, index_slugs: set[str]) -> bool:
    """Is this page represented in the regenerated index?

    ``read_index`` derives entry slugs from the display name (e.g.
    ``[[Artificial Intelligence]]`` -> ``artificial-intelligence``), so a page
    is "in the index" if its full slug, short slug, or title-slug appears.
    """
    if page.slug in index_slugs:
        return True
    short = page.slug.rsplit("/", 1)[-1]
    if short in index_slugs:
        return True
    return slugify(page.fm.title) in index_slugs


def _render_report_markdown(report: LintReport) -> str:
    """Deterministic markdown report in the style of the lint agent's format.

    Executive summary with severity counts, per-severity issue sections
    (C/H/M/L numbered), and a summary statistics table. Used by
    ``write_lint_report`` for the no-LLM path.
    """
    today = date.today().isoformat()
    lines = [f"# Wiki Lint Report — {today}", ""]

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Pages audited: {report.pages_audited}")
    for sev in ("critical", "high", "medium", "low"):
        n = sum(1 for i in report.issues if i.severity == sev)
        lines.append(f"- {sev.capitalize()} issues: {n}")
    lines.append("")

    for sev in ("critical", "high", "medium", "low"):
        sev_issues = [i for i in report.issues if i.severity == sev]
        lines.append(f"## {sev.capitalize()} Issues")
        lines.append("")
        if not sev_issues:
            lines.append("None.")
            lines.append("")
            continue
        for idx, issue in enumerate(sev_issues, start=1):
            label = sev[0].upper()
            lines.append(f"### {label}{idx}. {issue.kind.replace('-', ' ').title()}")
            lines.append(f"- **Affected:** {issue.slug}")
            lines.append(f"- **Finding:** {issue.detail}")
            lines.append(f"- **Action:** {issue.action}")
            lines.append("")

    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total pages | {report.pages_audited} |")
    for kind in _KIND_ORDER:
        label = {
            "orphan": "Orphan pages",
            "missing-index": "Missing index entries",
            "broken-link": "Broken links",
            "missing-frontmatter": "Missing frontmatter",
            "missing-related": "Missing Related sections",
            "empty": "Empty pages",
            "stale": "Stale pages (>90 days)",
        }[kind]
        lines.append(f"| {label} | {report.counts.get(kind, 0)} |")
    lines.append("")
    return "\n".join(lines)
