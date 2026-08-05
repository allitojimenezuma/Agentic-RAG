"""Unit tests for the deterministic wiki health check (0 LLM calls)."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

from agentic_rag.io.markdown_parser import serialize_frontmatter
from agentic_rag.wiki.health import _render_report_markdown, health_check
from agentic_rag.schemas.lint import LintReport
from agentic_rag.schemas.wiki import Frontmatter
from agentic_rag.tools.lint_tools import write_lint_report
from agentic_rag.tools.shared import init_shared_tools

REF_DATE = date(2026, 7, 1)
STALE_DATE = date(2025, 6, 1)

# Long-enough intro paragraph so the <50-word check does not fire (except on empty-page).
_LONG = (
    "This page contains a sufficiently long body of text so that the deterministic "
    "word count check does not flag it as empty. It describes its topic in enough "
    "detail to be considered real content with many words and useful information "
    "for the health check, which only flags pages below the fifty word threshold."
)


def _content(title: str, related: str | None = None, extra: str = "") -> str:
    """A body with the long intro paragraph and an optional Related section."""
    body = f"# {title}\n\n{_LONG}\n\n{extra}".rstrip()
    if related:
        body += f"\n\n## Related\n\n{related}\n"
    return body


def _write_page(wiki: Path, slug: str, body: str, updated: date | None = REF_DATE) -> None:
    """Write a page with optional frontmatter (None => no frontmatter)."""
    path = wiki / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if updated is None:
        path.write_text(body, encoding="utf-8")
        ts = datetime(REF_DATE.year, REF_DATE.month, REF_DATE.day).timestamp()
        os.utime(path, (ts, ts))  # synthesized fm.updated = mtime = REF_DATE
    else:
        fm = Frontmatter(slug=slug, type="concept", title=slug.rsplit("/", 1)[-1],
                         sources=[], updated=updated, tags=[])
        path.write_text(serialize_frontmatter(fm) + body, encoding="utf-8")


def _build_wiki(tmp_path: Path) -> Path:
    wiki = tmp_path / "wiki"
    for sub in ("entities", "concepts", "sources", "comparisons"):
        (wiki / sub).mkdir(parents=True, exist_ok=True)

    # Normal page: frontmatter, Related, >=50 words, in index, 6 inbound links.
    _write_page(wiki, "entities/python",
                _content("Python", related="- [[concepts/frontmatterless]]\n"
                         "- [[concepts/empty-page]]\n- [[entities/stale-entity]]\n"
                         "- [[concepts/broken-link-page]]\n- [[concepts/no-related-page]]"))

    # Orphan: no inbound links from other content pages.
    _write_page(wiki, "concepts/ml",
                _content("ML", related="- [[entities/python]]"))

    # Missing frontmatter: raw content does not start with '---'.
    _write_page(wiki, "concepts/frontmatterless",
                _content("Frontmatterless", related="- [[entities/python]]"),
                updated=None)

    # Empty: fewer than 50 words of content.
    _write_page(wiki, "concepts/empty-page",
                "# Empty\n\nJust a few words here.\n\n## Related\n\n- [[entities/python]]\n")

    # Stale: updated >90 days before the most recent content page.
    _write_page(wiki, "entities/stale-entity",
                _content("Stale Entity", related="- [[entities/python]]"),
                updated=STALE_DATE)

    # Broken link: raw [[target]] resolves to no page slug.
    _write_page(wiki, "concepts/broken-link-page",
                _content("Broken Link Page",
                         related="- [[concepts/nonexistent-page]]\n- [[entities/python]]"))

    # Missing Related: no section heading equal to "Related" (case-insensitive).
    _write_page(wiki, "concepts/no-related-page",
                _content("No Related Page", extra="## Notes\n\n- Some notes here."))

    # Lint-report page: derived artifact, excluded from all content stats.
    (wiki / "lint-report-2026-07-26.md").write_text(
        "# Lint Report\n\nAll good.\n", encoding="utf-8")

    # Index covering all 7 content pages (read_index parses display-name slugs).
    (wiki / "index.md").write_text(
        "# Wiki Index\n\n## Entities\n"
        "- [[Python]] - A programming language | Sources: manual | Updated: 2026-07-01\n"
        "- [[Stale Entity]] - Old content | Sources: manual | Updated: 2025-06-01\n"
        "\n## Concepts\n"
        "- [[ML]] - Machine learning | Sources: manual | Updated: 2026-07-01\n"
        "- [[Frontmatterless]] - No frontmatter | Sources: manual | Updated: 2026-07-01\n"
        "- [[Empty Page]] - Short page | Sources: manual | Updated: 2026-07-01\n"
        "- [[Broken Link Page]] - Broken links | Sources: manual | Updated: 2026-07-01\n"
        "- [[No Related Page]] - Missing related | Sources: manual | Updated: 2026-07-01\n",
        encoding="utf-8")
    (wiki / "log.md").write_text("# Wiki Log\n", encoding="utf-8")
    return wiki


def _run(tmp_path):
    wiki = _build_wiki(tmp_path)
    return health_check(wiki)


class TestHealthCheck:
    def test_counts_and_audited_pages(self, tmp_path):
        report = _run(tmp_path)
        assert report.pages_audited == 7  # lint-report-*.md excluded
        assert report.counts == {
            "orphan": 1,
            "broken-link": 1,
            "missing-frontmatter": 1,
            "missing-related": 1,
            "empty": 1,
            "stale": 1,
        }
        assert "missing-index" not in report.counts  # all pages are in index.md

    def test_issue_kinds_per_slug(self, tmp_path):
        report = _run(tmp_path)
        kinds = {i.slug: i.kind for i in report.issues}
        assert kinds["concepts/ml"] == "orphan"
        assert kinds["concepts/broken-link-page"] == "broken-link"
        assert kinds["concepts/frontmatterless"] == "missing-frontmatter"
        assert kinds["concepts/no-related-page"] == "missing-related"
        assert kinds["concepts/empty-page"] == "empty"
        assert kinds["entities/stale-entity"] == "stale"
        assert "entities/python" not in kinds  # normal page has no issues

    def test_lint_report_page_excluded(self, tmp_path):
        report = _run(tmp_path)
        assert all(not i.slug.startswith("lint-report-") for i in report.issues)

    def test_parenthetical_and_accented_titles_register_in_index(self, tmp_path):
        """Regression for the 2026-08-05 report H2: pages whose display titles
        contain parentheses or accents were flagged missing-index although they
        ARE listed — read_index derived entry slugs without the canonical
        slugify normalization. Titles like 'CSAR (Cloud System Architecture for
        Robotics)' and 'Javier González Jiménez' must register."""
        wiki = tmp_path / "wiki"
        (wiki / "entities").mkdir(parents=True)
        pages = [
            ("entities/python", "Python", "- [[entities/csar-cloud-system-architecture-for-robotics]]\n- [[entities/javier-gonzalez-jimenez]]\n- [[entities/jose-raul-ruiz-sarmiento]]\n"),
            ("entities/csar-cloud-system-architecture-for-robotics", "CSAR (Cloud System Architecture for Robotics)", "- [[entities/python]]\n"),
            ("entities/javier-gonzalez-jimenez", "Javier González Jiménez", "- [[entities/python]]\n"),
            ("entities/jose-raul-ruiz-sarmiento", "José Raúl Ruiz Sarmiento", "- [[entities/python]]\n"),
        ]
        for slug, title, related in pages:
            fm = Frontmatter(slug=slug, type="entity", title=title, sources=[],
                             updated=REF_DATE, tags=[])
            body = _content(title, related=related)
            path = wiki / f"{slug}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialize_frontmatter(fm) + body, encoding="utf-8")
        # Index entries exactly as regenerate_index would write them: [[Title]].
        (wiki / "index.md").write_text(
            "# Wiki Index\n\n## Entities\n"
            "- [[Python]] - hub | Sources: manual | Updated: 2026-07-01\n"
            "- [[CSAR (Cloud System Architecture for Robotics)]] - Cloud infra | Sources: manual | Updated: 2026-07-01\n"
            "- [[Javier González Jiménez]] - Researcher | Sources: manual | Updated: 2026-07-01\n"
            "- [[José Raúl Ruiz Sarmiento]] - Researcher | Sources: manual | Updated: 2026-07-01\n",
            encoding="utf-8",
        )
        report = health_check(wiki)
        kinds = {i.slug: i.kind for i in report.issues}
        assert "missing-index" not in kinds.values()
        assert all(slug not in kinds for slug, _, _ in pages)

    def test_broken_link_detected_from_raw_target(self, tmp_path):
        report = _run(tmp_path)
        bi = next(i for i in report.issues if i.kind == "broken-link")
        assert "[[concepts/nonexistent-page]]" in bi.detail
        assert "Fix or remove" in bi.action

    def test_severity_mapping(self, tmp_path):
        report = _run(tmp_path)
        severities = {i.kind: i.severity for i in report.issues}
        assert severities["missing-frontmatter"] == "critical"
        assert severities["orphan"] == "high"
        assert severities["broken-link"] == "high"
        assert severities["empty"] == "high"
        assert severities["missing-related"] == "medium"
        assert severities["stale"] == "medium"

    def test_deterministic(self, tmp_path):
        wiki = _build_wiki(tmp_path)
        assert health_check(wiki).model_dump() == health_check(wiki).model_dump()

    def test_zero_llm_calls(self):
        import agentic_rag.wiki.health as health

        src = Path(health.__file__).read_text(encoding="utf-8")
        assert "langchain" not in src

    def test_render_report_markdown_contains_counts(self, tmp_path):
        report = _run(tmp_path)
        md = _render_report_markdown(report)
        assert "Pages audited: 7" in md
        assert "## Critical Issues" in md
        assert "### C1. Missing Frontmatter" in md
        assert "| Orphan pages | 1 |" in md
        assert "| Missing frontmatter | 1 |" in md
        assert "| Broken links | 1 |" in md

    def test_write_lint_report_accepts_model(self, tmp_path):
        wiki = _build_wiki(tmp_path)
        init_shared_tools(str(wiki))
        report = health_check(wiki)
        result = write_lint_report.invoke({"report": report})
        assert "Lint report written" in result
        today = date.today().isoformat()
        written = (wiki / f"lint-report-{today}.md").read_text(encoding="utf-8")
        assert "Pages audited: 7" in written
        assert "| Orphan pages | 1 |" in written
