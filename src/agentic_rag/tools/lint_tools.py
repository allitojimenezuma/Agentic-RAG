"""Tools for the lint agent: deterministic health check + report writer."""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.tools import tool

from agentic_rag.lint.health import _render_report_markdown, health_check
from agentic_rag.schemas.lint import LintReport
from agentic_rag.tools.shared import get_wiki_path

logger = logging.getLogger(__name__)


@tool
def write_lint_report(report: LintReport | str) -> str:
    """Write a lint report to wiki/lint-report-YYYY-MM-DD.md with today's date.

    Accepts either a structured ``LintReport`` (rendered deterministically via
    ``_render_report_markdown``) or a raw markdown string (back-compat for the
    scripted test harness)."""
    today = date.today().isoformat()
    report_path = get_wiki_path() / f"lint-report-{today}.md"
    content = report if isinstance(report, str) else _render_report_markdown(report)
    logger.debug("Writing lint report to %s", report_path)
    report_path.write_text(content, encoding="utf-8")
    return f"Lint report written to: lint-report-{today}.md"


@tool
def run_health_check() -> str:
    """Run a deterministic structural health check of the wiki: orphan, missing-index, broken-link, missing-frontmatter, missing-related, empty, and stale pages. Zero LLM calls — instant and free. Call this FIRST in any lint run; it is the ground truth for structural findings."""
    report = health_check(get_wiki_path())
    lines = [f"Pages audited: {report.pages_audited} | Issues: {len(report.issues)}"]
    lines.extend(
        f"[{issue.severity}] {issue.kind}: {issue.slug} — {issue.detail}"
        for issue in report.issues
    )
    return "\n".join(lines)
