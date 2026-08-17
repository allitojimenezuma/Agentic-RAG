"""Tools for the lint agent: the report writer.

The deterministic structural audit lives behind the ``health`` sub-command of
``tools.nav.wiki_command`` (0 LLM calls); this module holds the only write
tool the lint agent has — the report file.
"""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.tools import tool

from agentic_rag.wiki.health import _render_report_markdown
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