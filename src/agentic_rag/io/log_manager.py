"""Log manager: append-only wiki/log.md with timestamped, prefixed entries."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from agentic_rag.schemas.wiki import LogEntry

# Pattern: ## [YYYY-MM-DD HH:MM] <op> | <title>
_LOG_PREFIX_RE = re.compile(
    r"^## \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s+(\S+)\s+\|\s+(.+)$",
    re.MULTILINE,
)


def append_log(wiki_path: Path, entry: LogEntry) -> None:
    """Append a log entry to wiki/log.md with the standard prefix format."""
    logger.info("Appending to log: %s | %s", entry.op, entry.title)
    log_path = wiki_path / "log.md"
    timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M")
    header = f"## [{timestamp_str}] {entry.op} | {entry.title}\n"

    lines = [header]
    if entry.details:
        for line in entry.details.strip().split("\n"):
            lines.append(f"- {line.lstrip('- ').strip()}\n")
    lines.append("\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def tail_log(wiki_path: Path, n: int = 10) -> list[LogEntry]:
    """Parse and return the last N log entries from wiki/log.md."""
    logger.debug("Reading last %d log entries", n)
    log_path = wiki_path / "log.md"
    if not log_path.is_file():
        return []

    content = log_path.read_text(encoding="utf-8")

    # Find all entry start positions
    matches = list(_LOG_PREFIX_RE.finditer(content))
    if not matches:
        return []

    # Take the last N entries
    recent = matches[-n:]
    entries: list[LogEntry] = []

    for i, m in enumerate(recent):
        timestamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
        op = m.group(2)
        title = m.group(3)

        # Extract details: lines between this entry and the next (or EOF)
        start = m.end()
        end = recent[i + 1].start() if i + 1 < len(recent) else len(content)
        detail_block = content[start:end].strip()

        # Clean up detail lines
        detail_lines = []
        for line in detail_block.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                line = line[2:]
            if line:
                detail_lines.append(line)

        entries.append(
            LogEntry(
                timestamp=timestamp,
                op=op,
                title=title,
                details="\n".join(detail_lines),
            )
        )

    return entries
