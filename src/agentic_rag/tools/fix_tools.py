"""Tools for the fix agent: file editing and restricted command execution."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from agentic_rag.tools.shared import get_wiki_path

logger = logging.getLogger(__name__)


def _validate_command_within_wiki(command: str) -> str | None:
    """Validate that a command operates only within wiki_path."""
    wiki_path = get_wiki_path().resolve()

    blocked_patterns = [
        "cd /", "cd ~", "cd ..",
        "rm -rf /", "rm -rf ~",
        "sudo", "chmod", "chown",
        "mv /", "cp /",
    ]
    cmd_lower = command.lower().strip()
    for pattern in blocked_patterns:
        if cmd_lower.startswith(pattern) or f" {pattern}" in cmd_lower:
            return f"Blocked: command '{pattern}' not allowed"

    if cmd_lower.startswith("cd "):
        try:
            parts = shlex.split(command)
            target = Path(parts[1]).resolve()
            if not str(target).startswith(str(wiki_path)):
                return f"Blocked: cd to '{target}' is outside wiki_path"
        except Exception:
            pass

    return None


def run_command(command: str) -> str:
    """Execute a shell command within wiki_path."""
    wiki_path = get_wiki_path().resolve()
    error = _validate_command_within_wiki(command)
    if error:
        logger.warning("Command blocked: %s", error)
        return error

    logger.debug("Executing in %s: %s", wiki_path, command)
    try:
        result = subprocess.run(
            command, shell=True, cwd=str(wiki_path),
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code] {result.returncode}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def edit_wiki_page(slug: str, old_text: str, new_text: str) -> str:
    """Replace text in a wiki page. Use for fixing content, frontmatter, links.

    Args:
        slug: Page slug (e.g., 'entities/python', 'concepts/ai')
        old_text: Exact text to find (must match exactly, including whitespace)
        new_text: Replacement text
    """
    wiki_path = get_wiki_path()
    page_path = wiki_path / f"{slug}.md"
    if not page_path.exists():
        return f"Page not found: {slug}"

    content = page_path.read_text(encoding="utf-8")
    if old_text not in content:
        return f"Text not found in {slug}: {old_text!r}"

    count = content.count(old_text)
    new_content = content.replace(old_text, new_text, 1)
    page_path.write_text(new_content, encoding="utf-8")
    logger.info("Edited %s: replaced 1 occurrence of %r", slug, old_text)
    return f"Replaced 1 occurrence in {slug}.md ({count} remaining)"


@tool
def remove_index_entry(slug: str) -> str:
    """Remove a stale entry from wiki/index.md by slug.

    Args:
        slug: Page slug to remove (e.g., 'entities/python')
    """
    wiki_path = get_wiki_path()
    index_path = wiki_path / "index.md"
    if not index_path.exists():
        return "Index not found"

    content = index_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    new_lines = []
    removed = False
    for line in lines:
        if slug in line and line.strip().startswith("- "):
            removed = True
            continue
        new_lines.append(line)

    if not removed:
        return f"No index entry found for {slug}"

    index_path.write_text("\n".join(new_lines), encoding="utf-8")
    logger.info("Removed index entry for %s", slug)
    return f"Removed {slug} from index.md"


@tool
def execute_command(command: str) -> str:
    """Execute a shell command within the wiki directory.

    Args:
        command: Shell command to execute (e.g., 'ls entities/')
    """
    return run_command(command)
