"""Pure source-section chunker: split markdown into chunks at level-2+ headings.

Used by the ingest agent to extract per-section on large sources: the agent reads
a source once via ``read_source``, then processes one chunk at a time. No LLM, no
I/O — this is a pure function.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# A heading line that starts a new chunk: level 2 or deeper (``##``, ``###``, ...).
_CHUNK_HEADING_RE = re.compile(r"^#{2,} ")
# A heading line that may serve as a breadcrumb: level 1 or 2 (``#``, ``##``).
_BREADCRUMB_RE = re.compile(r"^#{1,2} ")


def chunk_by_heading(markdown: str, max_chars: int = 4000) -> list[str]:
    """Split ``markdown`` into chunks at ``## `` headings (level-2 and deeper).

    Each chunk starts at a ``## ``/``### ``/... heading and runs until the next
    such heading. The most recent ``# ``/``## `` heading line preceding the chunk
    is prepended as a breadcrumb, so every chunk carries its document context.
    A chunk larger than ``max_chars`` is returned whole — headings are the unit,
    chunks are never split further. Empty/whitespace-only input returns ``[]``.
    Pure function: no I/O, no state, never raises.
    """
    if not markdown or not markdown.strip():
        return []
    lines = markdown.splitlines()
    starts = [i for i, line in enumerate(lines) if _CHUNK_HEADING_RE.match(line)]
    if not starts:
        return ["\n".join(lines)]

    chunks: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start:end])
        crumb = _most_recent_breadcrumb(lines[:start])
        chunks.append(f"{crumb}\n{body}" if crumb else body)
    return chunks


def _most_recent_breadcrumb(lines: list[str]) -> str:
    """Return the most recent ``# ``/``## `` heading line in ``lines``, or ``""``."""
    crumb = ""
    for line in lines:
        if _BREADCRUMB_RE.match(line):
            crumb = line
    return crumb
