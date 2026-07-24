"""AGENTS.md parser/loader — reads the wiki schema for agent system prompts."""

from __future__ import annotations

from pathlib import Path

_DEFAULT_AGENTS_MD = """# Wiki Schema

This file defines conventions for the LLM-maintained wiki. Every agent loads this into its system prompt.

## Page Types

| Type | Directory | Description |
|------|-----------|-------------|
| `entity` | `entities/` | Named real-world things: people, organizations, software, hardware, companies |
| `concept` | `concepts/` | Abstract ideas, techniques, patterns, workflows |
| `source` | `sources/` | Summary of an ingested raw source (PDF, docx, etc.) |
| `comparison` | `comparisons/` | Side-by-side analysis of two entities or concepts |
| `overview` | (root) | High-level summary pages |

## Naming Convention

- Slugs: lowercase, hyphens for spaces, no special characters.
- Files: `entities/<slug>.md`, `concepts/<slug>.md`, `sources/<slug>.md`, `comparisons/<a>-vs-<b>.md`.
- Example: "3D Gaussian Splatting" → `concepts/3d-gaussian-splatting.md`.

## Cross-Reference Format

- Use Obsidian-style `[[Page Name]]` for all internal links.
- Every page MUST have a `## Related` section at the bottom with cross-links to related pages.
- Use `[[Page Name|alias]]` when you need display text different from the page name.

## Frontmatter

Every page MUST have YAML frontmatter:

```yaml
---
slug: page-slug
type: entity | concept | source | comparison | overview
title: Display Title
sources:
  - source-name.pdf
updated: 2025-01-01
tags:
  - tag1
  - tag2
---
```

## Update Rules

1. **New info supersedes old**: when a source provides updated facts, update the page rather than creating a duplicate.
2. **Flag contradictions**: if new info directly contradicts existing content, call `flag_contradiction` and wait for human resolution.
3. **Always update index**: after creating or updating any page, call `update_index` to keep the index current.
4. **Always log**: after any operation, call `append_log` with the operation details.
5. **Date all changes**: update the `updated` frontmatter field on every write.

## Hard Rules

1. **NEVER write outside `wiki/`**.
2. **NEVER modify `raw/`**.
3. **NEVER delete a page without human approval** via the `delete_wiki_page` tool (HITL).
4. **NEVER ignore a contradiction** — always call `flag_contradiction`.
5. **NEVER leave orphan pages**.
6. **ALWAYS update index and log** after any page creation, update, or deletion.
"""


def load_agents_md(path: Path) -> str:
    """Read AGENTS.md and return its content.

    If the file is missing, return a sensible default matching the schema spec.
    """
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return _DEFAULT_AGENTS_MD
