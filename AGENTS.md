# Wiki Schema

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

- `slug`: matches the filename (without `.md`).
- `type`: one of the page types above.
- `title`: human-readable display title.
- `sources`: list of source filenames that contributed to this page.
- `updated`: ISO 8601 date of last update.
- `tags`: optional labels for categorization.

## Update Rules

1. **New info supersedes old**: when a source provides updated facts, update the page rather than creating a duplicate.
2. **Flag contradictions**: if new info directly contradicts existing content, call `flag_contradiction` and wait for human resolution. Never silently overwrite a conflicting claim.
3. **Always regenerate the index**: after creating or updating any page, call `regenerate_index` so `index.md` stays current (it is a derived view — never hand-edit it).
4. **Always log**: after any operation, call `append_log` with the operation details.
5. **Date all changes**: update the `updated` frontmatter field on every write.

## Index Entry Format

Entries in `index.md` follow this pattern per category:

```
## Entities
- [[Page Name]] - Brief description | Source: source-name | Updated: YYYY-MM-DD

## Concepts
- [[Concept Name]] - Brief description | Sources: N | Updated: YYYY-MM-DD

## Sources
- [Source Title](sources/slug.md) - Ingested: YYYY-MM-DD
```

## Log Entry Format

Entries in `log.md` use a parseable prefix:

```
## [YYYY-MM-DD HH:MM] <operation> | <title>
- Detail lines with bullet points
```

Operations: `ingest`, `query`, `lint`, `create`, `update`, `delete`.

## Hard Rules

1. **NEVER write outside `wiki/`**. All page creation and updates must be within the wiki directory.
2. **NEVER modify `raw/`**. Raw sources are immutable.
3. **NEVER delete a page without human approval** via the `delete_wiki_page` tool (HITL).
4. **NEVER ignore a contradiction** — always call `flag_contradiction`.
5. **NEVER leave orphan pages** — ensure every page is linked from at least one other page or the index.
6. **ALWAYS update index and log** after any page creation, update, or deletion.
7. **NEVER call the same tool twice with the same arguments**. If a tool already returned a result, do not call it again with identical inputs — use the result you already have. Duplicate calls waste tokens and produce duplicate side effects.
8. **NEVER create a page that already exists**. Check the current wiki index (provided in your system prompt) before calling `create_page`. If the page is already listed there, use `update_page` instead.
9. **NEVER read the same page twice in one turn**. If you already ran `wiki_command("read <slug>")` on a slug, use the content you received — do not read it again.

## Data Confidence

- Treat extracted figures and images as low-confidence unless the source confirms.
- Mark uncertain claims with qualifiers in the page content.
- Prefer textual evidence over image-derived claims.
