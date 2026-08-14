# Compact Wiki Index Plan — Primary Pages (MOCs) + Reachability Health Check

Status: **PLANNED** (no code changed yet)
Branch: `feat/compact-index` (worktree at `.worktree/`)
Date: 2026-08-06

## Problem

`wiki/index.md` is a derived view rebuilt by `regenerate_index` after every create/update.
`tools/shared.py:get_index_summary` reads the raw file and injects it into the system prompt
of **all four agents** (ingest, query, lint, fix). Today the index contains **55 entries**
(24 concepts + 27 entities + 4 sources), each with a **~160-char content summary + sources
list + updated date** → roughly **14 KB (~4K tokens) per agent run, growing linearly** with
every page and every bit of page content.

Goals:

1. **Shorter index** — no page content; only page slugs + their outbound links.
2. **Only primary pages (MOCs)** in the index — decided by the agent at page creation.
3. **Health check = reachability** — every page must be reachable (any number of steps)
   from the pages listed in the index.

## Current Architecture (map)

| Concern | File | Role |
|---------|------|------|
| Index codec | `src/agentic_rag/io/index.py` | `read_index`/`write_index`/`_format_entry`; parses display-name entries and re-derives slugs via `slugify` |
| Index regeneration | `src/agentic_rag/wiki/dedupe_index.py` | `regenerate_index` — groups ALL pages by type, builds `IndexEntry` with 160-char summaries |
| Wiki model | `src/agentic_rag/wiki/model.py` | `load_wiki` → `Page{slug, fm, sections, outbound_links, word_count}` — **`outbound_links` are already resolved slugs** (exact + slugified + unicode-fuzzy) |
| Health check | `src/agentic_rag/wiki/health.py` | 7 kinds: `orphan` (no inbound), `missing-index`, `broken-link`, `missing-frontmatter`, `missing-related`, `empty`, `stale` |
| Schemas | `src/agentic_rag/schemas/wiki.py` | `Frontmatter{slug,type,title,sources,updated,tags}`; `IndexEntry{slug,summary,type,sources,updated,display_name}`; `Index{categories}` |
| Creation/fix tools | `tools/ingest_tools.py`, `tools/fix_tools.py` | `create_page`/`update_page`/`add_frontmatter` — no primary concept today |
| Prompt injection | `tools/shared.py:get_index_summary` | Raw `index.md` → every agent system prompt (no change needed; just reads the smaller file) |
| Other consumers | `cli.py:status` (counts entries), `docs/wiki-engine.html`, tests | See Ripple effects |

## Proposed Design

### A. `primary` frontmatter flag (source of truth)

- `Frontmatter.primary: bool = False` in `schemas/wiki.py`.
- `create_page(..., primary: bool = False)` / `update_page(..., primary: bool | None = None)`
  gain the parameter — **the agent decides at creation** whether the page is a MOC/hub
  (entry point that links out to many related pages) vs a leaf/detail page.
- Old pages without the field parse as `primary=False` (pydantic default) — they drop out
  of the index until promoted. No crashes.

### B. Slim index format — slugs + outbound links only

```markdown
# Wiki Index

## Concepts
- [[concepts/wiki-based-rag]] → [[concepts/bm25]], [[concepts/wiki-health-check]], [[entities/agentic-rag]]
- [[concepts/agentic-workflow-design]] → [[concepts/tool-calling]], [[entities/langchain]]

## Entities
- [[entities/agentic-rag]] → [[entities/langchain]], [[entities/streamlit]], [[concepts/wiki-based-rag]]
```

- Full-path slugs (`[[concepts/x]]`) — resolves in Obsidian; exact slug matching in
  `read_index`/health check (kills the CSAR/parenthetical slugify fallback complexity).
- Outbound links come from already-resolved `page.outbound_links`.
- **Dropped**: summaries, `Sources:` fields, `Updated:` dates. Size becomes ~10 entries
  (~1 KB) and stops growing with page content.

### C. Reachability health check

- **New `unreachable` kind replaces `orphan`** (`wiki/health.py`): BFS from all index
  (primary) pages over resolved outbound links; any content page not visited is flagged
  "not reachable from the index in any number of steps". Strictly stronger than orphan
  (a page can have inbound links yet still be unreachable if its only inbounds come from
  other unreachable pages).
- `missing-index` redefined: only **primary** pages must be listed (guards a stale index).
- Lint agent prompt + `_render_report_markdown` + fix-agent kind→tool map updated.

## Decisions (confirmed with user)

1. **Migration**: backfill `primary: true` on existing hubs (inbound-degree ≥ 3 or
   type `overview`), regenerate, then promote stragglers flagged `unreachable` via
   `set_primary` — converges in one pass.
2. **Source pages**: eligible for `primary` like any page (agent decides; usually false).
3. **Fix tool**: new `set_primary(slug, primary)` tool so the fix agent can promote/demote
   pages (otherwise `unreachable` findings have no remedy under the pinned tool map).

## Implementation Plan (file by file)

| # | File | Change | Verify |
|---|------|--------|--------|
| 1 | `src/agentic_rag/schemas/wiki.py` | `Frontmatter.primary: bool = False`; slim `IndexEntry{slug, type, outbound: list[str]}` | `uv run pytest tests/unit/test_markdown_parser.py` |
| 2 | `src/agentic_rag/io/index.py` | New entry regex/format (slug → outbound links); rewrite `read_index`/`write_index`/`_format_entry`; drop source-entry special case | `uv run pytest tests/unit/test_index.py` (rewritten for new format) |
| 3 | `src/agentic_rag/wiki/dedupe_index.py` | `_build_categories` filters `fm.primary`; outbound = `page.outbound_links`; drop `_page_summary`/`_summarize` | idempotency: two `regenerate_index` runs → byte-identical `index.md` |
| 4 | `src/agentic_rag/tools/ingest_tools.py` | `create_page(..., primary=False)`, `update_page(..., primary=None)` → frontmatter | additions to `tests/unit/test_ingest_tools.py` |
| 5 | `src/agentic_rag/tools/fix_tools.py` | `add_frontmatter(..., primary=False)`; new `set_primary(slug, primary)` tool | additions to `tests/unit/test_fix_tools.py` |
| 6 | `src/agentic_rag/wiki/health.py` | BFS reachability from index pages; `unreachable` replaces `orphan`; `missing-index` only for primary; exact slug matching | `tests/unit/test_health.py` rewritten (hub page primary; one leaf linked; one truly unreachable) |
| 7 | `AGENTS.md`, `src/agentic_rag/schemas/agents_md.py`, `src/agentic_rag/agents/prompts.py` | `primary` in frontmatter spec; new index-entry format; "decide primary at creation" rule; `unreachable` in lint workflow + fix tool map | `tests/unit/test_agents_md_loader.py` |
| 8 | `src/agentic_rag/tools/nav.py` | `wiki_scan` shows `(primary)` marker; `run_health_check` docstring | `tests/unit/test_nav_scan.py` |
| 9 | `tests/fixtures/eval_wiki/`, `tests/fixtures/eval_broken_wiki/` | Add `primary: true` to hub pages so zero-issue conformance holds | `uv run pytest tests/levels/test_corpus_selfcheck.py` (21 pages, zero issues; broken fixture = exactly 3 defects) |
| 10 | `tests/levels/level2/test_state_consistency.py`, `test_tool_selection.py`, `tests/unit/test_cli.py` | Scripted `create_page` calls primary-aware; slug-in-index asserts keep passing | `uv run pytest tests/ -q` |
| 11 | `docs/wiki-engine.html`, `README.md` | Update documented index format + entry example | manual review |
| 12 | **Migration** (one-time, in worktree) | Backfill `primary: true` on hubs, `regenerate_index`, `health_check` → promote stragglers | `health_check` = 0 issues; index ≤ ~10 entries |

**Full gate:** `uv run pytest tests/ -q` green + real `wiki/index.md` regenerated with only
primary pages.

## Ripple Effects / Notes

- `read_index` consumers: `wiki/health.py` + `cli.py:status` (counts entries only) — both
  covered by items 2/6. `get_index_summary` needs no code change.
- **Worktree caveat**: the live `wiki/` is gitignored (`/wiki*/`), so a fresh checkout does
  not contain it. `wiki/` has been copied into `.worktree/wiki/` (284K) so migration +
  acceptance tests can run there. It stays untracked in the branch.
- BFS seed = index pages, so primary pages are always reachable by definition (no spurious
  self-flagging).
- Per-entry `Updated:` dates leave the index; recency stays available via `wiki_scan`.
- `archive/` plan doc is untracked in main; copy into the worktree if the branch should
  carry it.

## Test Suite Invariants That Must Keep Passing

- `tests/levels/test_corpus_selfcheck.py`: `eval_wiki/` = 21 pages, zero issues;
  `eval_broken_wiki/` = exactly `{(missing-frontmatter, entities/broken-fm),
  (broken-link, entities/linker), (missing-related, entities/lonely)}`.
- `tests/levels/level2/test_tool_selection.py` + `test_turn_efficiency.py`:
  ingest terminal pair `regenerate_index → append_log`; index text contains the created
  page slug (scripted `create_page` must pass `primary=True` or assert adjusts).
- `tests/unit/test_health.py` `test_zero_llm_calls`: health module stays import-free of
  `langchain`.
