# Wiki Lint Report — 2026-07-27

## Executive Summary
- Pages audited: 22
- Critical issues: 1
- High issues: 3
- Medium issues: 2
- Low issues: 1

## Critical Issues
### C1. Missing YAML Frontmatter on 17 Pages
- **Affected:** concepts/3d-gaussian-splatting, concepts/agentic-workflow-design, concepts/ai-workflow-automation, concepts/continuous-improvement-system, concepts/llm-fine-tuning-with-qlora, concepts/machine-learning, concepts/real-estate-tokenization, concepts/safe-by-design-ai, concepts/tool-calling, entities/apple-silicon, entities/azure, entities/bhs-corrugated-spain, entities/mlx, entities/modernbert, entities/málaga, entities/polygon-network, entities/university-of-malaga, entities/unknowngravity
- **Finding:** 17 of 22 pages lack YAML frontmatter (--- delimiters with slug, type, title, sources, updated fields). Only 4 pages have proper frontmatter: concepts/ai, entities/alvaro-jimenez-martinez, entities/python, and the lint report. This violates the schema requirement that every page MUST have frontmatter.
- **Action:** Add proper YAML frontmatter to all 17 affected pages with slug, type, title, sources, updated (ISO 8601 date), and tags fields.

## High Issues
### H1. Index Entries Don't Match Page Slugs
- **Affected:** All 21 content pages (entities/*, concepts/*)
- **Finding:** The index.md uses bare slugs (e.g., `python`, `alvaro-jimenez-martinez`) instead of full relative paths (e.g., `entities/python`, `entities/alvaro-jimenez-martinez`). Additionally, the index lists `artificial-intelligence` but the actual page slug is `concepts/ai`. This breaks the index-page correspondence and may cause navigation failures.
- **Action:** Update index.md to use full relative paths (e.g., `entities/python`, `concepts/ai`) and correct the `artificial-intelligence` entry to `concepts/ai`.

### H2. Missing Source Page for `cv`
- **Affected:** sources/cv.md (does not exist)
- **Finding:** The index lists a source `cv` with 11 entity pages referencing it, but no `sources/cv.md` page exists on disk. All entity pages cite "Cv" or "Resume of Álvaro Jiménez Martínez" as their source, but the source page is missing.
- **Action:** Create `sources/cv.md` with proper frontmatter summarizing the resume source, or update affected pages to remove the invalid source reference.

### H3. Stale Content — 19 Pages with `updated=unknown`
- **Affected:** concepts/3d-gaussian-splatting, concepts/agentic-workflow-design, concepts/ai-workflow-automation, concepts/continuous-improvement-system, concepts/llm-fine-tuning-with-qlora, concepts/machine-learning, concepts/real-estate-tokenization, concepts/safe-by-design-ai, concepts/tool-calling, entities/apple-silicon, entities/azure, entities/bhs-corrugated-spain, entities/mlx, entities/modernbert, entities/málaga, entities/polygon-network, entities/university-of-malaga, entities/unknowngravity, lint-report-2026-07-26
- **Finding:** Only 3 pages have valid `updated` dates (concepts/ai: 2026-07-25, entities/alvaro-jimenez-martinez: 2026-07-26, entities/python: 2026-07-26). The remaining 19 pages have `updated=unknown`, making it impossible to determine staleness or prioritize updates.
- **Action:** Set `updated` dates to the page's last known modification date for all 19 affected pages.

## Medium Issues
### M1. Page Type Mismatch
- **Affected:** entities/alvaro-jimenez-martinez
- **Finding:** This page is in the `entities/` directory but has `type: concept` in its frontmatter. It should be `type: entity` to match the schema convention.
- **Action:** Update the frontmatter to set `type: entity`.

### M2. Lint Report Orphan Page
- **Affected:** lint-report-2026-07-26
- **Finding:** The lint report has 0 inbound links and 0 outbound links. While lint reports are expected to be orphans, it has `type: unknown` which is not a valid page type.
- **Action:** Either delete this report (if it's a previous run) or update its type to a valid value if it should be tracked.

## Low Issues
### L1. Inconsistent Related Section Headers
- **Affected:** concepts/ai (uses `## Related`), all other pages (use `## Related Items`)
- **Finding:** The schema specifies `## Related` as the section header, but 11 pages use `## Related Items` instead. This is a minor formatting inconsistency.
- **Action:** Standardize all related sections to use `## Related` as the header.

## Summary Statistics
| Metric | Value |
|--------|-------|
| Total pages | 22 |
| Orphan pages | 1 |
| Broken links | 0 |
| Missing frontmatter | 17 |
| Stale pages (>90 days) | 19 |
| Index mismatches | 21 |
| Missing source pages | 1 |
| Page type mismatches | 1 |