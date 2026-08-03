# Wiki Lint Report — 2026-07-26

## Executive Summary
- Pages audited: 21
- Critical issues: 17
- High issues: 2
- Medium issues: 19
- Low issues: 5

The wiki is topically coherent (all pages revolve around one person's projects and form a fully-connected graph) but fails hard schema compliance: 17 of 21 pages have **no YAML frontmatter at all**, one page is effectively empty, and the person-hub page declares the wrong type. No broken links, no content orphans, no stale pages, and **no contradictions** were found. The 19 "missing Related" flags are actually a heading-name deviation (`## Related Items` vs the schema-required `## Related`) — cross-reference links exist on all of them.

## Critical Issues

### C1. Missing YAML frontmatter on 17 pages
- **Affected:** concepts/3d-gaussian-splatting, concepts/agentic-workflow-design, concepts/ai-workflow-automation, concepts/continuous-improvement-system, concepts/llm-fine-tuning-with-qlora, concepts/real-estate-tokenization, concepts/safe-by-design-ai, concepts/tool-calling, entities/apple-silicon, entities/azure, entities/bhs-corrugated-spain, entities/mlx, entities/modernbert, entities/málaga, entities/polygon-network, entities/university-of-malaga, entities/unknowngravity
- **Finding:** Each page begins directly with `# Title` — no YAML block, so no `slug`, `type`, `title`, `sources`, `updated`, or `tags` are recorded. This breaks the schema's frontmatter requirement and means no `updated` date is tracked for these pages (stale detection is impossible).
- **Action:** Add a YAML frontmatter block to each page: `slug` (filename), `type` (`concept` or `entity` per directory), `title` (display title), `sources` (e.g. `resume-of-alvaro-jimenez-martinez` or `manual`), `updated: 2026-07-26`, and `tags`. Then call `update_index` and `append_log` per Update Rules.

## High Issues

### H1. Effectively empty page: concepts/ai
- **Affected:** concepts/ai
- **Finding:** Body is 28 words ("Machine intelligence.") — below the 50-word threshold. This page is a hub: it is linked from concepts/machine-learning and links to 5 other AI pages, so its thinness is a real data gap rather than a cosmetic one.
- **Action:** Expand with a proper definition, scope, subfields, and brief history; retain the existing `## Related` links. Update `updated` afterward.

### H2. Type/location mismatch on the person-hub page
- **Affected:** entities/alvaro-jimenez-martinez
- **Finding:** File lives in `entities/` but frontmatter declares `type: concept`. Additionally `slug` is `entities/alvaro-jimenez-martinez` (directory prefix included, contrary to "slug matches filename"), `title` is the lowercase slug form instead of a display title ("Álvaro Jiménez Martínez"), and `sources` is empty `[]` even though 100% of content derives from his resume.
- **Action:** Set `type: entity`, `slug: alvaro-jimenez-martinez`, `title: Álvaro Jiménez Martínez`, and populate `sources` with the resume source file. Verify the `index.md` entry reflects type `entity`.

## Medium Issues

### M1. Non-standard Related heading on 19 pages
- **Affected:** concepts/3d-gaussian-splatting, concepts/agentic-workflow-design, concepts/ai-workflow-automation, concepts/continuous-improvement-system, concepts/llm-fine-tuning-with-qlora, concepts/real-estate-tokenization, concepts/safe-by-design-ai, concepts/tool-calling, entities/alvaro-jimenez-martinez, entities/apple-silicon, entities/azure, entities/bhs-corrugated-spain, entities/mlx, entities/modernbert, entities/málaga, entities/polygon-network, entities/python, entities/university-of-malaga, entities/unknowngravity
- **Finding:** The deterministic check flags all 19 as "missing Related section," but semantic review shows every one of them **does** contain a cross-reference list — under the heading `## Related Items` instead of the schema-required `## Related`. Links are present and correct; only the heading name deviates.
- **Action:** Rename `## Related Items` → `## Related` on each of the 19 pages. No links need to be added or removed.

## Low Issues

### L1. Orphaned lint-report artifact
- **Affected:** lint-report-2026-07-26
- **Finding:** A previous lint report file sits in the wiki root with zero inbound/outbound links. It is a report artifact (not content), so it is not a true orphan page, but it pollutes the link graph and page count.
- **Action:** Move lint reports outside `wiki/` (e.g. a `reports/` dir excluded from the graph), or delete the old one when writing a new report.

### L2. Slug convention deviations
- **Affected:** concepts/ai, entities/alvaro-jimenez-martinez
- **Finding:** These two slugs include a directory prefix (`concepts/ai`, `entities/alvaro-jimenez-martinez`) while the schema requires `slug` to equal the filename. The other frontmatter-bearing pages (concepts/machine-learning, entities/python) use the correct bare form.
- **Action:** Normalize to `ai` and `alvaro-jimenez-martinez` (or explicitly document the prefixed convention if intentional).

### L3. Missing source pages and empty sources lists
- **Affected:** all resume-derived pages; entities/alvaro-jimenez-martinez
- **Finding:** 17+ pages cite "Source: Resume of Álvaro Jiménez Martínez," but no `sources/` page exists for that document, and the person page lists `sources: []`. The two manual pages (entities/python, concepts/machine-learning) use `manual` as a placeholder source name.
- **Action:** Ingest the resume as `sources/resume-of-alvaro-jimenez-martinez.md` and reference it in every page's `sources` frontmatter; keep `manual` for the manually authored pages.

### L4. No overview or comparison pages
- **Affected:** wiki-wide
- **Finding:** The wiki contains only concept and entity pages. There is no root overview page, and concepts/ai is too thin to serve as the AI overview hub. No comparisons exist (e.g., MLX vs PyTorch, Azure vs AWS) even though several entity pairs invite them.
- **Action:** Optional: add an overview page and consider 1–2 comparison pages for the strongest entity pairs.

### L5. Near-duplicate coverage to monitor
- **Affected:** concepts/agentic-workflow-design, concepts/ai-workflow-automation
- **Finding:** Both pages describe Álvaro's AI-agent work (local coding agent vs BHS Jira incident workflow). They remain distinct in scope (agent architecture/loop engineering vs operational workflow automation with n8n/Azure), and **no hard duplicate was confirmed**. However, they overlap on tool calling, refinement loops, and local-LLM/Apple Silicon topics, so content could drift into duplication.
- **Action:** No change required now; keep boundaries explicit and cross-link the two pages if either grows.

## Contradictions
**None found.** Cross-checked every shared factual claim across all pages: BHS internship dates (Mar–Jun 2026), UnknownGravity tenure (May–Dec 2025), 40% manual-ticket reduction, 553 auto-resolved tickets, 139 repeat hardware issues, GPA 8.6/10 (top 10%), thesis grade 10/10 with honors, €2.2M listed assets, 50% VRAM reduction, and the mlx-modernbert project stack. All are consistent wherever repeated. No `flag_contradiction` needed.

## Summary Statistics
| Metric | Value |
|--------|-------|
| Total pages | 21 (content) |
| Orphan pages | 0 (1 unlinked report artifact) |
| Broken links | 0 |
| Missing frontmatter | 17 |
| Stale pages (>90 days) | 0 |
| Empty pages (<50 words) | 1 |
| Missing/cross-ref heading deviation | 19 |
| Contradictions | 0 |
