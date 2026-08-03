# Progress

<!-- One line per task. Status markers: PENDING | RUNNING | COMPLETED | BLOCKED -->

- [x] [COMPLETED] T1: Unblock tests: restore `find_inbound_links`+`extract_concepts` in `tools/lint_tools.py` as self-contained implementations; sync `recursion_limit` to 30 everywhere; register `path_guard_middleware` in `agents/factory.py` `tools/lint_tools.py` `config.py` `agents/factory.py`
- [x] [COMPLETED] T2: Build source-of-truth `Wiki` model (`load_wiki`, `Page`, `Section`) from `list_pages`+frontmatter+headings/links `src/agentic_rag/wiki/model.py`
- [x] [COMPLETED] T3: BM25 `wiki.search` over curated page fields + bounded depth-1 link expansion; add `rank_bm25` dep `src/agentic_rag/wiki/search.py` `pyproject.toml`
- [x] [COMPLETED] T4: `regenerate_index` from `Wiki` (no raw-H1 summaries); regenerate live `wiki/index.md` `src/agentic_rag/wiki/dedupe_index.py`
- [ ] [PENDING] T5: Nav tools `wiki_search`/`wiki_read_page`/`wiki_summary`/`wiki_link_graph` over the model+search `src/agentic_rag/tools/nav.py`
- [ ] [PENDING] T6: `QueryAnswer`/`SourceCitation` schema + `NavCapture` + `submit_query_answer` cite-or-die tool + pure `validate_citations` `src/agentic_rag/schemas/query.py` `src/agentic_rag/tools/grounding.py`
- [ ] [PENDING] T7: Rebuild query agent on nav tools + `submit_query_answer`; update `build_query_prompt`; render `QueryAnswer` in `query` CLI with back-compat fallback `src/agentic_rag/agents/query.py` `src/agentic_rag/agents/prompts.py` `src/agentic_rag/cli.py`
- [ ] [PENDING] T8: Deterministic `health_check`+`LintReport`+`schemas/lint.py`; restructure lint agent (`run_health_check` tool, `write_lint_report` accepts model|str); lint aliases delegate to model `src/agentic_rag/lint/health.py` `src/agentic_rag/schemas/lint.py` `src/agentic_rag/agents/lint.py` `src/agentic_rag/tools/lint_tools.py` `src/agentic_rag/agents/prompts.py`

- T3 exports: `search(wiki: Wiki, query: str, *, k=8, types=None, tags=None, expand_links=True, depth=1) -> list[SearchHit]` in `src/agentic_rag/wiki/search.py`; `SearchHit(slug, score, sections, matched_via)` (pydantic).
- T3 note: BM25 doc = title + tags + headings + section texts (intro text = inferred summary); tokenization NFKD→ASCII→lower→split non-alphanumeric; `types`/`tags` filter candidates BEFORE BM25; direct hits skip zero-token-overlap pages; expansion = min(2,k) per source page, total cap k, EXPAND_SCORE=0.1 fixed, `matched_via='expand-link'`; empty inputs → [].
- T3 note: dep added as `rank-bm25>=0.0.5` (uv-normalized name) in pyproject.toml + uv.lock (installed 0.2.2).
- T3 note: `matched_via` values: "title" / "tags" / "section:<heading>" / "body" (first wins); `sections` = headings of matched body sections.
- T4 exports: `regenerate_index(wiki_path: Path) -> Path` in `src/agentic_rag/wiki/dedupe_index.py` — atomic rewrite of `wiki/index.md` from `load_wiki`, returns the index path. Excludes `lint-report-*` by filename; groups by `fm.type` via `_category_for_type`; entry format reuses `write_index`/`_format_entry`.
- T4 note: summaries = first-section text, `[[link]]`/markdown-stripped, single-line, ≤160 chars word-boundary truncation with "…", fallback `fm.title` — never a raw H1. Deterministic (byte-identical re-runs). Live wiki/index.md regenerated: 21 entries, 0 raw-H1 summaries (was ~9).
- T4 note: data quirk — `entities/alvaro-jimenez-martinez.md` declares `type: concept` in frontmatter, so it groups under Concepts (11 concepts + 10 entities). Grouping follows fm.type per spec.
- T1 exports: `find_inbound_links(slug: str) -> str` in `src/agentic_rag/tools/lint_tools.py` — restored alias; self-contained via `list_pages()` + `[[link]]` regex; returns "Found N page(s)" / "...orphan". Used by T8 (keep as alias).
- T1 exports: `extract_concepts(content: str) -> str` in `src/agentic_rag/tools/lint_tools.py` — restored alias; delegates to `extract_headings`/`extract_links`; lists headings + `[[target]]` links. Used by T8 (keep as alias).
- T1 exports: `path_guard_middleware` now registered in `agents/factory.py::build_agent` (middleware order: audit_logging, path_guard, token_capture). Blocks write-tools with `raw/`, absolute, or `..` paths. T5/T6 nav-tool args (`slug`, `_source_path`) pass as reads; do not add nav tools to `write_tools`.
- T1 note: `tools/shared.py::read_wiki_page` propagates `FileNotFoundError` (try/except removed — pre-existing regression fix, approved by orchestrator). Tests assert propagation.
- T1 note: `recursion_limit` was already 30 in `config.py` (single runtime source via `settings.recursion_limit`) — no change needed.
- T2 exports: `load_wiki(wiki_path: Path) -> Wiki` in `src/agentic_rag/wiki/model.py` — source-of-truth model; 22 pages on live wiki (21 content + lint-report; index/log excluded by list_pages).
- T2 exports: `Page(slug:str, rel_path:Path, fm:Frontmatter, sections:list[Section], outbound_links:list[str], word_count:int)`, `Section(heading:str, level:int, text:str)`, `Wiki(pages:list[Page], by_slug:dict[str,Page])` in same module.
- T2 note: `outbound_links` are RESOLVED slugs (3-step resolver: exact → slugify short-name → unicode-preserving short-name; drops unresolved + self-links). `[[Málaga]]`→`entities/málaga`, `[[MLX]]`→`entities/mlx`.
- T2 note: 18/22 live pages have NO frontmatter — synthesized fm: type from dir, title from first H1, updated=mtime date, sources/tags=[]. load_wiki never raises on frontmatter-less/malformed pages.
- T2 note: sections include H1 + all headings; preamble text prepended to first section (or synthetic `heading=""` section). Frontmatter excluded from sections/links/word_count. Empty wiki dir → `Wiki([], {})`.
- T2 note: model INCLUDES `lint-report-*.md` as an unknown-type page — T8 health_check must skip lint-report pages for content stats; T4 regenerate_index excludes them.
- T2 infra: `.gitignore` line 2 changed `wiki/` → `/wiki/` (root-anchored) — the old unanchored pattern was swallowing `src/agentic_rag/wiki/`; data dir still ignored.