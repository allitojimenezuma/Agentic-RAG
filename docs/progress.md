# Progress

<!-- One line per task. Status markers: PENDING | RUNNING | COMPLETED | BLOCKED -->

- [x] [COMPLETED] T1: Unblock tests: restore `find_inbound_links`+`extract_concepts` in `tools/lint_tools.py` as self-contained implementations; sync `recursion_limit` to 30 everywhere; register `path_guard_middleware` in `agents/factory.py` `tools/lint_tools.py` `config.py` `agents/factory.py`
- [ ] [PENDING] T2: Build source-of-truth `Wiki` model (`load_wiki`, `Page`, `Section`) from `list_pages`+frontmatter+headings/links `src/agentic_rag/wiki/model.py`
- [ ] [PENDING] T3: BM25 `wiki.search` over curated page fields + bounded depth-1 link expansion; add `rank_bm25` dep `src/agentic_rag/wiki/search.py` `pyproject.toml`
- [ ] [PENDING] T4: `regenerate_index` from `Wiki` (no raw-H1 summaries); regenerate live `wiki/index.md` `src/agentic_rag/wiki/dedupe_index.py`
- [ ] [PENDING] T5: Nav tools `wiki_search`/`wiki_read_page`/`wiki_summary`/`wiki_link_graph` over the model+search `src/agentic_rag/tools/nav.py`
- [ ] [PENDING] T6: `QueryAnswer`/`SourceCitation` schema + `NavCapture` + `submit_query_answer` cite-or-die tool + pure `validate_citations` `src/agentic_rag/schemas/query.py` `src/agentic_rag/tools/grounding.py`
- [ ] [PENDING] T7: Rebuild query agent on nav tools + `submit_query_answer`; update `build_query_prompt`; render `QueryAnswer` in `query` CLI with back-compat fallback `src/agentic_rag/agents/query.py` `src/agentic_rag/agents/prompts.py` `src/agentic_rag/cli.py`
- [ ] [PENDING] T8: Deterministic `health_check`+`LintReport`+`schemas/lint.py`; restructure lint agent (`run_health_check` tool, `write_lint_report` accepts model|str); lint aliases delegate to model `src/agentic_rag/lint/health.py` `src/agentic_rag/schemas/lint.py` `src/agentic_rag/agents/lint.py` `src/agentic_rag/tools/lint_tools.py` `src/agentic_rag/agents/prompts.py`

## Interface handoffs
<!-- Populated by the orchestrator from each task's handoff contract block. Do not pre-fill beyond the stub. -->
- T1 exports: `find_inbound_links(slug: str) -> str` in `src/agentic_rag/tools/lint_tools.py` — restored alias; self-contained via `list_pages()` + `[[link]]` regex; returns "Found N page(s)" / "...orphan". Used by T8 (keep as alias).
- T1 exports: `extract_concepts(content: str) -> str` in `src/agentic_rag/tools/lint_tools.py` — restored alias; delegates to `extract_headings`/`extract_links`; lists headings + `[[target]]` links. Used by T8 (keep as alias).
- T1 exports: `path_guard_middleware` now registered in `agents/factory.py::build_agent` (middleware order: audit_logging, path_guard, token_capture). Blocks write-tools with `raw/`, absolute, or `..` paths. T5/T6 nav-tool args (`slug`, `_source_path`) pass as reads; do not add nav tools to `write_tools`.
- T1 note: `tools/shared.py::read_wiki_page` propagates `FileNotFoundError` (try/except removed — pre-existing regression fix, approved by orchestrator). Tests assert propagation.
- T1 note: `recursion_limit` was already 30 in `config.py` (single runtime source via `settings.recursion_limit`) — no change needed.