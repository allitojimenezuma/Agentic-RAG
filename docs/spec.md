# Spec — Agentic wiki navigation, grounded query, and deterministic lint (Pass A)

## Intent

This change upgrades the agentic-wiki system (per `IDEA.md` — an LLM-maintained, interlinked
markdown wiki that *compounds* across sources, navigating the curated graph rather than
re-deriving answers from raw chunks) so the agents are fast, concise, and produce
senior-quality, grounded output. Concretely, Pass A:

1. Makes the **filesystem + frontmatter the source of truth**, with `index.md` a *regenerated
   derived view* (eliminating the hand-maintained duplicate that has drifted — ~9 of 20 entries
   have summaries polluted with page H1 headings).
2. Replaces the three overlapping, weak retrieval tools (`read_index` full dump + substring
   `search_index` + unbounded link-BFS `find_relevant_pages`) with **one consolidated,
   deterministic navigation tool** `wiki_search`, giving the agent ranked pages + scoped link
   expansion in a single call.
3. Gives the **query agent structured output** (`QueryAnswer` with `answer`, `citations`,
   `confidence`, `suggestion`) and **pipeline-bound citations** — every cited slug is validated
   against the set of pages actually navigated that turn (cite-or-die, not prompt pleading).
4. Splits **lint into a deterministic structural checker** (free, instant) plus an **optional
   LLM prose pass** for the semantic judgment that genuinely needs a model.
5. Unblocks iteration: the test suite currently fails at *collection* (imports of removed
   functions) — Pass A makes it green again.

This is **not** classic chunk-RAG over raw sources. The wiki is already the curated retrieval
unit; navigation is the graph (`index → page → cross-links`). No vector store is used this pass
(see Open questions). Pass A does **not** change ingest or fix agents (deferred to Pass B).

## Scope

### In scope
- `docs/research-agentic-rag-best-practices.md` is the cited reasoning brief (already written);
  this spec operationalizes findings #6, #11, #12, #13, #14, #15, #21, #22 from it.
- **Source of truth model**: a new `Wiki` in-memory model built from `list_pages()` + parsed
  frontmatter; `index.md` regenerated from it by a function, not maintained by agents.
- **`wiki_search` tool** (BM25 over curated page fields + depth-1 bounded link expansion),
  replacing `read_index` / `search_index` / `find_relevant_pages` for *navigation*.
  Adds `rank_bm25` to dependencies.
- **Query agent**: uses `wiki_search`, a `submit_query_answer` cite-or-die tool, citation-binding
  validator; CLI renders `QueryAnswer` (answer + citations + confidence + suggestion).
- **Lint**: deterministic `health_check(wiki_path)` → `LintReport` (orphan, broken-link,
  missing-index, missing-frontmatter, missing-related, empty, stale); an LLM agent that takes
  the structured report and emits the prose report (optional semantic judgment).
- **Test suite green**: restore `find_inbound_links` / `extract_concepts` as thin aliases or
  update the two importing tests so `pytest` collects and passes.
- Register (or intentionally delete) the dead `path_guard_middleware`; reconcile
  `recursion_limit` to a single value.

### Out of scope (deferred)
- **Pass B**: ingest-agent compilation (structured extraction, deterministic
  create-update-conflict matching); fix-agent refactor (remove shell, add page-edit tools).
- Vector / hybrid embeddings store (Chroma/FAISS) or `qmd` integration. Triggered only when the
  wiki corpus crosses a documented threshold (default ≥300 pages) or measured recall drops.
- Changing the model / proxy (single local model at `127.0.0.1:8484`).
- Image handling, web URL ingestion, Marp/canvas exports, MCP server, durable checkpointer.
- Filing query answers back as wiki pages.
- `\[\[link]]` display-name→slug resolution behavior changes beyond fixing the Unicode path bug
  exposed by `málaga.md` (covered as a side-effect of the source-of-truth model).

## Conventions
- Language: Python 3.11+. Package `agentic-rag` (import root `agentic_rag`), `src/` layout.
- Env / deps via `uv`. New dependency `rank_bm25>=0.0.5` added to `pyproject.toml` `[project.dependencies]`.
- Test command (executors and gate run this): `uv run pytest` (config: `testpaths=["tests"]`, `asyncio_mode="auto"`).
- Import style: `from __future__ import annotations`; PEP 8; module-level `logging.getLogger(__name__)`; no third-party globals beyond the existing `_WIKI_PATH` pattern in `tools/shared.py`.
- Tools are LangChain `@tool`-decorated functions returning `str` (existing pattern); navigation/search return pre-formatted token-efficient strings (Anthropic Writing Effective Tools #22).
- Structured query output is via a `submit_query_answer` `@tool` (NOT `create_agent(response_format=...)`) so it is testable with the existing `ScriptedChatModel` harness and grounding is enforced at the boundary. `create_agent` does accept `response_format` (langchain 1.3.14) but that path is brittle to the fake-LLM harness; the tool approach is the chosen technique.
- Agents built via `agents/factory.py::build_agent` (wraps `create_agent` + token-capture middleware); do not bypass it.
- Wiki page unit = curated page (`entities/<slug>.md`, etc.). Slugs resolve recursively (`io/wiki_io.py::_resolve_page_path`). Frontmatter parsed by `io/markdown_parser.py::parse_frontmatter` / serialized by `serialize_frontmatter`. Existing `slugify` (NFKD→ASCII) is the canonical name→slug transform; **preserve it**.
- Atomic writes via `tempfile` + `replace` (existing pattern in `io/wiki_io.py`, `io/index_manager.py`).
- `AGENTS.md` schema (page types, frontmatter, `[[link]]` format, update rules) is injected into agent prompts — unchanged.
- Keep existing middleware pair `audit_logging_middleware` + `token_capture_middleware` in `build_agent`.

## Interfaces

### New / changed modules

**`src/agentic_rag/wiki/model.py`** (NEW) — the source-of-truth in-memory model.
```python
def load_wiki(wiki_path: Path) -> Wiki
```
- Reads every `.md` under `wiki_path` via `io/wiki_io.list_pages` (excludes `index.md`, `log.md`).
- For each page: `slug = relative_path.removesuffix(".md")`; parse frontmatter via `parse_frontmatter`; extract `## section` headings and `[[link]]` outbound links via existing `extract_links` / `extract_headings`.
- `Page(slug:str, rel_path:Path, fm:Frontmatter, sections:list[Section], outbound_links:list[str], word_count:int)`.
- `Section(heading:str, level:int, text:str)` where `text` is the body under that heading (frontmatter excluded).
- `Wiki(pages:list[Page], by_slug:dict[str,Page])`.
- Missing/partial frontmatter: `fm` is synthesized with inferred `type` from dir (`{"entities":"entity","concepts":"concept","sources":"source","comparisons":"comparison"}`) and `title` from first H1, `updated`=file mtime date, `sources`/`tags`=`[]` — same fallback logic already in `lint_tools.wiki_link_summary`/`read_all_pages`. **Deduplicate that logic into the model.** Must not raise on a page with no frontmatter.

**`src/agentic_rag/wiki/search.py`** (NEW) — deterministic navigation over a `Wiki`.
```python
def search(wiki: Wiki, query: str, *, k: int = 8, types: list[str] | None = None,
           tags: list[str] | None = None, expand_links: bool = True, depth: int = 1) -> list[SearchHit]
```
- BM25 over the per-page "document" = `title + summary(frontmatter fields if present else inferred) + tags + all section headings + section text bodies` (NFKD-ASCII lowercased tokenization; reuse `slugify`-style normalization for tokens, split on non-alphanumeric). No embeddings this pass.
- `types`/`tags`: Python predicate filters (Self-Querying finding #10) — when present, filter the candidate set *before* BM25.
- Returns `SearchHit(slug:str, score:float, sections:list[str], matched_via:str)` sorted desc, top-`k`.
- **Bounded link expansion**: if `expand_links` and `depth>=1`, for each direct hit, add linked pages (from `page.outbound_links`) not already in the set as `expand-link` hits with a small fixed score; cap expansion at `min(2, k)` extra pages per direct hit and a total cap of `k` expansion hits. **Never unbounded** (fixes current `find_relevant_pages` whole-graph pull).
- Token-efficient tool output string format defined in `tools/` (below).

**`src/agentic_rag/wiki/dedupe_index.py`** (NEW) — regenerate `index.md` from `Wiki`.
```python
def regenerate_index(wiki_path: Path) -> Path   # writes wiki/index.md atomically, returns its path
```
- Groups pages by `fm.type`; writes the exact `AGENTS.md` index-entry format produced by `io/index_manager._format_entry` (reuse it). Summaries: prefer `fm`-derived one-line summary (first section or title) — **never** stuff a raw H1 (this is the current corruption). If no usable summary, use title.
- Excludes `index.md`, `log.md`, `lint-report-*.md`.
- Atomic write (temp + replace).

**`src/agentic_rag/tools/nav.py`** (NEW) — the one consolidated navigation tool:
```python
@tool def wiki_search(query:str, k: int=8, types: str|None=None, tags: str|None=None) -> str
@tool def wiki_read_page(slug: str, section: str|None=None) -> str
@tool def wiki_summary() -> str          # compact 1-line-per-page catalog (replaces read_index dump)
@tool def wiki_link_graph() -> str        # the deterministic inbound/outbound summary (replaces wiki_link_summary)
```
- `wiki_search` calls `wiki.search(...)` and returns a token-efficient string:
  `Found N relevant: <slug> (score=X, sections: s1; s2) [+ linked: <slug>]`. `types`/`tags` accept comma-separated.
- `wiki_read_page`: thin wrapper over shared `read_page`; if `section` given return only that section's text (uses `Wiki` sections if cached, else re-read + split). Backward-compatible string output.
- `wiki_link_graph`: the existing `lint_tools.wiki_link_summary` output, moved here (it stays deterministic). Lint reuses it.
- `init_shared_tools(wiki_path)` is extended (or a parallel `init_nav(wiki_path)`) to set the path for the nav module. Prefer reusing the existing `_WIKI_PATH` global to avoid a second global.
- The **old** `tools/shared.read_index`, `search_index`, `read_wiki_page` and `tools/query_tools.find_relevant_pages` are **kept importable** for tests/back-compat but the query agent switches to the nav tools.

**`src/agentic_rag/schemas/query.py`** (NEW) — structured query output.
```python
class SourceCitation(BaseModel): slug:str; title:str; section:str|None=None
class QueryAnswer(BaseModel):
    answer: str                      # markdown with inline [[Page]] citations
    citations: list[SourceCitation]  # every cited slug MUST be in the turn's navigated set
    confidence: Literal["high","medium","low"]
    suggestion: str                 # empty string if coverage is good
```
- Exported from `schemas/__init__.py`.

**`src/agentic_rag/tools/grounding.py`** (NEW) — cite-or-die finalization tool + per-invocation capture store.
```python
class NavCapture:                          # per-invocation mutable store
    navigated: set[str]                    # slugs returned by wiki_search / wiki_read_page
def new_nav_capture() -> NavCapture        # built once per agent invocation
def validate_citations(answer: QueryAnswer, navigated_slugs: set[str]) -> QueryAnswer  # pure, tested
@tool
def submit_query_answer(answer: str, citations: list[SourceCitation],
                        confidence: Literal["high","medium","low"]) -> QueryAnswer:
    ...                                    # builds QueryAnswer, validates, returns it
```
- **Finalization is a TOOL, not `create_agent(response_format=...)`.** Rationale: it is testable with the existing `ScriptedChatModel` harness (scripted `AIMessage`s call `submit_query_answer` directly), and citation grounding is enforced *at the boundary* rather than depending on agent-level structured-output semantics that fight a fake model. (`create_agent` accepts `response_format` in installed langchain 1.3.14, but that path is brittle to the fake harness.)
- The nav tools (`wiki_search`, `wiki_read_page`) record each returned slug into a shared `NavCapture` (set via an `init_nav_capture()` mirror of the existing `_WIKI_PATH` global pattern, or closure). The query agent REQUIRES ending with exactly one `submit_query_answer` call (system prompt enforces; the validator makes it safe).
- `submit_query_answer` (an arg-validated `@tool`) builds a `QueryAnswer` and runs `validate_citations` against the capture's `navigated` set:
  - drops any `SourceCitation` whose `slug` is not in `navigated_slugs` (cite-or-die #15);
  - **does not crash** — returns the cleaned `QueryAnswer`; logs dropped slugs at WARNING;
  - returns the `QueryAnswer` as the tool result string (CLI reconstructs the model from the tool-call args, below).
- `validate_citations` is also exposed as a pure function for direct unit tests (no agent needed). The agent passes `suggestion` as a `submit_query_answer` argument (already a `QueryAnswer` field); LOW-confidence queries must populate it.

**`src/agentic_rag/agents/query.py`** — rebuilt to:
- tools = `[wiki_search, wiki_read_page, wiki_summary, submit_query_answer]` (NOT `read_index`, `search_index`, `find_relevant_pages`, `find_inbound_links`).
- `build_query_agent` creates a `NavCapture` per invocation, registers it with the nav tool module (existing `_WIKI_PATH`-style global: `_NAV_CAPTURE`), and stashes it on `agent._nav_capture` (mirrors the existing `agent._token_tracker` stashing).
- **No `response_format`** — finalization is the `submit_query_answer` tool; the system prompt is the only enforcement that the agent must call it (validator makes it safe even if it slips up).
- System prompt (`agents/prompts.py::build_query_prompt`) updated: "You have `wiki_search` (one call retrieves ranked relevant pages), then `wiki_read_page` for the few you cite. You MUST end by calling `submit_query_answer`. Every `citations[].slug` and every `[[X]]` in `answer` must be a page you obtained from `wiki_search`/`wiki_read_page` this turn — unknown citations are dropped. If the wiki doesn't cover it, set `confidence='low'` and `suggestion` accordingly."

**`src/agentic_rag/cli.py`** — `query` command:
- After `agent.invoke`, locate the `submit_query_answer` tool-call args in the message trace (`result["messages"]` ToolMessages → reconstruct `QueryAnswer`), run `validate_citations(answer, agent._nav_capture.navigated)`, and render:
  `Answer:\n{answer}\n\nConfidence: {conf}\nCitations: ...\nSuggestion: ...`.
- **Compat**: the CLI integration tests (`tests/integration/test_cli.py`) build a fake agent via `create_agent(model=ScriptedChatModel(...), tools=[])` whose final message is a plain `AIMessage(content=str)` with no `submit_query_answer` call and no `_nav_capture`. The query CLI MUST detect "no submit tool-call found" / "no `_nav_capture`" and fall back to `typer.echo(result["messages"][-1].content)` so those tests stay green. Do not change the fake-agent test harness.

**`src/agentic_rag/agents/lint.py`** + **`src/agentic_rag/tools/lint_tools.py`** — restructure:
- New deterministic core `src/agentic_rag/lint/health.py`:
```python
def health_check(wiki_path: Path) -> LintReport
```
  computes: orphan (0 inbound from content pages), missing-index, broken-link, missing-frontmatter, missing-related, empty (<50 words), stale (>90 days older than most-recent `updated`). Uses `wiki.load_wiki` + `wiki.search` graph data; 0 LLM calls. Returns a structured `LintReport`.
- `LintReport` model in `src/agentic_rag/schemas/lint.py`: `pages_audited:int; issues:list[Issue]; counts:dict[str,int]`. `Issue(slug:str, kind:str, severity:Literal["critical","high","medium","low"], detail:str, action:str)`.
- `write_lint_report(report: LintReport | str)` writes `wiki/lint-report-<date>.md` from the model deterministically (so even no-LLM mode produces the file). Keep the existing `@tool write_lint_report` accepting either a `LintReport` or a raw markdown string (back-compat for the scripted test).
- Lint agent tools becomes `[wiki_link_graph, wiki_read_page(), write_lint_report]` + receives the deterministic `health_check` result injected as a tool-output-style context message or via the system prompt. **Pin:** the agent first calls a tool `run_health_check()` (thin `@tool` over `health_check(wiki_path)`) to get the structured issues, then only emits the prose report + does semantic judgment (duplicate-coverage). This keeps it agentic where it adds value.
- Restore `find_inbound_links` + `extract_concepts` **as aliases** in `lint_tools.py` for test-collection compatibility (delegating to the new model/search), OR update the two importing tests to the new API — choose the approach that changes the least code; **pin: restore as aliases** so existing integration tests pass unchanged.

**`src/agentic_rag/middleware/guardrails.py`** — `path_guard_middleware` is **registered** in `build_agent` middleware list (currently defined but unused) OR deleted. **Pin: register it** in `build_agent` (after audit logging), accounting for nav tools' `slug`/`_source_path` args; add `read_index`/`search_index` to the read-set so writes are still blocked.

### Existing interfaces touched (back-compat constraints)
- `tools/shared.py::init_shared_tools`, `get_wiki_path` — unchanged signature; nav module reuses `_WIKI_PATH`.
- `io/index_manager.py` `read_index`, `find_in_index`, `upsert_entry`, `remove_entry`, `write_index` — kept (regenerate uses `write_index`/`_format_entry`). `find_in_index` becomes deprecated-but-present for tests.
- `io/wiki_io.py::list_pages`, `_resolve_page_path`, `read_page`, `write_page`, `delete_page`, `page_exists` — unchanged.
- `io/markdown_parser.py` `extract_links`, `extract_headings`, `parse_frontmatter`, `serialize_frontmatter`, `slugify` — unchanged.
- `agents/prompts.py` — only `build_query_prompt` and `build_lint_prompt` change in Pass A.
- `token_tracker.py`, `middleware/logging.py` — unchanged.

## Tasks summary

High-level ordering (atomic tasks live in `docs/progress.md`):
1. Unblock tests (restore aliases, sync `recursion_limit`, register guardrail).
2. Source-of-truth `Wiki` model + section/link extraction.
3. BM25 `wiki.search` + bounded link expansion.
4. `regenerate_index` from the model; regenerate the live `wiki/index.md`.
5. Nav tools (`wiki_search`/`wiki_read_page`/`wiki_summary`/`wiki_link_graph`).
6. `QueryAnswer` schema + `submit_query_answer` cite-or-die tool + `NavCapture` + `validate_citations`.
7. Query CLI rendering (structured + compat fallback).
8. Deterministic `health_check` + `LintReport` + lint agent restructure; restore lint test imports as aliases.

## Acceptance
- `uv run pytest` exits 0 (currently fails at collection). All pre-existing unit + integration tests still pass (aliases keep the lint/`find_inbound_links`/`extract_concepts` tests green; CLI tests green via the compat fallback).
- `Wiki` model loads the live `wiki/` (24 pages) without raising on the accented `málaga.md` or any frontmatter-less page.
- `regenerate_index(wiki_path)` rewrites `wiki/index.md` so **no entry summary is a raw `# H1` heading** and the link graph has no orphan entries that actually have inbound links.
- `wiki_search("machine learning models")` returns the `concepts/machine-learning` and related pages ranked above noise, **within ≤1 second** for the 24-page wiki and **without reading the whole graph** (expansion capped).
- Querying `"What is MLX?"` returns a `QueryAnswer` with `confidence`, and **every cited slug is a page `wiki_search`/`wiki_read_page` returned that turn** (validated — a deliberately fabricated citation in a test is dropped by `validate_citations`).
- `agentic-rag query "What is MLX?"` (against live wiki) prints answer + citations + confidence, not a raw model string.
- `health_check(wiki/)` runs **0 LLM calls**, returns a `LintReport` with correct counts; `agentic-rag lint` writes `wiki/lint-report-<date>.md` deterministically (same input → same structural issues; only the prose wrapper may vary).
- The dead `path_guard_middleware` is either registered (a write blocked when a tool is given a `raw/` path or absolute path is asserted by a unit test) or removed.

## Open questions
- None blocking. Vector/hybrid-embeddings graduation is *intentionally out of scope* and gated on a measured threshold (≥300 pages or a recall regression): not an open question, a deferred decision. Pass B (ingest compilation + fix-agent refactor) is a separate spec — not an open question here.

---

# Pass B — Ingest compilation + fix-agent refactor + eval harness

Pass A shipped the navigation/grounding/lint foundation and exported reusable interfaces
(Pass A handoffs in `docs/progress.md` are the contract Pass B builds on). Pass B finishes
the job so the two remaining agents do not undermine that foundation:

1. **Ingest** still mutates the derived `index.md` via `update_index` (re-corrupting what T4
   regenerates) and navigates with the 3 old weak tools (`read_index`/`search_index`/
   `find_relevant_pages`) — the exact slug-hallucination that produced ~9/20 polluted
   summaries. Pass B migrates ingest to `wiki_search`/`wiki_read_page` + the `Wiki` model,
   drops `update_index` (index regenerated at end), wires the already-defined-but-unused
   `ExtractionResult` schema via a `submit_extraction` tool (mirroring Pass A's
   `submit_query_answer`), and adds a **deterministic `match_page` matcher** (replaces the
   LLM round-trips guessing slugs with one Python call over `wiki.by_slug` + `search`).
2. **Fix** still shells out via `execute_command` (footgun + its own prompt says "NEVER run
   for loops/sed/pipes") and parses `lint-report-<date>.md` markdown to find issues.
   Pass B removes the shell tool, adds purpose-built safe page-edit tools, and makes `fix`
   consume the structured `LintReport` from Pass A's `health_check` directly.
3. A tiny **eval harness** (recall@k on `wiki.search` + a hallucination gate on
   `validate_citations`) protects the Pass A investment and gives the gate-reviewer a
   deterministic regression signal.

## Pass B Scope

### In scope (Pass B)
- `tools/ingest_grounding.py` (NEW): `submit_extraction` finalization tool (pure, mirrors
  `submit_query_answer`) that validates an `ExtractionResult` and returns its JSON. Reuses
  the existing `schemas/extraction.py` models (`Entity`/`Concept`/`Contradiction`/
  `ExtractionResult`) UNCHANGED — they are already defined and exported.
- `wiki/match.py` (NEW): pure `match_page` decision function + `@tool match_page(name, page_type)`.
- `regenerate_index` `@tool` (shared by ingest + fix; location pinned below).
- `io/chunker.py` (NEW): pure `chunk_by_heading(markdown, max_chars=4000) -> list[str]`.
- `agents/ingest.py` + `agents/prompts.py::build_ingest_prompt`: tool list rewrite + prompt rewrite.
- `tools/fix_tools.py` (CHANGED): remove `execute_command` + `remove_index_entry`; add
  `add_frontmatter` / `fix_link` / `append_related_section`.
- `agents/fix.py` + `agents/prompts.py::build_fix_prompt` + `cli.py::fix`: consume structured
  `LintReport`; drop shell HITL; auto-approve safe tools.
- `tests/eval/` (NEW): recall@k + hallucination-gate tests (no LLM).

### Out of scope (Pass B)
- Changes to `schemas/extraction.py` field shapes (reuse as-is).
- `create_page` / `update_page` / `delete_wiki_page` / `flag_contradiction` / `append_log`
  tool signatures (unchanged — ingest keeps using them).
- HITL policy for `delete_wiki_page` + `flag_contradiction` (unchanged).
- Vector/embeddings graduation (still gated per Pass A).
- Web URL ingestion, image handling, Marp/canvas, MCP server, durable checkpointer.

## Pass B Conventions (additions)
- Finalization-as-`@tool` pattern is the project standard (established by Pass A's
  `submit_query_answer`); `submit_extraction` follows it exactly so it is testable with
  `tests/fixtures/fake_llm.py::ScriptedChatModel`.
- Deterministic helpers (`match_page`, `chunk_by_heading`, `regenerate_index`, the fix tools)
  do 0 LLM calls; only extraction + the per-page write decisions use the model.
- New `@tool`s reuse `get_wiki_path()` from `tools/shared.py` (no new path global).
- `regenerate_index` `@tool` lives in **`tools/nav.py`** (already imports `load_wiki`;
  keeps all index/navigation tools in one module); `ingest` and `fix` both import it.

## Pass B Interfaces

**`src/agentic_rag/tools/ingest_grounding.py`** (NEW) — mirrors `tools/grounding.py`.
```python
@tool
def submit_extraction(entities: list[Entity], concepts: list[Concept],
                       contradictions: list[Contradiction]) -> str:
    ...    # returns ExtractionResult.model_dump_json(); PURE (no writes, no side effects)
```
- MUST be called by the ingest agent BEFORE any `create_page`/`update_page` (prompt enforces).
- No `NavCapture`-style store needed: extraction has no citation-binding; the tool exists
  purely to force a structured, testable extraction boundary.
- Returns the validated `ExtractionResult` JSON so the CLI/tests can inspect it from the
  `ToolMessage` (same reconstruction pattern Pass A's query CLI uses).

**`src/agentic_rag/wiki/match.py`** (NEW) — deterministic create/update/conflict matcher.
```python
def match_page(wiki: Wiki, name: str, page_type: str) -> MatchResult   # pure
@tool
def match_page_tool(name: str, page_type: str) -> str                 # thin @tool wrapper
```
- `MatchResult(decision: Literal["exact","similar","none","conflict"], slugs: list[str], detail: str)`.
- **Decision rule (PINNED — reuses Pass A `search` + `matched_via`, no score thresholds):**
  1. `candidate = slugify(name)`; if `candidate` (or its short form `candidate.rsplit('/',1)[-1]`)
     is a key in `wiki.by_slug` → `("exact", [resolved_slug], "exact slug match")`
  2. else `hits = search(wiki, name, k=5, types=[page_type])`; direct = `[h for h in hits if h.matched_via != "expand-link"]`:
     - `len(direct) == 1` → `("similar", [direct[0].slug], "BM25 match — update existing")`
     - `len(direct) >= 2` → `("conflict", [direct[0].slug, direct[1].slug], "multiple candidates — flag contradiction")`
     - `len(direct) == 0` → `("none", [], "no existing page — create new")`
  3. When `page_type` produces no `by_slug` directory (e.g. `"source"`), still run step 2
     without `types` filter (fall back to untyped search).
- Tool return string: `"<decision>: <slugs joined ', '> — <detail>"`.
- This replaces the old `search_index(name)` + `read_wiki_page(slug)` round-trip guessing.

**`src/agentic_rag/tools/nav.py`** (CHANGED — add one tool):
```python
@tool def regenerate_index() -> str    # calls wiki.dedupe_index.regenerate_index(get_wiki_path()); returns "Index regenerated."
```
- Replaces ingest's `update_index` calls and fix's `remove_index_entry`. The index is a
  derived view (Pass A T4); regenerating after writes keeps it in sync.

**`src/agentic_rag/io/chunker.py`** (NEW) — pure source-section chunker.
```python
def chunk_by_heading(markdown: str, max_chars: int = 4000) -> list[str]
```
- Splits on `^## ` headings (level-2 and deeper); a chunk exceeding `max_chars` is not
  split further (headings are the unit). Prepends the most recent `# `/`## ` breadcrumb
  to each chunk. Empty input → `[]`. No LLM. Used by the ingest prompt to extract
  per-section on large sources (the agent calls `read_source` once, then the prompt tells
  it to extract per chunk; chunker is importable for tests, not necessarily a tool).

**`src/agentic_rag/agents/ingest.py`** (CHANGED) — tool list rewrite:
- NEW tools = `[read_source, submit_extraction, match_page_tool, wiki_read_page,
  create_page, update_page, flag_contradiction, regenerate_index, append_log,
  delete_wiki_page]`.
- **DROPPED from the list** (keep importable for tests/back-compat): `read_index`,
  `search_index`, `find_relevant_pages`, the old `read_wiki_page` (shared), `update_index`.
- HITL middleware UNCHANGED (`delete_wiki_page` + `flag_contradiction`).

**`src/agentic_rag/agents/prompts.py::build_ingest_prompt`** (CHANGED) — rewrite:
- Mode 1 (file): `read_source` → `submit_extraction` (one structured pass over the source,
  chunk via `chunk_by_heading` mentally if large) → for each `Entity`/`Concept`: call
  `match_page(name, type)` once → branch on decision: `exact`/`similar` → `update_page`;
  `none` → `create_page`; `conflict` → `flag_contradiction` (HITL). Update `## Related`
  cross-links. Create a `sources/<slug>.md` summary page. End with `regenerate_index` +
  `append_log(op="ingest")`.
- Mode 2 (natural language): `wiki_search` to find affected pages → `wiki_read_page` →
  update/create → `regenerate_index` + `append_log`.
- Rules: never write outside `wiki/`; never modify `raw/`; never delete without HITL;
  never ignore a contradiction; **never call `update_index`** (index is regenerated); 
  always end with `regenerate_index` + `append_log`.

**`src/agentic_rag/tools/fix_tools.py`** (CHANGED):
```python
@tool def add_frontmatter(slug: str, title: str, page_type: str) -> str
@tool def fix_link(slug: str, old_target: str, new_target: str) -> str
@tool def append_related_section(slug: str, links: list[str]) -> str
```
- REMOVED: `execute_command` (and `run_command` helper) and `remove_index_entry`.
- `add_frontmatter`: reads page via `wiki_io.read_page`; builds `Frontmatter(slug, type=page_type,
  title=title, sources=[], updated=date.today(), tags=[])`; writes `serialize_frontmatter(fm)+body`
  via `wiki_io.write_page`. Error if page already has frontmatter (starts with `---`).
- `fix_link`: text-replace `[[old_target]]`→`[[new_target]]` AND `[[old_target|X]]`→`[[new_target|X]]`
  (single occurrence each, via `edit_wiki_page` logic or direct); returns count replaced.
- `append_related_section`: if page has no `## Related` section, appends `\n## Related\n\n` +
  `- [[link]]` per link; if present, appends the new links to it. Error string if page missing.
- KEEP `edit_wiki_page` (general text fix) for fixes not covered by the specific tools.

**`src/agentic_rag/agents/fix.py`** (CHANGED):
- tools = `[wiki_read_page, edit_wiki_page, add_frontmatter, fix_link,
  append_related_section, regenerate_index, delete_wiki_page]`.
- HITL ONLY on `delete_wiki_page` (drop the `execute_command` interrupt config + the
  `_needs_approval`/`_READ_ONLY_PREFIXES` machinery — no shell tool remains).
- Auto-approve all the new safe write tools (they never escape `wiki_path`).

**`src/agentic_rag/agents/prompts.py::build_fix_prompt`** (CHANGED) — rewrite:
- The agent receives the structured issues as context (CLI passes them, below), NOT by
  reading `lint-report-YYYY-MM-DD.md`.
- Issue-kind → tool map (PINNED): `missing-frontmatter`→`add_frontmatter`;
  `broken-link`→`fix_link`; `missing-related`→`append_related_section`;
  `missing-index`→`regenerate_index`; `orphan`/`empty`/`stale`→report only (need human
  judgment or content; use `edit_wiki_page` only if an obvious fix exists).
- Fix one issue per tool call, verify by `wiki_read_page`, move on.

**`src/agentic_rag/cli.py::fix`** (CHANGED):
- Run `health_check(settings.wiki_path)` → `LintReport`; serialize `report.issues` to a
  compact one-line-per-issue context string and pass it as the user message to the agent
  (e.g. `"Fix these lint issues:\n- [missing-frontmatter] entities/mlx: ...\n- ..."`).
- Keep the `--issue` arg as an optional filter: if provided, filter `report.issues` to
  those whose `kind`/`slug` matches before sending. Keep accepting `"latest"` as a no-op
  alias for back-compat (it now means "run health_check").
- Compat: if `health_check` raises (empty wiki), fall back to the old behavior of echoing
  "No issues" (do not crash).

## Pass B Existing interfaces touched (back-compat constraints)
- `schemas/extraction.py` — UNCHANGED (reused as-is; already exported from `schemas/__init__.py`).
- `tools/ingest_tools.py` `create_page`/`update_page`/`delete_wiki_page`/`flag_contradiction`/
  `append_log`/`read_source` — UNCHANGED signature/behavior. `update_index` stays importable;
  just no longer listed in the ingest agent's tools.
- `tools/query_tools.py::find_relevant_pages` — kept importable; dropped from ingest tools.
- `tools/shared.py` `read_index`/`search_index`/`read_wiki_page` — kept importable (unit tests).
- `wiki/search.py::search`, `wiki/model.py::load_wiki`, `wiki/dedupe_index.regenerate_index`,
  `schemas/lint.py`, `lint/health.py::health_check` — consumed UNCHANGED.
- `agents/factory.py::build_agent` + `path_guard_middleware` — UNCHANGED; the new fix tools
  are write-tools the guardrail must NOT block (their `slug` args are within-wiki reads/writes;
  the guardrail's existing write-tools set needs `add_frontmatter`/`fix_link`/
  `append_related_section` added — PIN: add them to the guardrail's write-tools set so a
  `raw/`/absolute/`..` slug is rejected, but in-wiki slugs pass).

## Pass B Tasks summary
9. `tools/ingest_grounding.py` `submit_extraction` + `io/chunker.py` chunker (ingest prep).
10. `wiki/match.py` deterministic `match_page` + `@tool match_page`.
11. Ingest agent migration (drop `update_index`+old retrieval; `wiki_search`/`submit_extraction`/
    `match_page`/`regenerate_index`; rewrite `build_ingest_prompt`).
12. Fix tools refactor (`add_frontmatter`/`fix_link`/`append_related_section`; remove
    `execute_command`/`remove_index_entry`; guardrail set update).
13. Fix agent + CLI consume structured `LintReport` (rewrite `build_fix_prompt` + `cli.fix`).
14. Eval harness (`tests/eval/` recall@k + hallucination gate).

## Pass B Acceptance
- `uv run pytest` exits 0 including the new `tests/eval/` suite (no LLM required for eval).
- Ingest agent's tool list NO LONGER contains `update_index`, `read_index`, `search_index`, or
  `find_relevant_pages`; it DOES contain `submit_extraction`, `match_page`, `regenerate_index`.
- A scripted ingest integration test (fake `ScriptedChatModel`) calls `submit_extraction` then
  `match_page` then `create_page`/`update_page` then `regenerate_index` then `append_log`; NO
  `update_index` call occurs; `index.md` is regenerated (one `regenerate_index` call).
- `match_page(wiki, "MLX", "entity")` on the live wiki returns `"exact: entities/mlx ..."`;
  `match_page(wiki, "Nonexistent Thing", "entity")` returns `"none ..."`; a query matching ≥2
  pages returns `"conflict: ..."`.
- Fix agent's tool list NO LONGER contains `execute_command`; it DOES contain `add_frontmatter`,
  `fix_link`, `append_related_section`, `regenerate_index`.
- `agentic-rag fix missing-frontmatter` (or `fix latest`) runs `health_check`, passes structured
  issues to the fix agent, and the agent's first action is driven by the issue kinds (no
  `read_wiki_page('lint-report-...')` markdown-parse call).
- A `missing-frontmatter` issue on a fixture page is fixed by `add_frontmatter` (page gains valid
  frontmatter, re-`load_wiki` parses it). A `broken-link` issue is fixed by `fix_link`.
- Eval: `tests/eval/test_search_recall.py` asserts recall@8 ≥ 0.8 on the curated ~15-query set
  against the live `wiki/`; `tests/eval/test_grounding_gate.py` asserts a fabricated citation is
  dropped by `validate_citations`.
- The guardrail blocks an `add_frontmatter`/`fix_link`/`append_related_section` call with a
  `raw/` or absolute slug (unit test); in-wiki slugs pass.

## Pass B Open questions
- None blocking. "Should `fix` auto-run deterministic-safe fixes (missing-frontmatter,
  broken-link, missing-related, regenerate_index) without an LLM at all?" is a later UX decision —
  Pass B keeps the fix agent in the loop (agentic judgment on orphan/empty/stale), so all kinds
  still route through the agent. Automating the purely-deterministic kinds is a deferred tweak.