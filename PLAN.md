# Agentic RAG — LLM Wiki Implementation Plan

Status: draft for agent-led implementation (orchestrator-subagents architecture).
Owner: parent orchestrator (Pi main session). Implemented by subagents.

## 1. Overview

Build an **agentic RAG system** that maintains a persistent, interlinked markdown wiki from raw sources. Unlike classic RAG (re-derive answers from chunks each query), the wiki is a *compounding artifact*: cross-references, contradictions, and synthesis are compiled once and kept current. See `IDEA.md` for the core concept and `wiki/` for a real working example.

Three layers (from `IDEA.md`):
1. **Raw sources** (`raw/`, immutable). LLM reads, never writes.
2. **Wiki** (`wiki/`, LLM-owned). Summaries, entity/concept/comparison pages, `index.md`, `log.md`.
3. **Schema** (`AGENTS.md`): conventions injected into agent system prompts.

### Framework choice (KEY CHANGE vs prior draft)

Agents are built with **LangChain `create_agent()`** + **middleware** (per `langchain-fundamentals` / `langchain-middleware` skills). We do **not** hand-roll a LangGraph `StateGraph` per agent. Each agent is a `create_agent` instance: model + tools (via `@tool`) + system prompt + optional `HumanInTheLoopMiddleware` + `MemorySaver` checkpointer. The agent loop *is* the graph. This is the supported modern path ("When creating LangChain agents, you MUST use `create_agent()`").

## 2. Goals / Non-Goals

**In scope (MVP)**
- Three agents: **Ingest**, **Query**, **Lint** — each a `create_agent` instance.
- Source ingestion via **MarkItDown** (md, txt, pdf, docx, pptx, xlsx, html, images-OCR, …).
- Retrieval = **index-driven** (`index.md` → drill into pages). No vector store yet.
- Human-in-the-loop (HITL) on: `delete_wiki_page` and **contradiction handling** only. Bookkeeping (`append_log`, `update_index`) and page writes auto.
- `AGENTS.md` schema loaded into every agent system prompt.
- Typer CLI: `ingest`, `query`, `lint`, `status`, `log`.
- Existing `wiki/` kept as integration-test fixture + real starting state.
- Tests: unit + integration + acceptance (lint of real wiki).

**Non-goals (deferred, scaffold only)**
- Vector store / hybrid search (Chroma). Scaffold interface, do not implement.
- Filing query answers back as new wiki pages.
- MCP server / multi-user / durable SQLite checkpointer / image handling / Marp export / web URL ingestion.

## 3. Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Agent framework | LangChain `create_agent` + middleware (NOT raw LangGraph StateGraph) |
| LLM provider | OpenAI-compatible via `base_url` (OpenAI, Azure, OpenRouter, local proxy). Model + key from `.env`. |
| Persistence | `MemorySaver()` in-process per CLI invocation (memoryless MVP). Durable checkpointer deferred. |
| Source ingestion | `markitdown[all]` (Microsoft MarkItDown) — converts any source → markdown |
| Wiki markdown parsing | `markdown-it-py` (parse existing wiki pages for links/structure) |
| Config | `pydantic-settings` `BaseSettings` (`.env` + `config.yaml`) |
| CLI | `typer` |
| Tests | `pytest`, `pytest-asyncio`, `respx` (HTTP mock) or a fake-LLM harness |

## 4. Directory Structure

```
agentic_rag/
├── src/agentic_rag/
│   ├── __init__.py
│   ├── config.py                 # Pydantic BaseSettings (Settings)
│   ├── paths.py                  # Path resolution helpers
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── agents_md.py          # AGENTS.md parser/loader → injected prompt section
│   │   ├── wiki.py               # Pydantic models for page, frontmatter, index entry, log entry, link
│   │   └── extraction.py         # Pydantic structured-output models (Entity, Concept, Contradiction, ExtractionResult)
│   ├── io/
│   │   ├── __init__.py
│   │   ├── source_loader.py      # MarkItDown wrapper: path|url → markdown str
│   │   ├── wiki_io.py            # read/write/delete wiki pages, list pages, frontmatter ops
│   │   ├── markdown_parser.py    # markdown-it-py: extract [[links]], headings, frontmatter
│   │   ├── index_manager.py      # read/update/parse index.md (atomic)
│   │   └── log_manager.py        # append-only log.md (timestamped, prefixed entries)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── shared.py             # read_index, read_wiki_page, search_index (text match over index)
│   │   ├── ingest_tools.py       # read_source, create_page, update_page, delete_wiki_page, update_index, append_log, flag_contradiction
│   │   ├── query_tools.py        # find_relevant_pages, read_page, synthesize (notebookLM-style)
│   │   └── lint_tools.py        # read_all_pages, find_inbound_links, extract_concepts, write_lint_report
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── factory.py            # build_agent(model, tools, system_prompt, middleware, checkpointer) helper
│   │   ├── ingest.py             # build_ingest_agent() — create_agent + middleware
│   │   ├── query.py              # build_query_agent()
│   │   ├── lint.py               # build_lint_agent()
│   │   └── prompts.py            # system prompt builder (injects AGENTS.md + role instructions)
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── logging.py            # wrap_tool_call: log every tool call + args (audit trail)
│   │   ├── guardrails.py         # wrap_tool_call: forbid writes outside wiki_path; forbid touching raw/
│   │   └── hitl.py               # HumanInTheLoopMiddleware config + contradiction resume helpers
│   ├── cli.py                    # Typer app: ingest/query/lint/status/log
│   └── main.py
├── wiki/                         # real starting state + integration fixture (unchanged content)
├── raw/                          # sample sources for tests
├── tests/
│   ├── unit/
│   │   ├── test_source_loader.py
│   │   ├── test_wiki_io.py
│   │   ├── test_markdown_parser.py
│   │   ├── test_index_manager.py
│   │   ├── test_log_manager.py
│   │   ├── test_agents_md_loader.py
│   │   └── test_tools.py
│   ├── integration/
│   │   ├── test_ingest_agent.py
│   │   ├── test_query_agent.py
│   │   ├── test_lint_agent.py
│   │   ├── test_hitl_contradiction_flow.py
│   │   ├── test_hitl_delete_flow.py
│   │   └── test_cli.py
│   ├── acceptance/
│   │   ├── test_wiki_health.py   # lint the shipped wiki/ with real LLM
│   │   └── test_ingest_real_source.py
│   ├── fixtures/
│   │   ├── wiki/                 # frozen copy of repo wiki/ for deterministic tests
│   │   ├── sources/              # small md, pdf, docx samples
│   │   └── fake_llm.py           # FakeChatModel returning scripted tool calls
│   └── conftest.py
├── AGENTS.md                     # wiki schema (conventions) — loaded into prompts
├── config/config.yaml.example
├── pyproject.toml
├── .env.example
└── README.md
```

## 5. Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # LLM (OpenAI-compatible)
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4.1-mini"

    # Paths
    wiki_path: Path = Path("./wiki")
    raw_sources_path: Path = Path("./raw")
    agents_md_path: Path = Path("./AGENTS.md")

    # Agent runtime
    recursion_limit: int = 12          # cap agent loop turns
    hitl_enabled: bool = True

    # Retrieval (MVP: index-only)
    retrieval_mode: str = "index"      # future: "hybrid" (index + vector)
    vector_db_path: Path | None = None # scaffold; unused in MVP

    # MarkItDown
    markitdown_llm_describe_images: bool = False  # requires key; off by default

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

`.env.example`:
```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
WIKI_PATH=./wiki
RAW_SOURCES_PATH=./raw
AGENTS_MD_PATH=./AGENTS.md
RETRIEVAL_MODE=index
```

## 6. AGENTS.md Schema (loaded into prompts)

Reside at repo root. Loaded by `schemas/agents_md.py` → injected verbatim into each agent system prompt under a `# Wiki Schema` section. Defines:

- **Page types**: entity, concept, source, comparison, overview.
- **Naming**: `entities/<slug>.md`, `concepts/<slug>.md`, `sources/<slug>.md`, `comparisons/<a>-vs-<b>.md`.
- **Cross-reference format**: Obsidian `[[Page Name]]`; every page has a `## Related` section.
- **Frontmatter** (YAML): `slug`, `type`, `title`, `sources` (list), `updated` (ISO date), `tags`.
- **Update rules**: new info supersedes old (flag contradictions via `flag_contradiction`); always `update_index` after writes; always `append_log`; date all changes.
- **Index entry format** and **log prefix format** (see §10).
- **What the agent must NEVER do**: write outside `wiki/`, modify `raw/`, delete without HITL, leave orphan pages.

`AGENTS.md` is co-evolved with the user over time (IDEA.md §Architecture). MVP ships a sensible default.

## 7. Source Ingestion — MarkItDown

`io/source_loader.py` wraps `markitdown.MarkItDown`:

```python
from markitdown import MarkItDown
from pathlib import Path

class SourceLoader:
    def __init__(self, settings: Settings):
        kwargs = {}
        if settings.markitdown_llm_describe_images:
            from openai import OpenAI
            kwargs["llm_client"] = OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
            kwargs["llm_model"] = settings.openai_model
        self._md = MarkItDown(**kwargs)

    def load(self, source: str) -> str:
        """source = path or URL (markitdown handles youtube/wikipedia/html too, but MVP only paths)."""
        result = self._md.convert(str(source))
        return f"# Source: {Path(source).name}\n\n{result.text_content}"
```

- Formats supported out of the box: pdf, docx, pptx, xlsx, html, csv, json, xml, ipynb, images (EXIF/OCR), epub, zip.
- For MVP, restrict CLI to local file paths (URL ingestion deferred — but `markitdown` already supports URLs, so enabling later is trivial).
- **Gotchas to handle** (from skill): DOCX embedded images use absolute paths; PDF OCR confidence not surfaced; XLSX merged cells flatten; HTML CSS layout collapses. Document in `AGENTS.md` guidance: "treat extracted figures as low-confidence unless source confirms."

## 8. Wiki I/O Layer

`io/wiki_io.py` — filesystem ops on `wiki/`:
- `list_pages() -> list[Path]` (recursive `.md`, excludes `index.md`/`log.md`).
- `read_page(slug) -> str` (raw markdown).
- `read_page_with_frontmatter(slug) -> tuple[Frontmatter, str]` (uses `markdown-it-py` + frontmatter parse).
- `write_page(slug, content, frontmatter) -> Path` (atomic write: temp + rename; creates parent dirs).
- `delete_page(slug)` (HITL-guarded via middleware, not here).
- `page_exists(slug) -> bool`.

`io/markdown_parser.py` — `markdown-it-py`:
- `extract_links(content) -> list[str]` — all `[[Target]]` links (Obsidian-style).
- `extract_headings(content) -> list[Heading]`.
- `parse_frontmatter(content) -> Frontmatter`.
- `serialize_frontmatter(fm) -> str`.
- `slugify(name) -> str`.

`io/index_manager.py`:
- `read_index() -> Index` (parse categories, entries).
- `upsert_entry(entry)`, `remove_entry(slug)`.
- `write_index(index)` (atomic).
- `find_in_index(query) -> list[IndexEntry]` (simple substring/keyword score on summary).

`io/log_manager.py`:
- `append(entry: LogEntry)` — append-only, prefix format `## [YYYY-MM-DD HH:MM] <op> | <title>`.
- `tail(n) -> list[LogEntry]` (parse trailing entries via regex on prefix).

## 9. Agents

All three built via `agents/factory.py`:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

def build_agent(model, tools, system_prompt, middleware=None):
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        checkpointer=MemorySaver(),   # required for HITL + per-invocation memory
    )
```

Invocation always threaded: `config={"configurable": {"thread_id": <uuid>}, "recursion_limit": settings.recursion_limit}`.

### 9.1 Ingest Agent

**Purpose:** read a source, extract entities/concepts, integrate into the wiki (create/update pages), flag contradictions (HITL), update index + log.

**Tools** (`tools/ingest_tools.py`, all `@tool`):

| Tool | Args | HITL | Notes |
|------|------|------|-------|
| `read_source` | `source_path: str` | no | calls `SourceLoader.load` |
| `read_index` | — | no | shared |
| `search_index` | `query: str` | no | shared, keyword match |
| `read_wiki_page` | `slug: str` | no | shared |
| `create_page` | `slug, type, content, frontmatter` | no | new pages only (errors if exists) |
| `update_page` | `slug, content, frontmatter` | no | overwrites existing |
| `delete_wiki_page` | `slug: str` | **YES** | HITL approve/reject |
| `update_index` | `entries` | no | upsert entries for changed pages |
| `append_log` | `op, title, details` | no | append-only log entry |
| `flag_contradiction` | `page_slug, existing_claim, new_claim, proposed_resolution` | **YES** | interrupt; resume approve→apply resolution; reject→keep old + note |

**Middleware:**
- `HumanInTheLoopMiddleware(interrupt_on={"delete_wiki_page": {"allowed_decisions": ["approve","reject"]}, "flag_contradiction": {"allowed_decisions": ["approve","edit","reject"]}})` — **requires** checkpointer + `thread_id`.
- `wrap_tool_call` logging (audit every tool call to stdout + a `wiki/.audit.jsonl`).
- `wrap_tool_call` path guard: reject any tool writing outside `settings.wiki_path`; hard-fail on touching `raw/`.

**System prompt** (built by `agents/prompts.py`):

```
You are the Ingest Agent for a persistent LLM-maintained wiki.

# Wiki Schema
<AGENTS.md contents injected here>

# Workflow
1. Call read_source(path) to get the source markdown (already converted by MarkItDown).
2. Identify entities and concepts. For each:
   a. search_index(name) and read_wiki_page(slug) to check existing coverage.
   b. If a page exists and the source changes a factual claim that CONFLICTS with the page → call flag_contradiction(page_slug, existing_claim, new_claim, proposed_resolution). Wait for the human decision before writing.
   c. Otherwise create_page (new) or update_page (exists, non-conflicting update).
3. Update every Related section and add cross-links [[Page]].
4. Create a source summary page under sources/<slug>.md.
5. Call update_index for all created/updated pages.
6. Call append_log with op="ingest", title=source name, details=list of pages touched.

# Hard rules
- Never write outside wiki/.
- Never modify raw/.
- Never delete a page without the delete_wiki_page tool (HITL).
- Never ignore a contradiction — always call flag_contradiction.
- Always end by updating index and log.
```

**Structured output (optional helper):** a non-agent extraction call `model.with_structured_output(ExtractionResult)` may be used as a *first pass* to prime the agent with a candidate entity/concept list, then the agent refines via tools. MVP keeps extraction inside the agent loop (model is capable); `schemas/extraction.py` ships the Pydantic models for future use and tests.

### 9.2 Query Agent

**Purpose:** answer a question against the wiki with citations, read-only.

**Tools** (`tools/query_tools.py`):

| Tool | Args | Notes |
|------|------|-------|
| `read_index` | — | always first |
| `search_index` | `query` | keyword/summary match |
| `read_wiki_page` | `slug` | follow `[[links]]` |
| `find_relevant_pages` | `query` | combines index scan + link traversal (returns slugs) |

**No HITL.** No `delete`/`write` tools exposed — agent cannot mutate wiki (consistent with Q5: no answer filing for MVP).

**System prompt:**
```
You are the Query Agent for a persistent LLM wiki.
# Wiki Schema
<AGENTS.md>

# Workflow
1. read_index → identify candidate pages.
2. search_index(question) → augment candidates.
3. read_wiki_page for each candidate; follow cross-links as needed.
4. Synthesize an answer with inline citations: "claim ([[/page-slug]])."
5. Return markdown answer + a "Sources consulted" list (page slugs + source titles).

# Rules
- Read-only. Do not call any write tool (none provided).
- If the wiki does not cover the question, say so explicitly and suggest sources to ingest.
```

### 9.3 Lint Agent

**Purpose:** health-check the wiki. Output a report; HITL only if it wants to delete/merge pages (rare).

**Checks:**
1. Contradictions between pages.
2. Stale claims superseded by newer sources (compare `updated` dates).
3. Orphan pages (no inbound `[[links]]`).
4. Missing pages (concept mentioned via `[[X]]` but no page file).
5. Missing cross-references (related pages not linked).
6. Data gaps (areas suitable for a new source/question).

**Tools** (`tools/lint_tools.py`):

| Tool | Args | Notes |
|------|------|-------|
| `read_all_pages` | — | returns `{slug: content}` |
| `read_index` | — | |
| `find_inbound_links` | `slug` | grep all pages for `[[slug\|name]]` |
| `extract_concepts` | `content` | heading/`[[link]]` extraction → concept names |
| `delete_wiki_page` | `slug` | **YES** HITL (only if lint decides to prune) |
| `write_lint_report` | `report` | saves to `wiki/lint-report-YYYY-MM-DD.md` |

**Middleware:** `HumanInTheLoopMiddleware(interrupt_on={"delete_wiki_page": {"allowed_decisions":["approve","reject"]}})`. Default behavior: **report only**, suggest; never auto-delete.

**System prompt:**
```
You are the Lint Agent. Audit wiki health and WRITE A REPORT. Default to suggestions, not deletions.
# Wiki Schema
<AGENTS.md>

# Workflow
1. read_all_pages + read_index.
2. For each page: find_inbound_links → detect orphans.
3. For each [[X]] link with no target file → missing page.
4. Compare overlapping claims across pages → contradictions / stale claims.
5. Suggest missing cross-references and data gaps (new questions/sources to investigate).
6. write_lint_report(report) to wiki/lint-report-YYYY-MM-DD.md.
7. If a page is clearly empty/duplicate and must be removed, call delete_wiki_page (HITL).

# Rules
- Prefer reporting over mutation.
- Never modify content pages directly.
- Cite page slugs + line references in findings.
```

## 10. `index.md` & `log.md` Formats

### index.md (content-oriented catalog)
```markdown
# Wiki Index

## Entities
- [[Python]] - High-level programming language | Source: manual | Updated: 2026-04-02
- [[Entity B]] - ... | Sources: 3 | Updated: 2026-04-01

## Concepts
- [[Tool calling]] - LLM function-invocation capability | Sources: 1 | Updated: 2026-07-20

## Sources
- [Cv](sources/cv.md) - Ingested: 2026-07-20

## Comparisons
- [[A vs B]] - ... | Sources: 2 | Updated: 2026-04-02
```
Keeps current `wiki/index.md` shape (see example). Slug in root namespace, file under `entities/`/`concepts/`/etc.

### log.md (append-only, prefix-parseable)
```markdown
# Wiki Log

## [2026-07-20 18:32] ingest | Cv
- Created: [[Álvaro ...]], [[Tool calling]], ...
- Contradiction flagged: <page> — <old> vs <new> — RESOLVED: kept new

## [2026-08-01 09:15] query | What is the relationship between A and B?
- Pages read: A, B, Tool calling

## [2026-08-02 10:00] lint | Health check
- Found: 2 orphans, 1 contradiction, 3 missing links
- Report: lint-report-2026-08-02.md
```
Prefix `## [YYYY-MM-DD HH:MM] <op> | <title>` enables `grep "^## \[" log.md | tail -5` (IDEA.md tip).

## 11. CLI (`cli.py`, Typer)

```
agentic-rag ingest <path>            # run ingest agent; HITL prompts inline
agentic-rag query "<question>"       # read-only answer + citations
agentic-rag lint                      # health report (writes report file)
agentic-rag status                    # page counts, last log entry, orphans quick-scan
agentic-rag log [--tail N]            # tail log.md
```

CLI flow for `ingest`:
```python
config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": settings.recursion_limit}
result = agent.invoke({"messages":[{"role":"user","content":f"Ingest {path}"}]}, config=config)
while "__interrupt__" in result:
    # present interrupt (delete_wiki_page args OR flag_contradiction details) to user
    decision = prompt_user(result["__interrupt__"])   # approve/reject/edit
    result = agent.invoke(Command(resume={"decisions":[{"type":decision.type, **decision.extra}]}), config=config)
print(result["messages"][-1].content)
```
`Command` from `langgraph.types`. Decision shapes per skill (`type: "approve"|"edit"|"reject"`, `feedback` for reject, `edited_action` for edit).

## 12. Testing Strategy

Three levels. Mock LLM via a `FakeChatModel` in `tests/fixtures/fake_llm.py` returning scripted `AIMessage` with `tool_calls`, so agent loop is deterministic without network.

### 12.1 Unit tests (`tests/unit/`)

- **test_source_loader.py**: pdf/docx/md fixtures → MarkItDown converts → markdown contains expected heading; non-existent file raises; frontmatter-less sources handled.
- **test_wiki_io.py**: write→read roundtrip; atomic write (no partial); `list_pages` excludes `index.md`/`log.md`; `page_exists`.
- **test_markdown_parser.py**: `extract_links` finds `[[A]]` and `[[A|alias]]`; `extract_headings` depth; `parse_frontmatter` YAML; `slugify` ("3D Gaussian Splatting" → `3d-gaussian-splatting`).
- **test_index_manager.py**: parse current `wiki/index.md`; upsert adds entry, preserves categories; remove_entry; `find_in_index` ranks by summary keyword.
- **test_log_manager.py**: append creates prefix correctly; `tail(5)` returns last 5; regex prefix parse.
- **test_agents_md_loader.py**: missing `AGENTS.md` → returns default; present → injected string contains schema section verbatim.
- **test_tools.py**: each tool round-trips against a temp `wiki/` fixture; `create_page` errors if slug exists; `update_page` errors if missing; `delete_wiki_page` removed from disk; `search_index` returns ranked slugs.

### 12.2 Integration tests (`tests/integration/`)

Use `FakeChatModel` to drive `create_agent` so loop runs without API.

- **test_ingest_agent.py**: feed small `raw/sample.md`; fake LLM scripted to call `read_source` → `search_index` → `create_page` ×2 → `update_index` → `append_log`. Assert: pages created on disk, index updated, log entry appended with correct prefix, recursion not exceeded.
- **test_query_agent.py**: ask "What is MLX?"; fake LLM calls `read_index` → `read_wiki_page("mlx")` → returns answer citing `[[MLX]]`; assert no write tool called.
- **test_lint_agent.py**: point at a fixture wiki with 1 orphan + 1 missing target; fake LLM calls `read_all_pages` → `find_inbound_links` → `write_lint_report`; assert report file exists and lists orphans.
- **test_hitl_contradiction_flow.py**: ingest scenario where `flag_contradiction` fires → `__interrupt__` present → `Command(resume={"decisions":[{"type":"approve"}]})` → agent applies resolution → final state has updated page + log notes "RESOLVED". Also test `reject` path logs "kept old".
- **test_hitl_delete_flow.py**: lint scenario → `delete_wiki_page` interrupts → approve → file gone from disk; reject → file remains, log notes "delete rejected".
- **test_cli.py**: `CliRunner` invoke `ingest/query/lint/status/log` against fixture wiki; assert exit codes, stdout shows interrupt prompts and resume reads.

### 12.3 Acceptance tests (`tests/acceptance/`)

Run against **real LLM** (skipped if `OPENAI_API_KEY` unset). Use shipped `wiki/` as starting state.

- **test_wiki_health.py**: run Lint agent over the real `wiki/`; assert it returns a structured report, no exceptions, recursion within limit, report file written. Record findings (orphans, contradictions) as baseline — do not hardcode counts (wiki evolves).
- **test_ingest_real_source.py**: ingest `raw/cv.pdf` (the CV already producing the shipped wiki) into a *fresh temp copy* of `wiki/`; assert ≥5 pages created/updated, index + log consistent (all created slugs appear in index, all index entries have files), no `[[X]]` link is dangling for created pages. Confirm MarkItDown + agent end-to-end works.

Acceptance gate: agents must not touch `raw/`, must not write outside `wiki/`, must respect HITL on delete + contradiction. Enforced by `middleware/guardrails.py` + verified by tests.

## 13. Implementation Phases & Subagent Orchestration

Implemented by the **parent orchestrator** using the pi-subagents **review-loop technique**: per phase → async `worker` implements → fresh-context `reviewer`s inspect the real diff → parent synthesizes accepted fixes → async forked `worker` applies → repeat until reviewers find no blockers (default 3 rounds). One writer thread per phase. Reviewers read-only.

Before any phase: optional `scout` recon to confirm file/integration points (cheap, fresh-context).

### Phase 0 — Scaffolding & config
Scope: `pyproject.toml`, deps, `config.py`, `paths.py`, dir layout, `.env.example`, `AGENTS.md` (default schema), `README.md` skeleton, `conftest.py`.
Subagent task example:
```
worker: "Implement Phase 0 scaffolding per PLAN.md §4–§6. Create pyproject.toml with deps from §15, config.py BaseSettings, AGENTS.md default schema. Run `pip install -e .` and report status."
reviewer (fresh): "Verify pyproject deps resolve, config loads from .env.example, AGENTS.md is valid markdown and matches §6. Report findings only."
```

### Phase 1 — Wiki I/O + parsers
Scope: `io/source_loader.py` (MarkItDown), `io/wiki_io.py`, `io/markdown_parser.py`, `io/index_manager.py`, `io/log_manager.py`, `schemas/wiki.py`, `schemas/agents_md.py`.
Tests: all unit tests in §12.1.
Subagent tasks:
```
worker: "Implement Phase 1 per PLAN.md §7–§8. All unit tests in tests/unit/test_{source_loader,wiki_io,markdown_parser,index_manager,log_manager,agents_md_loader}.py must pass. Use pytest."
reviewer: "Review Phase 1 diff for: path-safety (no traversal), atomic writes, correct [[link]] regex incl aliases, slug collisions, index/log format match §10. Report file:line findings; no edits."
```

### Phase 2 — Tools
Scope: `tools/shared.py`, `tools/ingest_tools.py`, `tools/query_tools.py`, `tools/lint_tools.py`, `schemas/extraction.py`. Tools operate against `io/` layer.
Tests: `test_tools.py`.
Subagent tasks:
```
worker: "Implement Phase 2 tools per §9 tool tables. Every @tool has a clear description + Args (skill warns about vague descriptions). test_tools.py passes."
reviewer: "Check tool descriptions are specific (§skills), schemas match Pydantic, HITL-marked tools (delete_wiki_page, flag_contradiction) are NOT accidentally wired for auto-execution. No edits."
```

### Phase 3 — Agents + middleware + prompts
Scope: `agents/factory.py`, `agents/prompts.py`, `agents/ingest.py|query.py|lint.py`, `middleware/{logging,guardrails,hitl}.py`. Build `create_agent` instances with `HumanInTheLoopMiddleware` + `MemorySaver` + thread_id config.
Tests: all integration tests in §12.2 (with `FakeChatModel`).
Subagent tasks:
```
worker: "Implement Phase 3 agents per §9. Use create_agent (NOT StateGraph). HITL on delete_wiki_page + flag_contradiction only. Guardrails forbid writes outside wiki/. Integration tests pass with FakeChatModel."
reviewer: "Verify (a) create_agent used per langchain-fundamentals skill, (b) HumanInTheLoopMiddleware config matches skill syntax + checkpointer present, (c) Command(resume=...) shapes match skill, (d) recursion_limit set, (e) AGENTS.md actually injected into prompts, (f) no raw LangGraph StateGraph. Report findings."
```

### Phase 4 — CLI + HITL wiring
Scope: `cli.py` (Typer), interrupt prompt + `Command(resume=...)` loop, `status`/`log`.
Tests: `test_cli.py`.
Subagent tasks:
```
worker: "Implement CLI per §11. Interrupt loop uses Command(resume={decisions:[{type:...}]}) exact shapes from langchain-middleware skill. test_cli.py passes."
reviewer: "Verify resume decision shapes (approve/edit/reject + feedback/edited_action), thread_id per invocation, recursion_limit applied. Report findings."
```

### Phase 5 — Acceptance + polish
Scope: `tests/acceptance/`, real-LLM runs, README, error handling pass.
Subagent tasks:
```
worker: "Write acceptance tests per §12.3. Run against real wiki/ with OPENAI_API_KEY. Gate: no writes outside wiki/, no touches to raw/, HITL respected. Update README with usage."
reviewer: "Run Lint agent over shipped wiki/; confirm report + no exceptions. Verify acceptance tests skip cleanly without key. Report findings."
```

Parent synthesizes all reviewer findings per phase, launches fix-worker for accepted fixes only, stops when reviewers find no blockers or 3 rounds reached. Builtin agents used: `scout`, `worker`, `reviewer`, optionally `oracle` for direction disputes and `context-builder` if a phase needs heavy handoff.

## 14. State / Memory

- MVP: `MemorySaver()` per CLI invocation. State lives only for the duration of one `ingest`/`lint` (needed for HITL resume within that call).
- No cross-session persistence. Defer SQLite/Postgres checkpointer (LangGraph supports drop-in).
- Each invocation gets a fresh `thread_id` (uuid4). HITL resume reuses the same `config`.

## 15. Dependencies (`pyproject.toml`)

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "langchain>=0.3",
    "langchain-openai>=0.2",
    "langgraph>=0.2",                 # for MemorySaver, Command
    "pydantic>=2",
    "pydantic-settings>=2",
    "markitdown[all]>=0.0.1",         # source ingestion
    "markdown-it-py>=3.0",            # wiki parsing
    "typer>=0.12",
    "python-dotenv>=1.0",
]
[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]
[project.scripts]
agentic-rag = "agentic_rag.cli:app"
```

## 16. Acceptance Criteria (whole project)

1. `pip install -e .` succeeds; `agentic-rag --help` lists ingest/query/lint/status/log.
2. `agentic-rag ingest raw/sample.md` against a temp copy of `wiki/` creates/updates ≥2 pages, updates `index.md`, appends `log.md` with the §10 prefix.
3. On a contradiction, `ingest` pauses with an inline prompt; `approve`/`reject` resume correctly and the decision is logged.
4. `delete_wiki_page` always pauses; rejecting keeps the file.
5. No tool ever writes outside `settings.wiki_path`; no tool touches `raw/` (enforced by guardrails middleware + verified by tests).
6. `agentic-rag query "What is MLX?"` returns a cited answer citing `[[MLX]]`; no wiki mutation.
7. `agentic-rag lint` writes `wiki/lint-report-YYYY-MM-DD.md` and does not delete anything by default.
8. All unit + integration tests pass with `FakeChatModel` (no network). Acceptance tests pass with real LLM (skipped without key).
9. Agents use `create_agent` + middleware (no raw `StateGraph`), per skills.

## 17. Risks / Open Items

- **R1** MarkItDown `[all]` is heavy (OCR/audio/transcription deps). If install bloats, switch to `markitdown[pdf,docx,pptx,xlsx]` and document. Decide at Phase 0.
- **R2** Fake-LLM harness fidelity: scripted tool calls may diverge from real model behavior. Mitigate with one real-LLM acceptance test.
- **R3** Contradiction detection is model-judgment, not deterministic. Acceptance test asserts *a* contradiction is flagged on a seeded conflicting source, not exact wording.
- **R4** `index.md` at scale: index-only retrieval degrades past a few hundred pages. Deferred vector store (`retrieval_mode="hybrid"`, `vector_db_path`) is the documented future path; scaffold the config field only.
- **R5** MCP server future: expose `ingest`/`query`/`lint` as MCP tools so Obsidian/Claude can drive the wiki. Out of MVP.

## 18. Future Work (scaffolded, not implemented)

- Vector store (Chroma) over wiki pages; hybrid index + vector search; `search_wiki` tool.
- Filing query answers back as wiki pages (`file_answer` tool on Query agent, HITL).
- Durable checkpointer (SQLite/Postgres) for resumable ingests.
- MCP server packaging.
- Web/URL ingestion (markitdown already supports YouTube/Wikipedia/RSS).
- Marp/HTML export; image handling.

## 19. Subagent Orchestration Summary

Pattern: **review-loop per phase** (pi-subagents skill). Parent owns the loop; one async `worker` implements; fresh-context `reviewer`s (distinct angles) inspect the real diff each round; parent synthesizes accepted fixes; forked async `worker` applies; stop at clean review or 3-round cap. Use `oracle` for architectural disputes, `scout` for cheap recon, `context-builder` for heavy handoffs. Per phase: optional scout → worker → reviewer fanout (review angles generated from the phase scope) → fix-worker → re-review. Never let reviewers edit; never run >1 writer on the active worktree (.pi worktree isolation only if true parallel writes needed — not the case here).