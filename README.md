# Agentic RAG

**A self-maintaining wiki with four focused LangChain agents.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-2b6cb0) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Tests: 516 passing](https://img.shields.io/badge/tests-516%20passing-success) ![Engine: 0 LLM calls](https://img.shields.io/badge/engine-0%20LLM%20calls-6b46c1)

Not classic RAG. Raw documents are compiled **once** into a living, interlinked Markdown wiki — then four focused LangChain agents (`ingest`, `query`, `lint`, `fix`) keep it current, answer from it with **enforced citations**, audit its health, and repair it.

A deterministic **0-LLM wiki engine** does the heavy lifting (search, page matching, indexing, health checks — all pure Python). Agents explore through **one read-only command tool** (`wiki_command`), and humans approve the destructive bits (page deletion, contradiction resolution) via LangGraph human-in-the-loop interrupts.

> **Highlights**
>
> - **Compile once, query often** — documents land in `raw/`; the ingest agent extracts entities, concepts and contradictions into curated, `[[cross-linked]]` wiki pages with a derived `index.md` and a `log.md` changelog. Queries navigate that curated knowledge instead of re-searching chunks every time.
> - **One command surface, typed mutations** — all four agents explore through a single pinned read-only tool (`wiki_command`: `scan` / `search` / `read` / `links` / `match` / `health`, joinable with `&&`). Read-only _by construction_: there is no shell to inject into. Mutations only happen through validated, path-guarded, HITL-gated write tools.
> - **Cite-or-die grounding** — the query agent has no finalize tool. Citations are extracted from its own final message, and any citation to a page it never actually navigated this turn is **dropped**. Answers are grounded by construction; confidence is inferred (`high` / `medium` / `low`).
> - **Deterministic core, LLM judgment at the edges** — search (BM25), matching, indexing and health checks run with **zero LLM calls**: free, fast, fully unit-testable. The LLM only adds judgment: extraction, semantic duplicates, final answers.
> - **Humans approve the damage** — page deletion and contradiction flags pause for `approve` / `edit` / `reject` in both the CLI and the Streamlit UI, via LangGraph interrupts resumed with `Command(resume=…)`.
> - **Tested at three levels, fast by default** — 522 collected cases: unit, scripted fake-model agent flows, and real-LLM answer quality judged with DeepEval. The 6 real-LLM tiers are opt-in (`pytest -m requires_llm`); a plain `pytest` runs all **516 headless tests in ~4 seconds**.

---

## Why not classic RAG?

| Classic RAG pain point                                                                     | How this project addresses it                                                                        |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Repeated work** — every query re-embeds, re-retrieves and re-synthesizes the same chunks | Knowledge is compiled once and kept current; queries become cheap navigation                         |
| **No cross-referencing** — chunks are flat, entities never learn about each other          | Wiki pages link each other (`[[Page]]`); the knowledge graph grows and is auditable                  |
| **Unverified citations** — answers just "sound right"                                      | Cite-or-die: every citation is checked against pages the agent actually visited this turn            |
| **Stale or broken knowledge**                                                              | `lint` + `fix` give the wiki a self-healing loop: deterministic audit, then repair with pinned tools |

## Architecture

Three layers:

1. **Raw sources** (`raw/`, immutable) — agents read, never write.
2. **Wiki** (`wiki/`, LLM-owned) — entity/concept/source pages, `index.md`, `log.md`.
3. **Schema** (`AGENTS.md`) — conventions injected into every agent system prompt (page types, naming, frontmatter, update rules).

_Architecture rule (from `docs/architecture.md`): layers depend only downward — `tools/` → `wiki/` + `io/` + `schemas/`; `agents/` → `tools/` + `schemas/`; `schemas/` → pydantic only. Never the reverse._

### Agent Framework

Every agent is built by `build_agent()` (`agents/factory.py`) using **LangChain `create_agent()`** with a middleware pipeline, a `MemorySaver` checkpointer, and a per-agent token tracker (no module globals):

- `audit_logging_middleware` — audit trail of every tool call
- `path_guard_middleware` — blocks any write into `raw/`, absolute paths, or `..` (see [Guardrails](#guardrails))
- `token_capture_middleware` — token usage bound to that agent's tracker
- agent-specific `HumanInTheLoopMiddleware` — approval workflows (delete pages, contradictions)

Agent loop recursion limits: **30** for query/lint/fix, **200** for ingest (multi-page ingestion needs many super-steps).

### The Four Agents

| Agent      | Job                                                                       | Tool surface                                                                                                                                                     | HITL                                                            |
| ---------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Query**  | Answer questions against the wiki with grounded citations                 | 1 — `wiki_command` (read-only by construction; no finalize tool — the answer is auto-built)                                                                      | none                                                            |
| **Ingest** | Read sources, extract structure, create/update pages, flag contradictions | 9 — `wiki_command`, `read_source`, `submit_extraction`, `create_page`, `update_page`, `flag_contradiction`, `regenerate_index`, `append_log`, `delete_wiki_page` | delete: approve/reject · contradiction: approve/**edit**/reject |
| **Lint**   | Audit wiki health (deterministic + semantic) and write a report           | 2 — `wiki_command` (health/scan/links), `write_lint_report`                                                                                                      | delete only                                                     |
| **Fix**    | Consume the lint report and apply pinned fixes — **no shell tool**        | 7 — `wiki_command` (reads), `edit_wiki_page`, `add_frontmatter`, `fix_link`, `append_related_section`, `regenerate_index`, `delete_wiki_page`                    | delete only                                                     |

**Fix kind → tool map**: `missing-frontmatter` → `add_frontmatter` · `broken-link` → `fix_link` · `missing-related` → `append_related_section` · `missing-index` → `regenerate_index` · `orphan` / `empty` / `stale` → report only (human/semantic decision).

## The Wiki Engine (0 LLM calls)

Everything below the agents is deterministic Python — unit-testable with no network and no model key. The `wiki_command` tool is a thin parser over exactly these functions:

| Module                  | Role                                                                                                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wiki/model.py`         | `Wiki` / `Page` / `Section` models; the filesystem + frontmatter is the source of truth; synthesizes frontmatter when missing                               |
| `wiki/search.py`        | BM25 search (k=8) over title/tags/headings/section text, type filters, bounded link expansion                                                               |
| `wiki/match.py`         | `match_page()` decision tree → `exact \| similar \| conflict \| none` — pure Python, no thresholds                                                          |
| `wiki/health.py`        | `health_check()` → 7 issue kinds (`orphan`, `missing-index`, `broken-link`, `missing-frontmatter`, `missing-related`, `empty`, `stale`) with a severity map |
| `wiki/dedupe_index.py`  | `regenerate_index()` — `index.md` is a derived view, rebuilt atomically from the model                                                                      |
| `io/wiki_io.py`         | Atomic read/write/delete of page files (temp file + rename — no corruption on crash)                                                                        |
| `io/markdown_parser.py` | Parses `[[links]]`, headings, YAML frontmatter; slugify                                                                                                     |
| `io/source_loader.py`   | MarkItDown wrapper — raw sources (pdf, docx, html, …) → markdown                                                                                            |
| `schemas/*.py`          | Pydantic contracts: wiki, extraction, query, lint, agents_md                                                                                                |

### `wiki_command` — the one read-only navigation tool

```text
wiki_command("scan")                                              # one-call overview of every content page
wiki_command('search "gpu" --k 8 && read entities/mlx')           # compound: search + read in one call
wiki_command('match "MLX" --type entity')                         # create vs update vs conflict
wiki_command("links --slug entities/mlx && health")               # link graph + structural audit
```

| Sub-command                                        | What it does                                                                                 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `scan [--max-chars N]`                             | One-call overview of every content page (slug, type, title, preview, link counts, date)      |
| `search "<query>" [--k N] [--type T] [--tags a,b]` | BM25-ranked pages + bounded linked pages; records every hit as _navigated_ (for cite-or-die) |
| `read <slug> [--section "Heading"]`                | Full page markdown, or a single section                                                      |
| `links [--slug S]`                                 | Inbound/outbound link summary — whole wiki or one page                                       |
| `match "<name>" --type <type>`                     | Deterministic `exact \| similar \| conflict \| none` decision                                |
| `health`                                           | Deterministic structural audit — 7 issue kinds, 0 LLM calls                                  |

**Read-only by construction** — there is no shell, no subprocess, no redirection; nothing to deny-list, jail, or patch. A test hammers it with `read ../../etc/passwd` and `search "rm -rf wiki"` and asserts the wiki is byte-identical afterwards.

## Guardrails

`path_guard_middleware` intercepts every write tool (`create_page`, `update_page`, `delete_wiki_page`, `edit_wiki_page`, `add_frontmatter`, `fix_link`, `append_related_section`, `write_lint_report`) and rejects any argument containing `raw/`, an absolute path, or `..`. `read_source` is exempt because it legitimately reads from `raw/`.

Edge cases are covered by tests, not hope:

- **Consistent slug resolution** — `mlx` and `entities/mlx` behave identically across every edit tool (a real bug fixed: `add_frontmatter("mlx")` used to read one file and write another).
- **No dangling links** — `fix_link` replaces all occurrences (plain + aliased) and refuses to create a new broken link.
- **Idempotent sections** — `append_related_section` never adds duplicate bullets.
- **Frontmatter-safe edits** — `edit_wiki_page` rejects no-op edits and any edit that would corrupt YAML frontmatter (schema-level changes must go through the dedicated tools).

Human-in-the-loop: page deletion always pauses for approval; ingest's contradiction flags support `approve` / `edit` / `reject`. The CLI and Streamlit UI drive the exact same decision shapes.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (fast Python package manager)
- An OpenAI-compatible API key (OpenAI, Azure, OpenRouter, local proxy)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd langchain-rag

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install runtime + dev tooling (uv sync installs the dev group by default)
uv sync

# Verify installation
agentic-rag --help
```

### Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API key and settings
# Required: OPENAI_API_KEY
```

**Environment Variables:**

| Variable           | Default                     | Description                                                            |
| ------------------ | --------------------------- | ---------------------------------------------------------------------- |
| `OPENAI_API_KEY`   | (required)                  | API key (OpenAI-compatible: OpenAI, Azure, OpenRouter, local proxy, …) |
| `OPENAI_BASE_URL`  | `https://api.openai.com/v1` | API endpoint override                                                  |
| `OPENAI_MODEL`     | `gpt-4.1-mini`              | Model name                                                             |
| `WIKI_PATH`        | `./wiki`                    | Wiki directory                                                         |
| `RAW_SOURCES_PATH` | `./raw`                     | Raw sources directory                                                  |
| `AGENTS_MD_PATH`   | `./AGENTS.md`               | Schema file path                                                       |
| `RECURSION_LIMIT`  | `30`                        | Max agent loop turns (query/lint/fix)                                  |
| `LOG_LEVEL`        | `INFO`                      | Logging level                                                          |
| `LOG_DIR`          | (none)                      | Log directory (`None` = console only)                                  |

> 💡 **Reasoning models work out of the box** — the model factory includes a slim `ReasoningPassthroughChat` that preserves `reasoning_content` across turns, so DeepSeek-style thinking-mode models served over the OpenAI-compatible API work without extra configuration.

## CLI Usage

### Ingest a Source

```bash
# Ingest a markdown file
agentic-rag ingest raw/sample.md

# Ingest any supported format (pdf, docx, pptx, xlsx, html, etc.)
agentic-rag ingest path/to/document.pdf

# Or pass natural language directly
agentic-rag ingest "Add a page about my new project"
```

The agent will:

1. **Read the source** — convert to markdown with MarkItDown (`read_source`).
2. **Submit structured extraction** — pure, deterministic JSON of entities, concepts, and contradictions (`submit_extraction`). No writes happen here.
3. **Match each name** — `wiki_command("match \"<name>\" --type <type>")` decides deterministically between `exact`/`similar` (→ `update_page`), `none` (→ `create_page`), or `conflict` (→ `flag_contradiction`, human approval).
4. **Finish the writes** — update `## Related` links on touched pages, write a source summary under `sources/<slug>.md`, rebuild the derived index (`regenerate_index`), and append the log entry (`append_log`).

The ingest agent never calls an "update index" tool directly — `index.md` is a derived view, rebuilt atomically from the wiki model.

### Query the Wiki

```bash
# Ask a question
agentic-rag query "What is MLX?"
```

The agent will:

1. **Search + navigate** — one read-only `wiki_command` call (pinned grammar: `search "<q>"`, `read <slug>`, `scan`, `links`, `health`, `match`, joinable with `&&`) — BM25 search with bounded link expansion, section-scoped reads, and link-graph context, all deterministic.
2. **Finalize a grounded answer** — finalization is automatic: there is no finalization tool. The model's final message is synthesized into a `QueryAnswer` with `[[Page]]` links extracted as citations, and `validate_citations` (NavCapture) drops any citation whose slug was never navigated — **cite-or-die**.
3. **Render** — the CLI prints Answer, Confidence, Citations, and Suggestion.

### Health Check

```bash
# Run lint agent
agentic-rag lint
```

The agent will:

1. **Run the deterministic health check** — `wiki_command("health")` (0 LLM calls) audits the wiki for 7 issue kinds: `orphan`, `missing-index`, `broken-link`, `missing-frontmatter`, `missing-related`, `empty`, `stale`. Severity map: `missing-frontmatter` = critical; `orphan`/`missing-index`/`broken-link`/`empty` = high; `missing-related`/`stale` = medium. `lint-report-*` pages are excluded from the audit.
2. **Apply semantic judgment** — the lint agent checks for duplicate coverage (the part only an LLM can judge).
3. **Write the report** — `write_lint_report` renders a structured `LintReport` to `wiki/lint-report-YYYY-MM-DD.md`.

### Fix Lint Issues

```bash
# Run health check and fix everything it finds
agentic-rag fix latest

# Filter by issue kind or page slug
agentic-rag fix missing-frontmatter
agentic-rag fix concepts/machine-learning
```

`fix` runs the same deterministic `health_check`, turns each issue into a one-line message, and hands it to the Fix Agent, whose fixes are pinned to a kind→tool map (see [The Four Agents](#the-four-agents)). **HITL only on `delete_wiki_page`** — everything else is auto-approved. There is **no shell tool** in the Fix Agent.

### View Status and Logs

```bash
# Show wiki statistics
agentic-rag status

# View log entries
agentic-rag log

# Tail last 5 entries
agentic-rag log --tail 5
```

## Web UI (Streamlit)

A multi-page Streamlit frontend driving all four agents — query, ingest, lint, fix — **in-process** (no HTTP server, no duplicated logic). Real token streaming with live tool-call chips, multi-turn memory, and structured answer + citations rendering for queries. The ingest/lint/fix pages add full human-in-the-loop approval (approve / reject, plus edit-resolution for ingest's contradiction requests). Chat transcripts are durable per agent and thread (stored under `frontend/history/`, sidebar thread selector + "New chat"). The ingest page also offers a picker of files already present under `raw/`.

```bash
uv sync
uv run streamlit run frontend/app.py
```

_Architecture: the Streamlit shell is thin — `query_driver.py`, `agent_driver.py`, and `history_store.py` are pure modules covered by unit tests, and the UI itself is verified with AppTest smoke tests._

## Evaluation

| Level                | Count | What it verifies                                                                                                                                                                                         |
| -------------------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/`        | 311   | Fast, isolated, no network: wiki I/O, index codec, markdown parser, path-guard matrix, the `wiki_command` dispatcher (grammar + read-only-by-construction proof), grounding, tools, HITL decision shapes |
| `tests/integration/` | 28    | Scripted fake-model agent flows (`FakeChatModel`): ingest, fix, grounded query, CLI                                                                                                                      |
| `tests/levels/`      | 183   | Layered behavior: level1 wiki-schema/path-guard/link integrity · level2 tool selection, turn efficiency, DAG trajectory · level3 recall@8, calibrated judges, contradictions                             |

**6 of these are real-LLM and opt-in.** The `requires_llm` marker is deselected by default (`addopts = -m 'not requires_llm'` in `pyproject.toml`), so a plain `pytest` finishes all 516 headless tests in ~4 seconds offline. Run the real-LLM tiers explicitly with `pytest -m requires_llm` — without an `OPENAI_API_KEY` they skip with a clear reason.

The judges run via LiteLLM (temperature 0) against the configured OpenAI-compatible endpoint. DeepEval metrics (`Faithfulness`, `AnswerRelevancy`, `ContextualRecall`) score `LLMTestCase`s where `retrieval_context` = the pages NavCapture shows the agent actually navigated. Baselines from the level-3 gate run: **faithfulness 1.00 · relevancy 0.85 · context recall 1.00**, with hard floors **0.80 / 0.70 / 0.80**.

```bash
uv run pytest                       # 516 headless tests, ~4s
uv run pytest -m requires_llm       # +6 real-LLM tests (needs a key; skips without)
```

## Development

### Project Structure

```
agentic_rag/
├── src/agentic_rag/        # Main package
│   ├── config.py           # Settings (pydantic-settings, .env)
│   ├── cli.py              # Typer CLI (ingest/query/lint/fix/status/log) — the entry point
│   ├── logging_config.py   # Colored console/file logging
│   ├── token_tracker.py    # Token usage tracking (attached to each agent)
│   ├── agents/             # LLM agent builders (one per agent)
│   │   ├── factory.py      # build_agent: create_agent + middleware pipeline + tracker
│   │   ├── llm.py          # get_model — ChatOpenAI factory + reasoning_content passthrough
│   │   ├── prompts.py      # System prompt builders (inject AGENTS.md + wiki index)
│   │   ├── ingest.py / query.py / lint.py / fix.py
│   ├── tools/              # LangChain @tool layer — the agents' action surface
│   │   ├── nav.py          # wiki_command — the pinned read-only command dispatcher (scan/search/read/links/match/health)
│   │   ├── grounding.py    # cite-or-die finalization (NavCapture, build_final_answer, validate_citations)
│   │   ├── extraction.py   # submit_extraction — structured extraction boundary (ingest)
│   │   ├── ingest_tools.py / lint_tools.py / fix_tools.py
│   ├── wiki/               # Deterministic wiki engine (0 LLM calls)
│   │   ├── model.py / search.py (BM25) / match.py / health.py / dedupe_index.py
│   ├── io/                 # Filesystem adapters (atomic writes, index, log, markdown parser, source loader)
│   ├── schemas/            # Pydantic contracts (wire + structured-output models)
│   └── middleware/         # audit logging + token capture + path guard (registered in factory)
├── frontend/               # Streamlit UI (same agents, in-process)
│   ├── app.py / builders.py / query_driver.py / agent_driver.py / ui_common.py / history_store.py
│   └── app_pages/          # query.py, ingest.py, lint.py, fix.py — one page per agent
├── raw/                    # Raw sources (immutable — agents read, never write; gitignored)
├── wiki/                   # LLM-owned wiki (runtime data; gitignored)
├── tests/                  # unit, integration, levels, fixtures
├── docs/                   # architecture.md + HTML deep dives
├── scripts/knowledge_graph.py  # → knowledge_graph.html (interactive vis.js graph of the wiki)
└── AGENTS.md               # Wiki schema conventions (injected into every agent prompt)
```

> **Note on tracked content:** `wiki/` (LLM-generated runtime data), `raw/` (your private source documents) and `frontend/history/` (chat transcripts) are gitignored on purpose. Static previews of the wiki, architecture and interview notes live in `docs/*.html`.

### Running Tests

```bash
# Run all tests (offline, fast — real-LLM tiers are deselected by default)
uv run pytest

# Run specific test types
uv run pytest tests/unit/          # Unit tests (no network)
uv run pytest tests/integration/   # Scripted fake-model agent flows
uv run pytest tests/levels/level1 -q   # Level 1: wiki schema, path guard, link integrity
uv run pytest tests/levels/level2 -q   # Level 2: tool selection, schemas, efficiency, state + DAG trajectory
uv run pytest tests/levels/level3 -q   # Level 3: recall@8 + calibrated faithfulness/relevancy judges, contradictions
uv run pytest tests/levels/ -q    # Full levels suite (deterministic tiers)

# Run ONLY the real-LLM tiers (live agents + DeepEval judges — slow, costs tokens)
uv run pytest -m requires_llm

# Run with coverage
uv run pytest --cov=agentic_rag --cov-report=html
```

### Adding a New Tool

1. Create the tool function with the `@tool` decorator in `src/agentic_rag/tools/`.
2. Add type hints and a docstring with an `Args:` section.
3. Register the tool in the relevant agent builder (`ingest.py`, `query.py`, `lint.py`, or `fix.py`).
4. **If the tool writes to the wiki**, add its name to the `write_tools` set in `src/agentic_rag/middleware/guardrails.py` so the path guard covers it.
5. Add unit tests in `tests/unit/test_tools.py`.

### Adding Middleware

1. Create a middleware function with `@wrap_tool_call` or `@before_model` decorator.
2. Add it to the agent's middleware list in `src/agentic_rag/agents/factory.py` (for all agents) or in the individual agent builder (for one agent).
3. Middleware patterns:
   - `wrap_tool_call`: intercept tool execution (logging, guards, retry)
   - `before_model` / `after_model`: inspect/modify state around model calls
   - `HumanInTheLoopMiddleware`: pause for human approval on dangerous tools

## Documentation

| Doc                                                              | Covers                                                                                                                                 |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| [docs/documentation.html](docs/documentation.html)               | The Agent Layer in depth: `create_agent` + middleware, per-agent tool inventories, HITL mechanics, CLI reference, testing architecture |
| [docs/wiki-engine.html](docs/wiki-engine.html)                   | The 0-LLM wiki engine: filesystem-as-truth, BM25 + bounded link expansion, the 7 health issue kinds, atomic writes                     |
| [docs/interview-prep-notes.html](docs/interview-prep-notes.html) | Simple-English overview: 60-second pitch, key ideas, talking points                                                                    |
| [docs/architecture.md](docs/architecture.md)                     | Layer-by-layer code map and import rules                                                                                               |
| `scripts/knowledge_graph.py`                                     | Generates `knowledge_graph.html` — an interactive vis.js graph of the wiki                                                             |

## FAQ

**Why a wiki instead of a vector store?**
The wiki is a compounding artifact: entities and concepts are compiled once, cross-linked, and kept current by the ingest agent. Queries navigate curated knowledge instead of re-retrieving chunks — and the index, links and changelog make the knowledge graph visible and auditable.

**Why a pinned command tool instead of real bash?**
A real shell turns every read into a write risk, and deny-lists lose against prompt injection (a malicious source file can smuggle `$(rm …)` into a command the model composes). The pinned grammar gives the same "one call, several operations" ergonomics with no interpreter to attack — read-only by construction, deterministic, offline-testable. This is _capability over filtering_: danger is physically impossible, not merely discouraged.

**Where do answers' citations come from?**
From the model's own final message: inline `[[Page]]` links are extracted as citations and validated against the NavCapture set (pages actually navigated this turn via `wiki_command` search/read). Anything else is dropped — cite-or-die.

**Can the fix agent run arbitrary commands?**
No — by design. It has no shell, its only read tool is the read-only `wiki_command`, and changes go through guarded, typed edit tools with a pinned kind→tool map and HITL on deletion.

**Do the real-LLM tests run by default?**
No. They are marked `requires_llm` and deselected by default (`addopts`), so `pytest` runs 516 headless tests in ~4 seconds. Run them explicitly with `pytest -m requires_llm`.

**Is this an "AI agent" that could do anything?**
No — it's a scoped tool-user. Reads happen through one read-only command surface, every mutation is a path-guarded typed tool, the loop is bounded by a recursion limit, and destructive actions pause for human approval.

## Tech Stack

- **Orchestration**: LangChain `create_agent()`, LangGraph (checkpointing, interrupts, `Command(resume=…)`)
- **LLM access**: `langchain-openai` + a slim `reasoning_content` passthrough for thinking-mode models
- **Search**: BM25 (`rank-bm25`) over titles/tags/headings — no embeddings, no vector store
- **Conversion**: MarkItDown (pdf, docx, pptx, xlsx, html, …)
- **Contracts**: Pydantic / pydantic-settings
- **CLI**: Typer
- **UI**: Streamlit
- **Evaluation**: DeepEval + LiteLLM (opt-in real-LLM tiers)

## License

[MIT](LICENSE) © Alvaro Jimenez
