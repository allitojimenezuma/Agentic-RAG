# Agentic RAG

An agentic RAG system that maintains a persistent, interlinked markdown wiki from raw sources. Unlike classic RAG (re-derive answers from chunks on every query), the wiki is a *compounding artifact*: entities, concepts, cross-references, and contradictions are compiled **once** and kept current. Query-time work becomes navigation + synthesis over a curated knowledge base instead of repeated chunk retrieval.

## Architecture

Three layers:

1. **Raw sources** (`raw/`, immutable) — agents read, never write.
2. **Wiki** (`wiki/`, LLM-owned) — entity/concept/source pages, `index.md`, `log.md`. Runtime data (gitignored).
3. **Schema** (`AGENTS.md`) — conventions injected into every agent system prompt.

### Agent Framework

Agents are built with **LangChain `create_agent()`** + a middleware pipeline. Every agent runs through:

- `audit_logging_middleware` — audit trail of tool calls
- `path_guard_middleware` — blocks `raw/`, absolute, and `..` paths on write tools (see [Guardrails](#guardrails))
- `token_capture_middleware` — token usage tracking
- agent-specific `HumanInTheLoopMiddleware` for approval workflows (delete pages, contradictions)

`MemorySaver` checkpointer provides per-invocation state (HITL resume within a call). Agent loop recursion limit defaults to **30** (see [Configuration](#configuration)).

### Agents

Four agents, each a focused LangChain agent:

- **Ingest Agent** — reads sources (MarkItDown), extracts structured entities/concepts/contradictions, creates/updates wiki pages via a deterministic matcher, and flags contradictions for human approval.
- **Query Agent** — answers questions against the wiki with a cite-or-die grounding gate (read-only).
- **Lint Agent** — audits wiki health with a deterministic 0-LLM health check plus semantic judgment (duplicate coverage), writes a report.
- **Fix Agent** — consumes the structured lint report and applies fixes via a pinned kind→tool map; HITL only on page deletion.

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

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
uv pip install -e ".[dev]"

# Verify installation
agentic-rag --help
```

Or install directly with uv (no venv activation needed):

```bash
uv pip install -e ".[dev]"
```

### Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API key and settings
# Required: OPENAI_API_KEY
# Optional: OPENAI_BASE_URL, OPENAI_MODEL, WIKI_PATH, etc.
```

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model to use |
| `WIKI_PATH` | `./wiki` | Wiki directory |
| `RAW_SOURCES_PATH` | `./raw` | Raw sources directory |
| `AGENTS_MD_PATH` | `./AGENTS.md` | Schema file path |
| `RETRIEVAL_MODE` | `index` | Retrieval mode (MVP: index-only) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_DIR` | (none) | Log directory (`None` = console only) |
| `RECURSION_LIMIT` | `30` | Max agent loop turns |
| `HITL_ENABLED` | `true` | Enable human-in-the-loop |

### OpenCode Go prompt caching

When `OPENAI_BASE_URL` points at the OpenCode Go gateway
(`https://opencode.ai/zen/go/v1`), every request is automatically instrumented
with prompt-cache fields: a stable per-agent `prompt_cache_key`
(`wiki-query`, `wiki-lint`, `wiki-fix`, `wiki-ingest`),
`prompt_cache_retention: "24h"`, `cache_control` breakpoints on the system
prompt, the last two messages, and the last tool — plus an
`x-opencode-session` header with the same session id, which is how the
[OpenCode Zen](https://opencode.ai) usage dashboard groups requests into
sessions and how the gateway pins requests to the same upstream provider
(`x-session-affinity`), keeping the upstream prompt cache warm. Cache reads
are 5–120× cheaper than input tokens on opencode-go models; the gateway
default is only ~5 minutes with no session key. Verified live:
`cache_read` went from 0 → 384/503 tokens on the second identical call.

Set `OPENCODE_GO_CACHE=0` in the environment to disable the instrumentation.
Models whose downstream API rejects `cache_control` markers (glm/zhipu) are
skipped automatically.

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
4. **Update `## Related` links** on pages touched by the extraction.
5. **Write a source summary** page under `sources/<slug>.md`.
6. **Rebuild the derived index** (`regenerate_index`) and **append the log entry** (`append_log`).

The ingest agent never calls an "update index" tool directly — `index.md` is a derived view, rebuilt atomically from the wiki model.

### Query the Wiki

```bash
# Ask a question
agentic-rag query "What is MLX?"
```

The agent will:

1. **Search + navigate** — one read-only `wiki_command` call (pinned grammar: `search "<q>"`, `read <slug>`, `scan`, `links`, `health`, `match`, joinable with `&&`) — BM25 search with bounded link expansion, section-scoped reads, and link-graph context, all deterministic.
3. **Finalize a grounded answer** — finalization is automatic: there is no finalization tool. The model's final message is synthesized into a `QueryAnswer` with `[[Page]]` links extracted as citations, and `validate_citations` (NavCapture) **drops any citation whose slug was never navigated** — cite-or-die.
4. **Render** — the CLI prints Answer, Confidence, Citations, and Suggestion.

### Health Check

```bash
# Run lint agent
agentic-rag lint
```

The agent will:

1. **Run the deterministic health check** — `wiki_command("health")` (0 LLM calls) audits the wiki for 7 issue kinds: `orphan`, `missing-index`, `broken-link`, `missing-frontmatter`, `missing-related`, `empty`, `stale`. Severity map: `missing-frontmatter` = critical; `orphan`/`missing-index`/`broken-link`/`empty` = high; `missing-related`/`stale` = medium. `lint-report-*` pages are excluded from the audit.
2. **Apply semantic judgment** — the lint agent checks for duplicate coverage (the part only an LLM can judge).
3. **Write the report** — `write_lint_report` renders a structured `LintReport` (or a plain string, back-compat) to `wiki/lint-report-YYYY-MM-DD.md`.

### Fix Lint Issues

```bash
# Run health check and fix everything it finds
agentic-rag fix latest

# Filter by issue kind or page slug
agentic-rag fix missing-frontmatter
agentic-rag fix concepts/machine-learning
```

`fix` runs the same deterministic `health_check`, turns each issue into a one-line user message (`[kind] slug: detail`), and hands it to the Fix Agent. Fixes are pinned to a kind→tool map:

| Issue kind | Tool |
|------------|------|
| `missing-frontmatter` | `add_frontmatter` |
| `broken-link` | `fix_link` |
| `missing-related` | `append_related_section` |
| `missing-index` | `regenerate_index` |
| `orphan` / `empty` / `stale` | report only (need human/semantic decision) |

**HITL only on `delete_wiki_page`** — everything else is auto-approved. There is **no shell tool** in the Fix Agent.

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

A multi-page Streamlit frontend driving all four agents — query, ingest, lint, fix — in-process
(no HTTP server). Real token streaming with live tool-call chips, multi-turn memory, and
structured answer + citations render for queries. The ingest/lint/fix pages add full
human-in-the-loop approval (approve / reject, plus edit-resolution for ingest's contradiction
requests). Chat transcripts are durable per agent and thread (stored under `frontend/history/`,
sidebar thread selector + "New chat"). The ingest page also offers a picker of files already
present under `raw/` — the UI only reads `raw/`; drop source documents there yourself first.

```bash
uv sync
uv run streamlit run frontend/app.py
```

## Development

### Running Tests

```bash
# Run all tests (offline, fast — real-LLM tiers are deselected by default)
uv run pytest

# Run with verbose output
uv run pytest -v

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

The six real-LLM tests (level2 trajectory acceptance + level3 judge calibration) are
**opt-in**: they are deselected by the default ``addopts -m 'not requires_llm'`` so a plain
``pytest`` finishes in seconds. Run them with ``pytest -m requires_llm``; without an
``OPENAI_API_KEY`` they skip (``tests/levels/conftest.py`` loads the repo ``.env`` into the
environment, so with your key in ``.env`` they actually run).

Current state (headless): **512 passed** (unit 307, integration 28, levels 177) in ~5s; **6
real-LLM tests opt-in** via ``pytest -m requires_llm``.

### Test Structure

```
tests/
├── unit/                  # Fast, isolated tests (no network)
│   ├── test_wiki_io.py
│   ├── test_index.py
│   ├── test_markdown_parser.py
│   ├── test_eval_hitl.py   # Headless HITL helpers: approve/reject/edit decision shapes
│   └── ...
├── integration/           # Scripted fake-model agent flows
│   ├── test_ingest_scripted.py
│   ├── test_fix_scripted.py
│   ├── test_query_grounded.py
│   └── test_cli.py
├── levels/                # Layered agent-behavior suite (deterministic tiers; real-LLM tiers opt-in)
│   ├── level1/            # Wiki schema conformance, path-guard matrix, link integrity
│   ├── level2/            # Tool selection, argument schemas, turn efficiency, state consistency,
│   │                      #   DAG trajectory contract + real-LLM trajectory acceptance tier
│   ├── level3/            # Recall@8 on curated/hard queries, calibrated faithfulness/relevancy
│   │                      #   judges (real-LLM tier), contradiction handling
│   ├── conftest.py        # Loads .env into os.environ; defines the requires_llm opt-in decorator
│   └── test_corpus_selfcheck.py
└── fixtures/              # Test fixtures, FakeChatModel, DeepEval judge helpers
```
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

## Project Structure

```
agentic_rag/
├── src/agentic_rag/        # Main package
│   ├── config.py           # Settings (pydantic-settings, .env)
│   ├── cli.py              # Typer CLI (ingest/query/lint/fix/status/log) — the entry point
│   ├── logging_config.py   # Colored console/file logging
│   ├── token_tracker.py    # Token usage tracking (attached to each agent)
│   ├── agents/             # LLM agent builders (one per agent)
│   │   ├── factory.py      # build_agent: create_agent + middleware pipeline + tracker
│   │   ├── llm.py          # get_model — ChatOpenAI factory (base_url/api_key)
│   │   ├── prompts.py      # System prompt builders (inject AGENTS.md + wiki index)
│   │   ├── ingest.py       # build_ingest_agent()
│   │   ├── query.py        # build_query_agent()
│   │   ├── lint.py         # build_lint_agent()
│   │   └── fix.py          # build_fix_agent()
│   ├── tools/              # LangChain @tool layer — the agents' action surface
│   │   ├── shared.py       # init_shared_tools / get_wiki_path / get_index_summary
│   │   ├── nav.py          # wiki_command — the pinned read-only command dispatcher (scan/search/read/links/match/health) + regenerate_index
│   │   ├── grounding.py    # cite-or-die finalization (NavCapture, build_final_answer, validate_citations)
│   │   ├── extraction.py   # submit_extraction — structured extraction boundary (ingest)
│   │   ├── ingest_tools.py # read_source, create/update/delete page, append_log, flag_contradiction
│   │   ├── lint_tools.py   # write_lint_report (health_check runs via wiki_command)
│   │   └── fix_tools.py    # edit_wiki_page, add_frontmatter, fix_link, append_related_section
│   ├── wiki/               # Deterministic wiki engine (0 LLM calls)
│   │   ├── model.py        # Wiki/Page/Section models + load_wiki (synthesized frontmatter)
│   │   ├── search.py       # BM25 search + bounded link expansion
│   │   ├── match.py        # match_page decision tree (exact/similar/conflict/none)
│   │   ├── health.py       # health_check → LintReport (deterministic, 7 issue kinds)
│   │   └── dedupe_index.py # regenerate_index — index.md is a derived view, rebuilt atomically
│   ├── io/                 # Filesystem adapters (read/write the wiki + raw sources)
│   │   ├── wiki_io.py      # atomic read/write/delete/list of page files
│   │   ├── index.py        # index.md codec (parse + format + atomic write)
│   │   ├── log.py          # log.md codec (append + tail)
│   │   ├── markdown_parser.py  # [[links]], headings, YAML frontmatter, slugify
│   │   └── source_loader.py    # MarkItDown wrapper (raw sources → markdown)
│   ├── schemas/            # Pydantic contracts (wire + structured-output models)
│   │   ├── wiki.py         # Frontmatter, IndexEntry, Index, LogEntry, Heading, Link
│   │   ├── extraction.py   # Entity, Concept, Contradiction, ExtractionResult
│   │   ├── query.py        # QueryAnswer, SourceCitation
│   │   ├── lint.py         # LintReport, Issue
│   │   └── agents_md.py    # AGENTS.md loader (embedded default schema)
│   └── middleware/         # LangChain middleware pipeline (registered in factory)
│       ├── logging.py      # audit_logging + token_capture middleware
│       └── guardrails.py   # path_guard — blocks writes outside wiki/ and into raw/
├── frontend/               # Streamlit UI (same agents, in-process)
│   ├── app.py              # st.navigation entry point (Settings guard + page list)
│   ├── builders.py         # @st.cache_resource agent builders + per-agent config factory
│   ├── query_driver.py     # query streaming adapter → typed events (ToolStart/ToolEnd/AnswerToken/FinalAnswer)
│   ├── agent_driver.py     # generic HITL streaming driver for ingest/lint/fix
│   ├── ui_common.py        # shared page shell: session state, sidebar, HITL widgets, run_turn
│   ├── history_store.py    # durable JSONL chat transcripts (frontend/history/)
│   └── app_pages/          # query.py, ingest.py, lint.py, fix.py — one page per agent
├── raw/                    # Raw sources (immutable — agents read, never write)
├── wiki/                   # LLM-owned wiki (runtime data; gitignored)
├── tests/                  # unit, integration, levels, eval, acceptance, fixtures
├── docs/                   # architecture.md (this map), cleanup-plan.md, HTML doc exports
├── archive/                # Superseded planning docs (PLAN/IDEA/spec)
└── AGENTS.md               # Wiki schema conventions (injected into every agent prompt)
```

## Key Concepts

### Wiki Schema (AGENTS.md)

The `AGENTS.md` file defines conventions injected into agent prompts:
- **Page types**: entity, concept, source, comparison, overview
- **Naming**: `entities/<slug>.md`, `concepts/<slug>.md`, etc.
- **Cross-references**: Obsidian-style `[[Page Name]]` links
- **Frontmatter**: YAML with slug, type, title, sources, updated, tags
- **Update rules**: New info supersedes old; flag contradictions; always update index + log

### Human-in-the-Loop (HITL)

Certain operations require human approval:
- **Delete wiki page**: Always pauses; approve to delete, reject to keep (all four agents).
- **Flag contradiction**: When a new source conflicts with an existing page; approve/edit/reject (ingest agent).

### Cite-or-die Grounding (Query)

The query agent has **no finalization tool**. `build_final_answer` auto-builds the `QueryAnswer` from the model's final message: every inline `[[Page]]` link becomes a citation, validated against the turn's `NavCapture` — a per-invocation set of slugs the agent actually navigated via `wiki_command` search/read. Citations for pages never visited are dropped (cite-or-die), and confidence is inferred: `high` when navigated pages are cited, `medium` when pages were navigated but none are cited, `low` when nothing was navigated. The same finalizer drives the CLI and the Streamlit UI, so both render identically.

### Atomic Writes

All wiki writes are atomic (temp file + rename) to prevent corruption on crashes. `regenerate_index` rebuilds `index.md` atomically from the wiki model — the index is a *derived view*, never hand-edited.

### Guardrails

`path_guard_middleware` intercepts every write tool (`create_page`, `update_page`, `delete_wiki_page`, `edit_wiki_page`, `fix_link`, `append_related_section`, …) and rejects any argument containing `raw/`, an absolute path, or `..`. `read_source` is exempt because it legitimately reads from `raw/`.

## License

[Add your license here]
