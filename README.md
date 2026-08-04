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
3. **Match each name** — `match_page_tool` decides deterministically between `exact`/`similar` (→ `update_page`), `none` (→ `create_page`), or `conflict` (→ `flag_contradiction`, human approval).
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

1. **Search** — `wiki_search` (BM25, k=8, with bounded link expansion).
2. **Navigate** — `wiki_read_page` on the hits, following `[[cross-references]]` as needed (`wiki_summary` for overviews).
3. **Submit a grounded answer** — `submit_query_answer` with `QueryAnswer(Answer, Citations, Confidence, Suggestion)`. `validate_citations` (NavCapture) **drops any citation whose slug was never navigated** — cite-or-die.
4. **Render** — the CLI prints Answer, Confidence, Citations, and Suggestion.

### Health Check

```bash
# Run lint agent
agentic-rag lint
```

The agent will:

1. **Run the deterministic health check** — `run_health_check` (0 LLM calls) audits the wiki for 7 issue kinds: `orphan`, `missing-index`, `broken-link`, `missing-frontmatter`, `missing-related`, `empty`, `stale`. Severity map: `missing-frontmatter` = critical; `orphan`/`missing-index`/`broken-link`/`empty` = high; `missing-related`/`stale` = medium. `lint-report-*` pages are excluded from the audit.
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

A minimal Streamlit chat frontend for the query agent (real token streaming, live tool-call
chips, multi-turn memory, structured answer + citations render). It runs the agent in-process
— no HTTP server.

```bash
uv sync --extra ui
uv run streamlit run frontend/app.py
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test types
uv run pytest tests/unit/          # Unit tests (no network)
uv run pytest tests/integration/   # Scripted fake-model agent flows
uv run pytest tests/eval/          # Eval: recall@8 + hallucination gate (no LLM)
uv run pytest tests/acceptance/    # Acceptance tests (real LLM, requires OPENAI_API_KEY)

# Run with coverage
uv run pytest --cov=agentic_rag --cov-report=html
```

Current state: **209 passed / 2 skipped** (the 2 skipped are acceptance tests needing a live LLM at `localhost`).

### Test Structure

```
tests/
├── unit/                  # Fast, isolated tests (no network)
│   ├── test_wiki_io.py
│   ├── test_index_manager.py
│   ├── test_markdown_parser.py
│   └── ...
├── integration/           # Scripted fake-model agent flows
│   ├── test_ingest_scripted.py
│   ├── test_fix_scripted.py
│   ├── test_query_grounded.py
│   └── test_cli.py
├── eval/                  # recall@8 = 1.00 on 15 curated live-wiki queries;
│                         # hallucination gate on validate_citations (no LLM)
├── acceptance/            # Real LLM tests (requires API key; skipped by default)
│   ├── test_wiki_health.py
│   └── test_ingest_real_source.py
└── fixtures/              # Test fixtures and FakeChatModel
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
│   ├── config.py           # Settings (pydantic-settings)
│   ├── paths.py            # Path helpers
│   ├── cli.py              # Typer CLI (ingest/query/lint/fix/status/log)
│   ├── main.py             # Entry point
│   ├── logging_config.py   # Logging setup
│   ├── token_tracker.py    # Token usage tracking
│   ├── agents/             # Agent builders
│   │   ├── factory.py      # create_agent + middleware pipeline
│   │   ├── ingest.py       # build_ingest_agent()
│   │   ├── query.py        # build_query_agent()
│   │   ├── lint.py         # build_lint_agent()
│   │   ├── fix.py          # build_fix_agent()
│   │   ├── model.py        # Model factory
│   │   └── prompts.py      # System prompt builders
│   ├── tools/              # LangChain tools
│   │   ├── shared.py       # Shared init + common tools
│   │   ├── nav.py          # wiki_search, wiki_read_page, wiki_summary
│   │   ├── grounding.py    # validate_citations (NavCapture)
│   │   ├── ingest_grounding.py  # Ingest-side grounding helpers
│   │   ├── ingest_tools.py # Ingest-specific tools
│   │   ├── fix_tools.py    # Fix-specific tools
│   │   ├── lint_tools.py   # Lint-specific tools
│   │   └── query_tools.py  # Query-specific tools
│   ├── wiki/               # Deterministic wiki engine (0 LLM)
│   │   ├── model.py        # load_wiki → Wiki/Page (synthesized frontmatter)
│   │   ├── search.py       # BM25 search + bounded link expansion
│   │   ├── match.py        # match_page decision tree (exact/similar/conflict/none)
│   │   └── dedupe_index.py # regenerate_index (derived view, atomic)
│   ├── lint/
│   │   └── health.py       # health_check → LintReport (deterministic, 7 kinds)
│   ├── schemas/            # Pydantic models
│   │   ├── wiki.py         # Page, Frontmatter, Index, LogEntry
│   │   ├── extraction.py   # Entity, Concept, Contradiction, ExtractionResult
│   │   ├── query.py        # QueryAnswer, Citation
│   │   ├── lint.py         # LintReport, LintIssue
│   │   └── agents_md.py    # AGENTS.md loader
│   ├── io/                 # File I/O (wiki, sources, index, log)
│   │   ├── wiki_io.py      # read/write/delete pages (atomic)
│   │   ├── source_loader.py    # MarkItDown wrapper
│   │   ├── index_manager.py    # Index helpers
│   │   ├── log_manager.py      # Append to log.md
│   │   ├── markdown_parser.py  # Parse [[links]], headings, frontmatter
│   │   └── chunker.py      # Text chunking
│   └── middleware/
│       ├── logging.py      # audit_logging + token_capture middleware
│       └── guardrails.py   # path_guard (write_tools set)
├── wiki/                   # LLM-owned wiki (persistent; gitignored runtime data)
├── raw/                    # Raw sources (immutable)
├── tests/                  # unit, integration, eval, acceptance, fixtures
├── AGENTS.md               # Wiki schema conventions
└── config/                 # Configuration examples
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

### Atomic Writes

All wiki writes are atomic (temp file + rename) to prevent corruption on crashes. `regenerate_index` rebuilds `index.md` atomically from the wiki model — the index is a *derived view*, never hand-edited.

### Guardrails

`path_guard_middleware` intercepts every write tool (`create_page`, `update_page`, `delete_wiki_page`, `edit_wiki_page`, `fix_link`, `append_related_section`, …) and rejects any argument containing `raw/`, an absolute path, or `..`. `read_source` is exempt because it legitimately reads from `raw/`.

## License

[Add your license here]
