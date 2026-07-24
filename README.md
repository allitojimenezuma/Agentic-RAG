# Agentic RAG

An agentic RAG system that maintains a persistent, interlinked markdown wiki from raw sources. Unlike classic RAG (re-derive answers from chunks each query), the wiki is a *compounding artifact*: cross-references, contradictions, and synthesis are compiled once and kept current.

## Architecture

Three layers:
1. **Raw sources** (`raw/`, immutable) — LLM reads, never writes.
2. **Wiki** (`wiki/`, LLM-owned) — summaries, entity/concept/comparison pages, `index.md`, `log.md`.
3. **Schema** (`AGENTS.md`) — conventions injected into agent system prompts.

### Agent Framework

Agents are built with **LangChain `create_agent()`** + middleware:
- `HumanInTheLoopMiddleware` for approval workflows (delete pages, contradictions)
- Custom middleware for logging, guardrails (path validation), and error handling
- `MemorySaver` checkpointer for per-invocation state (HITL resume within a call)

### Agents

- **Ingest Agent** — reads sources, extracts entities/concepts, creates/updates wiki pages, flags contradictions.
- **Query Agent** — answers questions against the wiki with inline citations (read-only).
- **Lint Agent** — health-checks the wiki: orphans, contradictions, missing links, stale claims.

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI-compatible API key (OpenAI, Azure, OpenRouter, local proxy)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd langchain-rag

# Install in development mode
pip install -e ".[dev]"

# Verify installation
agentic-rag --help
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
| `RECURSION_LIMIT` | `12` | Max agent loop turns |
| `HITL_ENABLED` | `true` | Enable human-in-the-loop |

## CLI Usage

### Ingest a Source

```bash
# Ingest a markdown file
agentic-rag ingest raw/sample.md

# Ingest any supported format (pdf, docx, pptx, xlsx, html, etc.)
agentic-rag ingest path/to/document.pdf

# The agent will:
# 1. Convert source to markdown using MarkItDown
# 2. Extract entities and concepts
# 3. Create/update wiki pages
# 4. Update index.md and log.md
# 5. Flag contradictions for human approval (HITL)
```

### Query the Wiki

```bash
# Ask a question
agentic-rag query "What is MLX?"

# The agent will:
# 1. Read index.md to find relevant pages
# 2. Search for matching entries
# 3. Read pages and follow [[cross-references]]
# 4. Synthesize an answer with inline citations
# 5. Return markdown with "Sources consulted" list
```

### Health Check

```bash
# Run lint agent
agentic-rag lint

# The agent will:
# 1. Read all wiki pages
# 2. Check for orphans (no inbound [[links]])
# 3. Check for missing pages (dangling [[X]] links)
# 4. Detect contradictions between pages
# 5. Write a report to wiki/lint-report-YYYY-MM-DD.md
```

### View Status and Logs

```bash
# Show wiki statistics
agentic-rag status

# View log entries
agentic-rag log

# Tail last 5 entries
agentic-rag log --tail 5
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test types
pytest tests/unit/          # Unit tests (no network)
pytest tests/integration/   # Integration tests (FakeChatModel)
pytest tests/acceptance/    # Acceptance tests (real LLM, requires OPENAI_API_KEY)

# Run with coverage
pytest --cov=agentic_rag --cov-report=html
```

### Test Structure

```
tests/
├── unit/                  # Fast, isolated tests (no network)
│   ├── test_wiki_io.py
│   ├── test_index_manager.py
│   ├── test_markdown_parser.py
│   └── ...
├── integration/           # Agent integration tests (FakeChatModel)
│   ├── test_ingest_agent.py
│   ├── test_query_agent.py
│   ├── test_lint_agent.py
│   └── test_cli.py
├── acceptance/            # Real LLM tests (requires API key)
│   ├── test_wiki_health.py
│   └── test_ingest_real_source.py
└── fixtures/              # Test fixtures and FakeChatModel
```

### Adding a New Tool

1. Create tool function with `@tool` decorator in `src/agentic_rag/tools/`
2. Add type hints and docstring with `Args:` section
3. Register tool in relevant agent builder (`ingest.py`, `query.py`, or `lint.py`)
4. Add unit tests in `tests/unit/test_tools.py`

### Adding Middleware

1. Create middleware function with `@wrap_tool_call` or `@before_model` decorator
2. Add to agent's middleware list in `src/agentic_rag/agents/`
3. Middleware patterns:
   - `wrap_tool_call`: Intercept tool execution (logging, guards, retry)
   - `before_model` / `after_model`: Inspect/modify state around model calls
   - `HumanInTheLoopMiddleware`: Pause for human approval on dangerous tools

## Project Structure

```
agentic_rag/
├── src/agentic_rag/      # Main package
│   ├── config.py         # Settings (pydantic-settings)
│   ├── paths.py          # Path helpers
│   ├── schemas/          # Pydantic models
│   │   ├── wiki.py       # Page, Frontmatter, Index, LogEntry
│   │   ├── agents_md.py  # AGENTS.md loader
│   │   └── extraction.py # Entity, Concept, ExtractionResult
│   ├── io/               # File I/O (wiki, sources, index, log)
│   │   ├── wiki_io.py    # read/write/delete pages
│   │   ├── source_loader.py  # MarkItDown wrapper
│   │   ├── index_manager.py  # Read/update index.md
│   │   ├── log_manager.py    # Append to log.md
│   │   └── markdown_parser.py  # Parse [[links]], headings, frontmatter
│   ├── tools/            # LangChain tools
│   │   ├── shared.py     # Tools shared across agents
│   │   ├── ingest_tools.py   # Ingest-specific tools
│   │   ├── query_tools.py    # Query-specific tools
│   │   └── lint_tools.py     # Lint-specific tools
│   ├── agents/           # Agent builders
│   │   ├── factory.py    # create_agent wrapper
│   │   ├── ingest.py     # build_ingest_agent()
│   │   ├── query.py      # build_query_agent()
│   │   ├── lint.py       # build_lint_agent()
│   │   └── prompts.py    # System prompt builders
│   ├── middleware/        # HITL, logging, guardrails
│   │   ├── logging.py    # Audit trail middleware
│   │   ├── guardrails.py # Path validation middleware
│   │   └── hitl.py       # HumanInTheLoopMiddleware config
│   ├── cli.py            # Typer CLI
│   └── main.py           # Entry point
├── wiki/                 # LLM-owned wiki (persistent)
├── raw/                  # Raw sources (immutable)
├── tests/                # Unit, integration, acceptance tests
├── AGENTS.md             # Wiki schema conventions
└── config/               # Configuration examples
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
- **Delete wiki page**: Always pauses; approve to delete, reject to keep
- **Flag contradiction**: When new source conflicts with existing page; approve to apply resolution, reject to keep old claim

### Atomic Writes

All wiki writes are atomic (temp file + rename) to prevent corruption on crashes.

## License

[Add your license here]
