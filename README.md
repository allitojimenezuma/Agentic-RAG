# Agentic RAG

An agentic RAG system that maintains a persistent, interlinked markdown wiki from raw sources. Unlike classic RAG (re-derive answers from chunks each query), the wiki is a *compounding artifact*: cross-references, contradictions, and synthesis are compiled once and kept current.

## Architecture

Three layers:
1. **Raw sources** (`raw/`, immutable) — LLM reads, never writes.
2. **Wiki** (`wiki/`, LLM-owned) — summaries, entity/concept/comparison pages, `index.md`, `log.md`.
3. **Schema** (`AGENTS.md`) — conventions injected into agent system prompts.

## Agents

- **Ingest Agent** — reads sources, extracts entities/concepts, creates/updates wiki pages, flags contradictions.
- **Query Agent** — answers questions against the wiki with inline citations (read-only).
- **Lint Agent** — health-checks the wiki: orphans, contradictions, missing links, stale claims.

## Setup

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API key

# Verify
agentic-rag --help
```

## CLI

```bash
agentic-rag ingest <path>       # Ingest a source into the wiki
agentic-rag query "question"    # Query the wiki
agentic-rag lint                # Health-check the wiki
agentic-rag status              # Wiki statistics
agentic-rag log [--tail N]      # View log entries
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=agentic_rag
```

## Project Structure

```
agentic_rag/
├── src/agentic_rag/      # Main package
│   ├── config.py         # Settings (pydantic-settings)
│   ├── paths.py          # Path helpers
│   ├── schemas/          # Pydantic models
│   ├── io/               # File I/O (wiki, sources, index, log)
│   ├── tools/            # LangChain tools
│   ├── agents/           # Agent builders
│   ├── middleware/        # HITL, logging, guardrails
│   ├── cli.py            # Typer CLI
│   └── main.py           # Entry point
├── wiki/                 # LLM-owned wiki (persistent)
├── raw/                  # Raw sources (immutable)
├── tests/                # Unit, integration, acceptance tests
├── AGENTS.md             # Wiki schema conventions
└── config/               # Configuration examples
```
