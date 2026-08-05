# Architecture — how to navigate this codebase

This is the layer-by-layer map. Read this before reading code.

## The one idea

Agentic RAG maintains a persistent, interlinked **Markdown wiki** from raw documents.
LLM agents compile knowledge **once** into curated wiki pages; queries navigate +
synthesize over that curated wiki instead of repeated chunk retrieval.

```
raw/ (immutable sources)
  │  read_source (MarkItDown)
  ▼
4 LangChain agents  ── middleware: audit logging → path guardrails → token capture → HITL
  ├─ ingest: read → submit_extraction → match_page (deterministic) → create/update pages
  │          → regenerate_index → append_log
  ├─ query:  wiki_search → wiki_read_page → auto-built answer (cite-or-die citations)
  ├─ lint:   run_health_check (0 LLM) + LLM semantic judgment → write_lint_report
  └─ fix:    health_check issues → pinned kind→tool map → in-wiki edits
        │
        ▼
wiki/ (the artifact: content pages + index.md + log.md; index is a derived view)
```

## The five groups inside `src/agentic_rag/`

| Group | Role | Mental model |
|---|---|---|
| `io/` | **Filesystem adapters** | "How do I read/write files?" — atomic page ops, index.md/log.md codecs, markdown parsing, MarkItDown |
| `wiki/` | **Deterministic domain engine** | "How does the wiki work?" — in-memory model, BM25 search, name matching, health audit. **0 LLM calls** |
| `tools/` | **Agent action surface** | "What can an agent call?" — every `@tool`, grouped by function (nav, grounding, extraction) and by agent (ingest/lint/fix) |
| `agents/` | **Agent builders** | "How is each agent assembled?" — tools + system prompt + middleware + HITL config |
| `schemas/` | **Pydantic contracts** | "What shapes flow between layers?" — frontmatter, index/log entries, extraction, QueryAnswer, LintReport |

Plus three cross-cutting modules at the package root: `config.py` (settings), `cli.py`
(Typer entry point), `logging_config.py` + `token_tracker.py` (observability).

## Where does X live?

| You want to… | Look here |
|---|---|
| Run the CLI | `src/agentic_rag/cli.py` (entry: `agentic-rag` → `cli:app` in `pyproject.toml`) |
| Change settings / add an env var | `src/agentic_rag/config.py` (pydantic-settings) |
| Build a new agent | `agents/<name>.py` → `build_<name>_agent(settings)` + register in `agents/__init__.py` |
| Add a tool an agent can call | `tools/` → register in the agent builder → add to `middleware/guardrails.py` `write_tools` if it writes |
| Read/parse a wiki page file | `io/wiki_io.py` (atomic read/write/delete) + `io/markdown_parser.py` (links, headings, frontmatter) |
| Load the wiki into memory | `wiki/model.py` → `load_wiki(path)` → `Wiki`/`Page` |
| Search the wiki | `wiki/search.py` (BM25) — exposed to agents as `nav.wiki_search` |
| Decide create vs update vs conflict | `wiki/match.py` → `match_page_tool` |
| Audit wiki health | `wiki/health.py` → `health_check()` (0 LLM) — exposed as `lint_tools.run_health_check` |
| Rebuild `index.md` | `wiki/dedupe_index.py` → `regenerate_index` (index is **derived**, never hand-edited) |
| Parse/write `index.md` or `log.md` | `io/index.py`, `io/log.py` (codecs) |
| Understand cite-or-die citations | `tools/grounding.py` (`NavCapture`, `build_final_answer`, `validate_citations`) |
| Understand the ingestion extraction step | `tools/extraction.py` (`submit_extraction`) |
| Understand HITL approval flows | `middleware/guardrails.py` + `HumanInTheLoopMiddleware` in each agent builder |
| Run the Streamlit UI | `frontend/app.py` → `app_pages/<agent>.py` per agent |
| Understand the UI's streaming drivers | `frontend/query_driver.py` (query streaming) vs `frontend/agent_driver.py` (generic HITL driver) |
| Change the wiki schema contract | `AGENTS.md` (root) — injected into every agent system prompt |

## Layer rules (what may import what)

- `tools/` imports `wiki/`, `io/`, `schemas/` — never the reverse.
- `agents/` imports `tools/`, `schemas/` — never the reverse.
- `wiki/` imports `io/` (filesystem reads) and `schemas/` — never `tools/` or `agents/`.
- `io/` imports only `schemas/` + stdlib. Pure adapters.
- `schemas/` imports only pydantic. No domain logic.
- `middleware/` is registered in `agents/factory.py` and runs on every tool call.

## Frontend map

```
frontend/
├── app.py            # st.navigation entry: Settings guard + page list (render-only shell)
├── builders.py       # @st.cache_resource get_*_agent() + agent_config(agent, thread_id)
├── query_driver.py   # query agent: astream → ToolStart/ToolEnd/AnswerToken/FinalAnswer events
├── agent_driver.py   # ingest/lint/fix: same events + InterruptEvent/FinalMessage + resume (HITL)
├── ui_common.py      # shared page shell: session state, sidebar threads, HITL widgets, run_turn
├── history_store.py  # durable JSONL transcripts (frontend/history/<agent>/<thread>.jsonl)
└── app_pages/        # query.py, ingest.py, lint.py, fix.py — thin input wiring over ui_common
```

Pure-Python, unit-testable modules (`query_driver`, `agent_driver`, `history_store`,
`builders` logic) have no Streamlit import; `ui_common.py` and the pages are
Streamlit-bound (verified via AppTest smoke, not unit tests).

## Naming conventions

- `wiki/model.py` = **data** models; `agents/llm.py` = **LLM** factory. If you see
  "model" alone, it's the data model.
- `io/index.py` / `io/log.py` are **codecs** for `index.md` / `log.md` (parse + format +
  atomic write) — not stateful "managers".
- `*_tools.py` = per-agent tool sets; `nav.py`/`grounding.py`/`extraction.py` = tools
  shared across agents or single-purpose finalizers.
- `schemas/` = contracts; `wiki/` = behavior. When in doubt, put the shape in
  `schemas/` and the logic in `wiki/`.
