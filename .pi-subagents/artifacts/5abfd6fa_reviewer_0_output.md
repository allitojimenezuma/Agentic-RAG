All files inspected. Here's my review:

## Review

### ✅ Check 1: pyproject.toml — PASS

All 9 runtime deps match §15 exactly:
- `langchain>=0.3`, `langchain-openai>=0.2`, `langgraph>=0.2` ✓
- `pydantic>=2`, `pydantic-settings>=2` ✓
- `markitdown[all]>=0.0.1`, `markdown-it-py>=3.0` ✓
- `typer>=0.12`, `python-dotenv>=1.0` ✓

Dev deps: `pytest>=8`, `pytest-asyncio>=0.23`, `respx>=0.21` ✓
`requires-python = ">=3.11"` ✓
`agentic-rag = "agentic_rag.cli:app"` ✓
hatchling build backend with `packages = ["src/agentic_rag"]` for src layout ✓
pytest config: `testpaths = ["tests"]`, `asyncio_mode = "auto"` ✓

### ✅ Check 2: config.py — PASS

All 11 fields from §5 present with correct types/defaults (`config.py:8-28`):
- `openai_api_key: str` (required) ✓
- `openai_base_url: str = "https://api.openai.com/v1"` ✓
- `openai_model: str = "gpt-4.1-mini"` ✓
- `wiki_path: Path`, `raw_sources_path: Path`, `agents_md_path: Path` ✓
- `recursion_limit: int = 12`, `hitl_enabled: bool = True` ✓
- `retrieval_mode: str = "index"`, `vector_db_path: Path | None = None` ✓
- `markitdown_llm_describe_images: bool = False` ✓

Uses `model_config` dict (pydantic-settings v2 modern style) vs §5's older `class Config` — correct improvement, not a deviation.

### ✅ Check 3: AGENTS.md — PASS

All §6 sections covered:
- 5 page types (entity, concept, source, comparison, overview) ✓
- Naming conventions (slug rules, directory/file paths, example) ✓
- `[[Page Name]]` and `[[Page Name|alias]]` cross-reference format ✓
- YAML frontmatter spec (slug, type, title, sources, updated, tags) ✓
- 5 update rules (supersedes, flag_contradiction, update_index, append_log, date) ✓
- Index entry format ✓
- Log entry format with `## [YYYY-MM-DD HH:MM] <op> | <title>` prefix ✓
- 6 hard rules ✓
- Bonus: Data Confidence section ✓

### ✅ Check 4: Directory layout — PASS

Phase 0 scope = directory stubs + `__init__.py`. All 5 subpackages present:
- `schemas/`, `io/`, `tools/`, `agents/`, `middleware/` — each with `__init__.py` ✓
- Root files: `__init__.py`, `config.py`, `paths.py`, `cli.py`, `main.py` ✓
- Module files (e.g. `schemas/wiki.py`, `io/source_loader.py`) correctly deferred to Phase 1+

### ✅ Check 5: .env.example — EXACT MATCH

All 7 vars match §5 spec verbatim:
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `WIKI_PATH`, `RAW_SOURCES_PATH`, `AGENTS_MD_PATH`, `RETRIEVAL_MODE` ✓

### ✅ Check 6: pip install -e . — PASS

- `tomllib.load()` confirms valid TOML ✓
- `pip install -e . --dry-run` exits 0, resolves all deps, "Would install agentic-rag-0.1.0" ✓

### Additional scaffolding present
- `README.md`: correct architecture summary, setup, CLI usage, project structure ✓
- `conftest.py`: fixtures for `wiki_path`, `raw_path`, `agents_md_path` ✓
- `config/config.yaml.example`: present ✓
- `cli.py`: stubs for all 5 subcommands ✓
- `paths.py`: `resolve_wiki_path` and `slugify` helpers ✓

No blockers, no deviations, no missing items.

---