# Spec — Agent Eval Suite: Levels 1–3 (from scratch)

## Intent

Replace the thin/broken eval coverage in `tests/eval/` with a from-scratch, three-tier
evaluation suite for the four agents (query, ingest, lint, fix). Tier 1 verifies
deterministic correctness with **zero LLM calls** (schema conformance, path guards,
link integrity, cite-or-die). Tier 2 verifies trajectory/tool-calling discipline with
scripted models (tool selection, argument schemas, turn efficiency, state consistency).
Tier 3 verifies RAG/output quality: reproducible Context Recall@K on a committed corpus,
faithfulness and answer-relevancy via **real-LLM judges** (model/url/key read from
`.env` via the existing `Settings()` loader; tests auto-skip when no key is present),
and end-to-end contradiction handling. The existing 321 unit/integration tests stay
untouched as the regression baseline.

## Scope

### In scope
- New committed eval corpus `tests/fixtures/eval_wiki/` — a clean, AGENTS.md-conformant
  wiki (~20 pages: entities, concepts, sources, comparisons + `index.md` + `log.md`),
  seeded from the on-disk `wiki copy/` content but **cleaned so `health_check` reports
  zero issues**. Tests copy it to a tmp dir; the committed tree is never mutated.
- New fixture modules:
  - `tests/fixtures/eval_corpus.py` — `EVAL_WIKI_SRC`, `CURATED_QUERIES` (reuse the
    existing 15 query→slug pairs from `tests/eval/test_search_recall.py`, which map
    onto the wiki-copy content), `HARD_QUERIES` (typo/synonym/cross-type variants),
    `copy_eval_wiki(tmp_path) -> Path`, and an `eval_wiki` pytest fixture.
  - `tests/fixtures/eval_judge.py` — real-LLM judge harness (`ChatOpenAI` built from
    `Settings()`: `openai_model`, `openai_base_url`, `openai_api_key`; temperature 0),
    structured JSON outputs parsed with pydantic, skipif support.
- New suites, all deterministic tier = 0 LLM calls, judge tier = real LLM:
  - `tests/levels/level1/`: `test_schema_conformance.py`, `test_path_guard_matrix.py`,
    `test_link_integrity.py`.
  - `tests/levels/level2/`: `test_tool_selection.py`, `test_argument_schemas.py`,
    `test_turn_efficiency.py`, `test_state_consistency.py`.
  - `tests/levels/level3/`: `test_context_recall.py`, `test_faithfulness.py`,
    `test_answer_relevancy.py`, `test_contradiction_handling.py`.
  - `tests/levels/conftest.py` — level-scoped fixtures (re-export `eval_wiki`, judge
    skip marker, fake-key env helper).
- Remove `tests/eval/test_search_recall.py` and `tests/eval/test_grounding_gate.py`
  (superseded — their coverage moves into the levels suite; user-approved rewrite),
  remove the now-empty `tests/eval/` directory, and update the README "Development"
  test-structure section and test commands accordingly.
- README "Development" update: document the levels suite + test commands +
  `tests/eval/` removal.

### Out of scope
- Rewriting existing `tests/unit/` / `tests/integration/` tests (321-passed baseline
  stays green and untouched).
- Any change to `src/agentic_rag/` (agents, tools, middleware, wiki engine) — hard
  acceptance gate.
- New runtime or dev dependencies (no ragas, no pandas/datasets). Judges are
  hand-rolled prompts over the already-installed `langchain-openai`.
- `pyproject.toml` changes (existing dev extras already hold pytest/pytest-asyncio/respx).
- CI pipeline configuration changes.
- Level 4 / production observability (explicitly excluded by the user).

## Conventions

- Language: Python 3.11+. Package import root `agentic_rag` (`src/` layout). Tests
  import fixtures as `from tests.fixtures.fake_llm import ScriptedChatModel` (pytest
  runs from repo root; `tests/` is a package).
- Test command (executors and gate run this): `uv sync --all-extras && uv run pytest`.
  Per-level: `uv run pytest tests/levels/level1 -q` (likewise level2/level3).
  Judge tests: `@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY")`
  — same pattern as `tests/acceptance/`.
- Agent construction in tests: build agents directly with the scripted model via
  `build_agent(model=ScriptedChatModel(...), tools=[...], system_prompt=..., middleware=[...])`
  (pattern from `tests/integration/test_query_grounded.py`). Never call
  `build_query_agent(settings)` etc. in the deterministic tier (those construct a real
  `ChatOpenAI` via `get_model`). `init_shared_tools(str(wiki_path))` must be called
  before invoking nav/ingest tools in a test.
- Env: `Settings()` exactly as the CLI does. Judge harness reads the same env the
  agents use — `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` (optionally from a
  user-supplied `.env`). No `.env` exists in the repo today, so judge tests skip until
  the user creates one (copy of `.env.example`).
- Corpus discipline: tests ALWAYS operate on a tmp copy via `copy_eval_wiki(tmp_path)`;
  never write to `EVAL_WIKI_SRC`. Post-write assertions go through the deterministic
  engine (`load_wiki`, `health_check`, `regenerate_index`, `match_page`), never ad-hoc
  file parsing.
- Tool error semantics (pin): every tool returns a string; errors are returned as
  error-prefixed strings (`"ERROR: ..."` / `"Error: ..."` per existing tools), never
  raised. Argument-schema evals assert this contract.
- Style: `from __future__ import annotations`; module-level docstrings explaining the
  level's intent; one focused class/function per concern; tests are self-documenting
  (no shared mutable state between tests; reset `NavCapture` between tests via the
  existing `_reset_nav_capture` autouse pattern).

## Interfaces

### `tests/fixtures/eval_corpus.py` (NEW)
```python
EVAL_WIKI_SRC: Path          # Path(__file__).parent / "eval_wiki"  (committed tree)
CURATED_QUERIES: list[tuple[str, str]]   # (natural-language query, ground-truth slug) — reuse the 15 pairs from tests/eval/test_search_recall.py
HARD_QUERIES: list[tuple[str, str]]      # >= 5 harder variants (typo / synonym / cross-type phrasing)
RECALL_K: int = 8
CURATED_RECALL_THRESHOLD: float = 0.90   # aggregate recall@8 over CURATED_QUERIES
HARD_RECALL_THRESHOLD: float = 0.70      # aggregate recall@8 over HARD_QUERIES

def copy_eval_wiki(tmp_path: Path) -> Path
    # shutil.copytree(EVAL_WIKI_SRC, tmp_path/"eval_wiki"); returns the copy path.
    # The committed tree is never mutated by tests.

@pytest.fixture
def eval_wiki(tmp_path: Path) -> Path    # returns copy_eval_wiki(tmp_path)
```
Pin: `RECALL_K = 8`, thresholds as above. Executor must verify empirically and report
the actual recall numbers; if the corpus can't reach the pinned thresholds, FIX THE
CORPUS (add a page/alias), never silently lower a threshold.

### `tests/fixtures/eval_judge.py` (NEW)
```python
def judge_model() -> ChatOpenAI | None
    # Settings() -> ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key,
    #   base_url=settings.openai_base_url, temperature=0). Returns None if no key set.

class FaithfulnessScore(BaseModel):
    score: float          # 0..1
    rationale: str        # 1-2 sentences: which claims are/aren't supported

class RelevancyScore(BaseModel):
    score: float          # 0..1
    rationale: str

def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> FaithfulnessScore
    # Prompt: decompose the answer into atomic claims; each claim must be supported by
    # the contexts (the navigated page texts, not just slugs). score = supported/total.
    # Strict JSON output; pydantic-parse; one retry on parse failure; else raise.

def judge_relevancy(question: str, answer: str) -> RelevancyScore
    # Prompt: does the answer directly address the question (no irrelevant preamble,
    # no refusal-dodge, covers the asked aspect)? Strict JSON output, same retry rule.
```
Pin: judges never silently pass — score must be a real float in [0,1] with rationale.
Judge tests are `skipif(not os.getenv("OPENAI_API_KEY"))`; deterministic proxies for
the same metrics live in the same test files and ALWAYS run.

### Reused (UNCHANGED)
- `ScriptedChatModel` / `ResponseState` from `tests/fixtures/fake_llm.py`.
- `build_agent` from `src/agentic_rag/agents/factory.py` (incl. its middleware pipeline
  — audit logging, path guard, token capture are always attached).
- Deterministic engine: `load_wiki` / `Wiki.by_slug` (`wiki/model.py`),
  `search(wiki, query, k=8)` (`wiki/search.py`), `match_page(wiki, name, page_type)`
  (`wiki/match.py`), `health_check(wiki_path)` (`lint/health.py`), `regenerate_index`
  (`wiki/dedupe_index.py`), `validate_citations` / `build_final_answer` / `NavCapture`
  (`tools/grounding.py`).
- Tool inventory (pin for L2): query = {`wiki_search`, `wiki_read_page`, `wiki_summary`};
  ingest = {`read_source`, `submit_extraction`, `match_page_tool`, `wiki_read_page`,
  `wiki_scan`, `wiki_link_graph`, `create_page`, `update_page`, `flag_contradiction`,
  `regenerate_index`, `append_log`, `delete_wiki_page`}; lint = {`run_health_check`,
  `wiki_link_graph`, `wiki_read_page`, `write_lint_report`}; fix = {`wiki_read_page`,
  `edit_wiki_page`, `add_frontmatter`, `fix_link`, `append_related_section`,
  `regenerate_index`, `delete_wiki_page`}. Write-tools set (path guard) =
  {`create_page`, `update_page`, `delete_wiki_page`, `write_lint_report`,
  `add_frontmatter`, `fix_link`, `append_related_section`}.

## Tasks summary
1. Eval corpus + fixture modules + level conftest (+ corpus self-check tests).
2. Level 1: schema conformance, path-guard matrix, link integrity.
3. Level 2: tool-selection invariants (per-agent whitelists, ordering, fix kind→tool map).
4. Level 2: argument schemas, turn efficiency, state consistency.
5. Level 3: Context Recall@K + deterministic faithfulness proxies.
6. Level 3: LLM-judge faithfulness/relevancy + contradiction handling end-to-end.
7. Remove `tests/eval/`, README update, full-suite verification.

## Acceptance
- `git diff --name-only HEAD` contains **no path under `src/`** and no change to
  `pyproject.toml` (hard gate). `.env`/`.env.example` untouched.
- `uv sync --all-extras && uv run pytest` green: existing baseline (321 passed /
  2 skipped) **plus** the new levels suite; judge tests skip without `OPENAI_API_KEY`.
  `uv run pytest tests/levels/ -q` green standalone.
- `tests/levels/level1/` and `tests/levels/level2/` are 100% deterministic — they pass
  with NO env vars set and make no network calls.
- `health_check(copy_eval_wiki(tmp))` reports zero issues (corpus self-check test).
- Every `CURATED_QUERIES`/`HARD_QUERIES` ground-truth slug exists in the corpus
  (self-check test, guards drift).
- Level 3 judge tests are proven to exercise the real judge when a key is present:
  `OPENAI_API_KEY=sk-test... uv run pytest tests/levels/level3 -q` runs the judge tier
  (with a stub `judge_model` unit test proving prompt→JSON→pydantic round-trip without
  a key).
- `tests/eval/` no longer exists; README test-structure section documents
  `tests/levels/` and no longer references `tests/eval/`.
- Turn-efficiency caps pinned: query happy path ≤ 5 tool calls, lint ≤ 4, fix ≤ 8,
  ingest ≤ 15 (executor tunes to observed happy path; may tighten, never loosen).

## Open questions
- none blocking. Note: no `.env` exists in the repo today, so the real-judge tier skips
  until the user supplies one (copy `.env.example`, set a key). All deterministic tiers
  run regardless.
