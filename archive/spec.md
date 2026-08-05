# Spec — Agent Eval Suite: Levels 1–3 (from scratch)

## Intent

Replace the thin/broken eval coverage in `tests/eval/` with a from-scratch, three-tier
evaluation suite for the four agents (query, ingest, lint, fix). Tier 1 verifies
deterministic correctness with **zero LLM calls** (schema conformance, path guards,
link integrity, cite-or-die). Tier 2 verifies trajectory/tool-calling discipline on
**two layers**: an always-on deterministic layer (system guarantees) and a real-LLM
trajectory layer that asserts **causal DAG contracts and invariants** (order
prerequisites, forbidden tools per role, mandatory terminal invariants) instead of
brittle fixed tool sequences — so real-model runs stay stable. All HITL flows in tests
are driven by **programmatic auto-approve/auto-reject resume** (never CLI `input()`),
so the suite is headless/CI-safe. Tier 3 verifies RAG/output quality: reproducible
Context Recall@K on a committed, **neutral corpus** (thresholds calibrated to the actual
retriever, never the corpus to the thresholds), faithfulness and answer-relevancy via
**calibrated real-LLM judges** (few-shot 1.0/0.0 anchors, strict Pydantic parsing;
model/url/key from `.env` via `Settings()`; tests auto-skip without a key), and
end-to-end contradiction handling. The existing 321 unit/integration tests stay
untouched as the regression baseline.

## Scope

### In scope
- New committed eval corpus `tests/fixtures/eval_wiki/` — a clean, AGENTS.md-conformant
  wiki (~20 pages: entities, concepts, sources, comparisons + `index.md` + `log.md`),
  seeded from the on-disk `wiki copy/` content but **cleaned so `health_check` reports
  zero issues**. Tests copy it to a tmp dir; the committed tree is never mutated and is
  **neutral** (never shaped to fit recall thresholds — see Conventions).
- New committed raw-source fixtures `tests/fixtures/eval_raw/` — small markdown sources
  (a normal `sample.md` and a `contradiction-source.md` whose claims conflict with a
  corpus page) used by the ingest trajectory/contradiction evals.
- New committed broken-wiki fixture `tests/fixtures/eval_broken_wiki/` — a tiny wiki
  (3-4 pages) with **seeded, pinned defects** for fix-agent evals: one
  missing-frontmatter page, one broken-link page, one missing-related page. Keeps the
  clean corpus pristine (fix evals never run against `eval_wiki/`).
- New fixture modules:
  - `tests/fixtures/eval_corpus.py` — `EVAL_WIKI_SRC`, `CURATED_QUERIES` (reuse the
    15 natural-language query→slug pairs from `tests/eval/test_search_recall.py`),
    `HARD_QUERIES` (≥5 typo/synonym/cross-type variants), `RECALL_K=8`, threshold
    floors, `copy_eval_wiki(tmp_path) -> Path`, `eval_wiki` fixture, and an
    `eval_env(tmp_path)` fixture returning both wiki + raw copies.
  - `tests/fixtures/eval_judge.py` — calibrated real-LLM judge harness (few-shot
    anchors, strict JSON + Pydantic, corrective retry).
  - `tests/fixtures/eval_hitl.py` — headless HITL auto-responders
    (`auto_decide` / `resume_auto`) driving `Command(resume={"decisions": [...]})`.
- New suites, all deterministic tier = 0 LLM calls, judge/trajectory tier = real LLM:
  - `tests/levels/level1/`: `test_schema_conformance.py`, `test_path_guard_matrix.py`,
    `test_link_integrity.py`.
  - `tests/levels/level2/`: `test_tool_selection.py`, `test_argument_schemas.py`,
    `test_turn_efficiency.py`, `test_state_consistency.py` (deterministic);
    `trajectory_contract.py` (pure DAG/invariant validator) +
    `test_trajectory_contract.py` (its deterministic unit tests);
    `test_trajectory_real_llm.py` (acceptance-tier real-model runs).
  - `tests/levels/level3/`: `test_context_recall.py`, `test_faithfulness.py`,
    `test_answer_relevancy.py`, `test_contradiction_handling.py`.
  - `tests/levels/conftest.py` — level-scoped fixtures (re-export `eval_wiki` /
    `eval_env` / HITL helpers, judge skip marker).
- Remove `tests/eval/test_search_recall.py` and `tests/eval/test_grounding_gate.py`
  (superseded — coverage moves into the levels suite; user-approved rewrite), remove
  the now-empty `tests/eval/` directory, and update the README "Development" section.

### Out of scope
- Rewriting existing `tests/unit/` / `tests/integration/` tests (321-passed baseline
  stays green and untouched).
- Any change to `src/agentic_rag/` (agents, tools, middleware, wiki engine) — hard
  acceptance gate.
- New runtime or dev dependencies (no ragas, no pandas/datasets). Judges are
  hand-rolled prompts over the already-installed `langchain-openai`.
- `pyproject.toml` changes.
- CI pipeline configuration changes.
- Level 4 / production observability (explicitly excluded by the user).

## Conventions

- Language: Python 3.11+. Package import root `agentic_rag` (`src/` layout). Tests
  import fixtures as `from tests.fixtures.fake_llm import ScriptedChatModel` (pytest
  runs from repo root; `tests/` is a package).
- Test command (executors and gate run this): `uv sync --all-extras && uv run pytest`.
  Per-level: `uv run pytest tests/levels/level1 -q` (likewise level2/level3).
  Real-LLM tiers: `@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY")`
  — same pattern as `tests/acceptance/`.
- Agent construction in deterministic tests: build agents directly with the scripted
  model via `build_agent(model=ScriptedChatModel(...), tools=[...], system_prompt=...,
  middleware=[...])` (pattern from `tests/integration/test_query_grounded.py`). Never
  call `build_*_agent(settings)` in the deterministic tier. `init_shared_tools(str(wiki_path))`
  must be called before invoking nav/ingest tools in a test. Real-LLM tiers use the
  REAL builders (`build_query_agent(settings)` etc., as `tests/acceptance/test_wiki_health.py`).
- **Headless HITL (hard rule)**: any test whose scripted/real flow triggers an
  interrupt (`flag_contradiction`, `delete_wiki_page`) MUST resume programmatically via
  `Command(resume={"decisions": [...]})` using `tests/fixtures/eval_hitl.py` helpers
  (auto-approve / auto-reject / auto-edit). NEVER drive the CLI runner into an
  interactive decision, NEVER `monkeypatch`/patch `input()` as a workaround. Tests
  must not hang in CI/headless environments, ever.
- **Corpus neutrality (hard rule)**: the committed corpora (`eval_wiki/`,
  `eval_raw/`, `eval_broken_wiki/`) are fixed after T1 and never modified to make a
  metric pass. Recall calibration adjusts QUERY PHRASING (natural language, never the
  page title verbatim) and THRESHOLDS to the actual behavior of
  `src/agentic_rag/wiki/search.py` (BM25) on the corpus; measured numbers are reported
  in test docstrings. Any threshold adjustment must be documented with evidence —
  silently altering the corpus to overfit is prohibited.
- Corpus discipline: tests ALWAYS operate on tmp copies (`copy_eval_wiki`,
  `eval_env`, or copy of `eval_broken_wiki`); never write to committed fixture trees.
  Post-write assertions go through the deterministic engine (`load_wiki`,
  `health_check`, `regenerate_index`, `match_page`), never ad-hoc file parsing.
- Tool error semantics (pin): every tool returns a string; errors are returned as
  error-prefixed strings (`"ERROR: ..."` / `"Error: ..."` per existing tools), never
  raised. Argument-schema evals assert this contract.
- Env: `Settings()` exactly as the CLI does. Judge + trajectory tiers read
  `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` (optionally from a user-supplied
  `.env`). No `.env` exists in the repo today → real tiers skip until the user creates
  one (copy of `.env.example`).
- Style: `from __future__ import annotations`; module-level docstrings explaining the
  level's intent; reset `NavCapture` between tests via the existing `_reset_nav_capture`
  autouse pattern.

## Interfaces

### `tests/fixtures/eval_corpus.py` (NEW)
```python
EVAL_WIKI_SRC: Path          # Path(__file__).parent / "eval_wiki"  (committed, neutral)
EVAL_RAW_SRC: Path           # Path(__file__).parent / "eval_raw"   (sample.md, contradiction-source.md)
EVAL_BROKEN_WIKI_SRC: Path   # Path(__file__).parent / "eval_broken_wiki" (seeded defects)

CURATED_QUERIES: list[tuple[str, str]]   # (natural-language query, ground-truth slug); reuse the 15 pairs from the old tests/eval/test_search_recall.py
HARD_QUERIES: list[tuple[str, str]]      # >= 5 harder variants (typo / synonym / cross-type phrasing)
RECALL_K: int = 8
CURATED_RECALL_FLOOR: float = 0.80       # aggregate recall@8 floor over CURATED_QUERIES
HARD_RECALL_FLOOR: float = 0.60          # aggregate recall@8 floor over HARD_QUERIES

def copy_eval_wiki(tmp_path: Path) -> Path
    # shutil.copytree(EVAL_WIKI_SRC, tmp_path/"eval_wiki"); returns the copy path.
@pytest.fixture
def eval_wiki(tmp_path: Path) -> Path    # copy_eval_wiki(tmp_path)
@pytest.fixture
def eval_env(tmp_path: Path) -> tuple[Path, Path]
    # copies BOTH EVAL_WIKI_SRC and EVAL_RAW_SRC into tmp; returns (wiki_path, raw_path)
def copy_broken_wiki(tmp_path: Path) -> Path   # copytree of EVAL_BROKEN_WIKI_SRC
```
Pin: recall floors as above are MINIMUMS. Executor measures real recall@8 on the corpus
and calibrates query phrasing to reach the floors (phrasing must stay natural, never
page-title mirrors); reports measured curated + hard recall in the test docstring. If a
floor is genuinely unachievable after good-faith phrasing work, it may be adjusted ONLY
with documented evidence (docstring with before/after numbers) — the corpus itself is
never touched for calibration.

### `tests/fixtures/eval_judge.py` (NEW — calibrated)
```python
def judge_model() -> ChatOpenAI | None
    # Settings() -> ChatOpenAI(model=settings.openai_model, api_key=..., base_url=...,
    #   temperature=0). None if no key set.

class FaithfulnessScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)   # Pydantic enforces bounds — no drift
    rationale: str

class RelevancyScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    rationale: str

def judge_faithfulness(question: str, answer: str, contexts: list[str]) -> FaithfulnessScore
def judge_relevancy(question: str, answer: str) -> RelevancyScore
```
Pinned calibration rules:
- System prompts MUST embed few-shot anchors with explicit 1.0 / 0.0 (and 0.5 partial)
  worked examples. Faithfulness anchors: a claim directly supported by context → 1.0;
  partially supported → 0.5; absent-or-contradicted by context → 0.0. Relevancy
  anchors: answer directly addresses the question → 1.0; off-topic / refusal-dodge →
  0.0. The anchors are fixed strings in the module (reviewable, not drifting).
- Output: instruct strict JSON; attempt `response_format={"type": "json_object"}` on
  the ChatOpenAI call; if the endpoint rejects it, retry WITHOUT the param (local
  proxies). Pydantic-parse; on parse failure, ONE corrective retry (re-send with the
  parse error and raw output in the message); on second failure raise `RuntimeError`
  including the raw model output for debugging. Never silently default to a passing score.
- Judge tests assert anchor separation: a grounded answer must score ≥ 0.7, a
  fabricated/off-topic answer ≤ 0.4 (run with stub model in the deterministic tier and
  with the real judge in the acceptance tier).

### `tests/fixtures/eval_hitl.py` (NEW — headless HITL)
```python
def auto_decide(state: dict, choice: str = "approve", *, feedback: str = "",
                index: int = 0, new_resolution: str = "") -> list[dict]
    # Parse state["__interrupt__"] (tolerant: getattr(i, "value", i); raw.get("action_requests", []);
    # any exception -> []). Build decisions EXACTLY mirroring cli.py:
    #   approve: [{"type": "approve"}] * max(len(actions), 1)
    #   reject:  [{"type": "reject", "feedback": feedback}] * max(len(actions), 1)
    #   edit:    [{"type": "approve"}] * len(actions), decisions[index] replaced by
    #            {"type": "edit", "edited_action": {"name": actions[index]["name"],
    #             "args": {**actions[index]["args"], "proposed_resolution": new_resolution}}}
def resume_auto(agent, config, *, choice: str = "approve", feedback: str = "",
                index: int = 0, new_resolution: str = "") -> dict
    # agent.invoke(Command(resume={"decisions": auto_decide(state, ...)}), config=config)
    # where state = agent.get_state(config).values or the interrupt-bearing result dict.
    # Used by every HITL-triggering test (contradiction, delete). NEVER calls input().
```
Pin: `resume_auto` is the ONLY way tests interact with interrupted runs; scripts that
trigger an interrupt and do not resume are test bugs. `eval_hitl` must have its own
deterministic unit tests (decision shapes approve/reject/edit + tolerant parsing).

### L2 trajectory contracts — `tests/levels/level2/trajectory_contract.py` (NEW, pure)
```python
class TrajectoryContract(BaseModel):
    agent: str
    message: str
    allowed: list[str]                      # tools the agent may call; anything else -> violation
    prerequisites: list[tuple[str, str]] = []   # edge (A, B): every call to B must come AFTER the
                                                # FIRST call to A (vacuous if B never called)
    required: list[str] = []                # each must appear >= 1
    required_any: list[tuple[str, ...]] = []    # each group: at least one member must appear
    write_tools: list[str] = []             # tools that arm the terminal invariant
    terminal_after: list[str] = []          # each must appear >= 1 AFTER the LAST call to any
                                            # write_tool (vacuous if no write tool was called)
    forbidden: list[str] = []               # explicit extra prohibitions (allowed already implies)
    max_calls: int = 30
    ends_with_interrupt: bool = False       # run must terminate in a HITL interrupt

class TrajectoryReport(BaseModel):
    passed: bool
    violations: list[str]                   # human-readable, one per failing rule

def validate_trajectory(contract: TrajectoryContract, tool_names: list[str]) -> TrajectoryReport
    # Pure, deterministic. Semantics:
    # - any tool not in allowed -> "forbidden tool '<name>' called at step <i>"
    # - prerequisite (A, B): first_index(B) <= first_index(A) -> violation (B absent -> ok)
    # - required missing -> "required tool '<name>' never called"
    # - required_any group with no member present -> violation
    # - terminal: let w = last index of any write_tool; each t in terminal_after must have an
    #   index > w (no write_tool called -> vacuous pass)
    # - len(tool_names) > max_calls -> violation
```
Pin: `validate_trajectory` is exercised by `tests/levels/level2/test_trajectory_contract.py`
with synthetic trajectories (pass + every failure mode) — the DAG logic itself is
deterministically verified, independent of any LLM.

### L2 real-LLM tier — `tests/levels/level2/test_trajectory_real_llm.py` (NEW, acceptance)
`TRAJECTORY_TASKS: list[TrajectoryContract]` — pinned ~8 tasks (see table). Run each
task's real agent (`build_*_agent(settings)`; `Settings()` env; recursion limits from
settings; thread_id per run via uuid4) over the `eval_env` / broken-wiki tmp copies;
record tool names in order from `result["messages"]` (AIMessage.tool_calls +
ToolMessage per step); call `validate_trajectory`; on failure print the FULL recorded
trajectory (step, tool, args) for human review. Aggregate pass rate ≥ 0.8 over the
task set; report tool-selection accuracy (correct-first-tool rate). Tasks ending in an
interrupt must be resumed with `resume_auto` before final assertion (or asserted
directly as `ends_with_interrupt`).

| Task | message | contract highlights |
|---|---|---|
| query single-hop | "What is MLX?" | allowed={wiki_search, wiki_read_page, wiki_summary}; prereq (wiki_search→wiki_read_page); required=[wiki_read_page]; max 5 |
| query multi-hop | "How does MLX relate to Apple Silicon?" | same allowed/prereq; required=[wiki_read_page]; max 5 |
| ingest happy path | "Ingest raw/sample.md" | allowed=ingest toolset; prereqs (read_source→submit_extraction, submit_extraction→match_page_tool, match_page_tool→create_page, match_page_tool→update_page); required=[read_source, submit_extraction, match_page_tool, regenerate_index, append_log]; required_any=[(create_page, update_page)]; write_tools=[create_page, update_page]; terminal_after=[regenerate_index, append_log]; max 15 |
| ingest contradiction | "Ingest raw/contradiction-source.md" | same allowed/prereqs up to match; required=[flag_contradiction]; write_tools=[]; ends_with_interrupt=True; max 15 |
| lint full check | "Run a full wiki health check. Report orphans, contradictions, missing links, and data gaps." | allowed=lint toolset; prereq (run_health_check→write_lint_report); required=[run_health_check]; max 4 |
| fix missing-frontmatter | "Fix the missing-frontmatter issue on entities/broken-fm" (broken-wiki copy) | allowed=fix toolset; required=[add_frontmatter]; forbidden=[fix_link, append_related_section, edit_wiki_page]; write_tools=fix writes; terminal_after=[regenerate_index]; max 8 |
| fix broken-link | "Fix broken links" (broken-wiki copy) | required=[fix_link]; forbidden=[add_frontmatter, append_related_section, edit_wiki_page]; terminal_after=[regenerate_index]; max 8 |
| fix missing-related | "Fix missing-related on entities/lonely" (broken-wiki copy) | required=[append_related_section]; forbidden=[add_frontmatter, fix_link, edit_wiki_page]; terminal_after=[regenerate_index]; max 8 |

Pin: never assert model wording; contracts describe causal order + invariants only.
Tool inventories (pin): query = {wiki_search, wiki_read_page, wiki_summary}; ingest =
{read_source, submit_extraction, match_page_tool, wiki_read_page, wiki_scan,
wiki_link_graph, create_page, update_page, flag_contradiction, regenerate_index,
append_log, delete_wiki_page}; lint = {run_health_check, wiki_link_graph,
wiki_read_page, write_lint_report}; fix = {wiki_read_page, edit_wiki_page,
add_frontmatter, fix_link, append_related_section, regenerate_index,
delete_wiki_page}. Write-tools set (path guard) = {create_page, update_page,
delete_wiki_page, write_lint_report, add_frontmatter, fix_link,
append_related_section}.

### Reused (UNCHANGED)
- `ScriptedChatModel` / `ResponseState` from `tests/fixtures/fake_llm.py`.
- `build_agent` from `src/agentic_rag/agents/factory.py` (always attaches audit
  logging, path guard, token capture).
- Deterministic engine: `load_wiki` / `Wiki.by_slug` (`wiki/model.py`),
  `search(wiki, query, k=8)` (`wiki/search.py`), `match_page(wiki, name, page_type)`
  (`wiki/match.py`), `health_check(wiki_path)` (`lint/health.py`),
  `regenerate_index` (`wiki/dedupe_index.py`), `validate_citations` /
  `build_final_answer` / `NavCapture` (`tools/grounding.py`).
- HITL resume contract: `langgraph.types.Command(resume={"decisions": [...]})`; interrupt
  detection via `state["__interrupt__"]` (shapes as in `src/agentic_rag/cli.py`).

## Tasks summary
1. Eval fixtures: neutral corpora (`eval_wiki/`, `eval_raw/`, `eval_broken_wiki/`) +
   `eval_corpus.py` + `eval_judge.py` (calibrated) + `eval_hitl.py` + level conftest +
   corpus self-check tests.
2. Level 1: schema conformance, path-guard matrix, link integrity (0 LLM).
3. Level 2 (deterministic): tool-selection invariants (per-agent whitelists, ordering,
   fix kind→tool map; HITL flows via `resume_auto`).
4. Level 2 (deterministic): argument schemas, turn efficiency, state consistency.
5. Level 2: `trajectory_contract.py` + its deterministic validator tests +
   `test_trajectory_real_llm.py` (DAG contracts, acceptance tier).
6. Level 3: Context Recall@K (calibrated floors, neutral corpus) + deterministic
   faithfulness proxies.
7. Level 3: few-shot judges (faithfulness/relevancy) + contradiction handling
   end-to-end with `resume_auto` (approve/edit/reject).
8. Remove `tests/eval/`, README update, full-suite verification.

## Acceptance
- `git diff --name-only HEAD` contains **no path under `src/`** and no change to
  `pyproject.toml` (hard gate). `.env`/`.env.example` untouched.
- `uv sync --all-extras && uv run pytest` green: existing baseline (321 passed /
  2 skipped) **plus** the levels suite; real-LLM tiers skip without `OPENAI_API_KEY`.
  `uv run pytest tests/levels/ -q` green standalone.
- `tests/levels/level1/` + deterministic level2/level3 tests pass with NO env vars and
  no network; they are headless-safe (grep-verifiable: no `input()` / `mock.patch("builtins.input")`
  anywhere in `tests/levels/`).
- `health_check(copy_eval_wiki(tmp))` reports zero issues (corpus self-check). The
  broken-wiki fixture reports EXACTLY the seeded defect kinds (self-check), and
  `eval_wiki/` is never used by fix evals.
- Every `CURATED_QUERIES`/`HARD_QUERIES` ground-truth slug exists in the corpus
  (self-check). Recall floors reached with measured numbers documented; corpus file
  mtimes/hashes unchanged by recall tests (neutrality).
- `eval_hitl.py` has deterministic tests proving approve/reject/edit decision shapes +
  tolerant interrupt parsing; every HITL-triggering test in the suite resumes via
  `resume_auto`.
- `trajectory_contract.py` validated by deterministic unit tests covering: pass,
  forbidden-hit, required-missing, required_any-miss, prerequisite-order violation,
  terminal-after-last-write violation, max_calls breach, ends_with_interrupt mismatch.
- Judge modules embed the pinned few-shot anchors; `judge_model()` round-trip is proven
  with a stub; with a key, judge tests run against the real model and assert anchor
  separation (grounded ≥ 0.7 / fabricated ≤ 0.4).
- With a key, `uv run pytest tests/levels/level2/test_trajectory_real_llm.py -q` passes
  ≥ 8/10 of `TRAJECTORY_TASKS` and prints tool-selection accuracy + failed trajectories.
- Turn-efficiency caps pinned: query ≤ 5, lint ≤ 4, fix ≤ 8, ingest ≤ 15 (executor
  tunes to observed happy path; may tighten, never loosen).
- `tests/eval/` no longer exists; README documents `tests/levels/` and the real-LLM tiers.

## Open questions
- none blocking. Note: no `.env` exists in the repo today, so the real-judge and
  real-trajectory tiers skip until the user supplies one (copy `.env.example`, set a
  key). All deterministic tiers run regardless.
