# Test Suite Improvement Plan — the 3 Levels (`tests/levels/`)

> **Status: EXECUTED 2026-08-06** — all phases complete. Summary:
> - **Phase 0 (D1)**: DeepEval 4.1.5 + litellm as dev deps; custom-provider
>   judge works via a json_schema→json_object shim
>   (`tests/fixtures/deepeval_judge.py`); no dependency conflicts.
> - **Phase 1**: plain-language summary docstrings on every file in
>   `tests/levels/`, `tests/fixtures/`, `tests/integration/`.
> - **Phase 2**: per-query retrieval checks folded into
>   `test_corpus_selfcheck.py`; deleted `test_context_recall.py`,
>   `tests/acceptance/` (both files), `wiki copy/`.
> - **Phase 3**: L2 real-LLM tier extended with on-disk/answer outcome
>   assertions → measured 8/8 = 1.00. Real code gaps found & fixed: lint
>   agent gained `wiki_scan` (its prompt already instructed it); fix
>   contracts propagated to the deliberate `wiki_link_graph` tool; lint cap
>   corrected 4→7 (the prompt's own budget); query prompt now REQUIRES ≥1
>   citation (model was answering without citing).
> - **Phase 4**: `test_answer_quality_real_llm.py` (DeepEval Faithfulness /
>   Answer Relevancy / Contextual Recall over 5 curated questions, floors
>   0.80/0.70/0.80, citation floor 0.80); anchor tests migrated to DeepEval
>   (3-sample averages); `tests/fixtures/eval_dataset.py` gold answers.
> - **Phase 5**: `eval_judge.py` retired (cross-calibration scored
>   identically); docs (§10) + spec.md updated; plan marked EXECUTED.
> - **Verified**: 335 unit/integration + 189 deterministic levels tests
>   green; real tiers: L2 trajectory 8/8, L3 judges + answer quality green
>   with key, skip without key.
>
> Your three asks: (1) I don't understand the levels → plain-language
> summaries on top of every test file; (2) Level 3 measures BM25 performance,
> but I want real-model quality (Faithfulness, Relevance…) via DeepEval/RAGAS;
> (3) more useful real-LLM tests for levels 2 and 3; plus: which files can be
> deleted.

---

## 1. What the 3 levels actually are (cheat-sheet, plain language)

The pyramid separates **"did the code behave correctly?"** (no LLM) from
**"did the model do a good job?"** (real LLM). Roughly:

| Level | Question it answers | LLM? | Cost | Files |
|---|---|---|---|---|
| **L1** | Is the wiki engine + safety net correct? (links, schema, path-guard, tool errors) | None (0-LLM) | 0 | `level1/` (5 files) |
| **L2** | Do the 4 agents call the right tools in the right order, stay within turn caps, leave the wiki consistent? | Scripted fake LLM + **1 real-LLM tier** | low; real tier = ~8 agent runs | `level2/` (4 files) |
| **L3** | Is the *answer the user gets* any good? (grounded? relevant? contradictions handled?) | Scripted + a few real judge calls | low-moderate | `level3/` (4 files) |

Key mental model: **L1 tests the deterministic engine, L2 tests agent
*behavior* (tool calls), L3 tests *output quality* (the answer text).**
The only real-LLM test in L2 today checks *tool call order*, not output.
The only real-LLM tests in L3 are 2 anchor checks each for Faithfulness and
Relevancy — everything else is scripted/deterministic.

---

## 2. File-by-file review (action: keep / modify / delete)

Every file below already HAS a technical docstring. Your ask #1 = rewrite each
into an *understandable* summary using this template (technical detail stays,
plain-language layer added):

```python
"""test_<name>.py — <one-line purpose in plain English>

WHAT IT TESTS:
- <bullet, plain language, what a human would observe>

HOW IT RUNS:
- 0-LLM (always, offline) | scripted fake model | real LLM (@requires_llm, skips without OPENAI_API_KEY)

WHY IT MATTERS (what a failure means):
- <one line: which bug/regression this catches>

RUN: uv run pytest tests/levels/<level>/test_<name>.py
"""
```

### Level 1 — deterministic safety net (all 0-LLM, offline, cheap)

| File | What it tests (plain) | Action |
|---|---|---|
| `test_schema_conformance.py` | Every wiki page obeys AGENTS.md: frontmatter keys, slug==filename, type==folder, `## Related` exists. 3 pages have pinned, documented deviations. | **Keep**, rewrite summary |
| `test_link_integrity.py` | Every `[[link]]` in every page resolves to a real page; no orphans. | **Keep**, rewrite summary |
| `test_path_guard_matrix.py` | The middleware blocks 5 attack patterns (`raw/`, `..`, absolute paths) on all 7 write tools; `read_source` is exempt. | **Keep**, rewrite summary |
| `test_argument_schemas.py` | Tools return `"ERROR:"` strings, never raise; schema-invalid agent calls recover; path-guard short-circuit shape. | **Keep**, rewrite summary |
| `test_trajectory_contract.py` | Unit tests of the DAG validator (`validate_trajectory`): every rule (forbidden tool, prerequisite order, caps, terminal, interrupt). ⚠️ **Mislabeled: its docstring says "Level 2"** while living in `level1/` — fix the label. | **Keep**, fix label + rewrite summary |

### Level 2 — orchestrator discipline (scripted + one real tier)

| File | What it tests (plain) | Action |
|---|---|---|
| `trajectory_contract.py` | Pure Pydantic models + validator used by the real-LLM tier. Not tests itself. | **Keep** (infra) |
| `test_tool_selection.py` | Scripted agents: query never writes; ingest follows causal order; lint runs health check first; fix maps issue-kind→tool; HITL interrupts resume headlessly. | **Keep**, rewrite summary |
| `test_state_consistency.py` | After ingest, disk is consistent: index idempotent, log appended, health check = 0 issues, page has valid frontmatter. | **Keep**, rewrite summary |
| `test_turn_efficiency.py` | Happy-path runs stay under caps (query ≤5, lint ≤4, fix ≤8, ingest ≤15); the cap constants are pinned exact. | **Keep**, rewrite summary |
| `test_trajectory_real_llm.py` | **The only real-LLM file in L2**: 8 pinned tasks, real agents, validates tool-call ORDER only (aggregate ≥0.8). **Does NOT check the output.** | **Modify** — add on-disk/answer outcome assertions (see §5) |

### Level 3 — RAG output quality

| File | What it tests (plain) | Action |
|---|---|---|
| `test_context_recall.py` | BM25 retriever recall@8: do 21 queries surface their ground-truth page in the top 8 hits? Plus "search doesn't mutate the corpus" hashing. | **DELETE** — see §6.1. BM25 is the *retriever*, not the model; 2 of its 4 test groups duplicate `test_corpus_selfcheck.py`; the unique per-query checks fold into the self-check. L3 should be answer quality. |
| `test_faithfulness.py` | Deterministic cite-or-die/confidence proxies (0-LLM) + 2 real-judge anchor tests (grounded ≥0.7, fabricated ≤0.4). | **Modify** — keep proxies; migrate real-judge tier to DeepEval (§5) |
| `test_answer_relevancy.py` | Deterministic proxy (non-empty, key-term present) + stub judge round-trip + 2 real-judge anchor tests. | **Modify** — same treatment |
| `test_contradiction_handling.py` | Scripted ingest of a conflicting source; all 3 HITL resume variants (approve/reject/edit) and their disk effects. | **Keep** (valuable, deterministic); optionally add a real-LLM variant (§5.3) |

### Fixtures (not tests, but need summaries too)

| File | Action |
|---|---|
| `tests/fixtures/eval_corpus.py` | **Keep**, rewrite summary (15 curated + 6 hard queries, floors, copy helpers) |
| `tests/fixtures/eval_judge.py` | Hand-rolled few-shot judge prompts (`judge_faithfulness`, `judge_relevancy`) + Pydantic score models. **Migrate target** — becomes obsolete if DeepEval lands (§6.2); keep as offline fallback meanwhile. |
| `tests/fixtures/eval_hitl.py` | Headless resume helpers (`resume_auto`) — no `input()` ever. **Keep**, rewrite summary |
| `tests/fixtures/fake_llm.py` | `ScriptedChatModel` — the fake model that makes 0-LLM agent tests possible. **Keep**, rewrite summary |

### Other suites (context, not the 3 levels)

| Path | Verdict |
|---|---|
| `tests/levels/test_corpus_selfcheck.py` | **Keep + grow**: fold in the per-query retrieval checks from `test_context_recall.py` before deleting it. |
| `tests/levels/conftest.py` | **Keep** — fixtures + `requires_llm` marker. |
| `tests/integration/` (5 files) | **Keep** — exercises the CLI (CliRunner) + legacy-tool bans; partially overlaps L2 scripted tests but covers entry points L2 doesn't. Rewrite summaries; consolidation optional later. |
| `tests/acceptance/test_ingest_real_source.py` | **DELETE** — it globs `raw/*.md`, and `raw/` has **no `.md` files** (verified) → it always skips. Dead. |
| `tests/acceptance/test_wiki_health.py` | **Fold or delete** — thin smoke test on the real `wiki/`; overlaps the new L2 real-LLM outcome tier. Recommend folding the useful assertion (lint completes on real wiki) into the new suite and deleting the file. |
| `wiki copy/` (repo root) | **DELETE** — dead duplicate of `wiki/`; the eval corpus was seeded from it and now lives in `tests/fixtures/eval_wiki/`. Not a test file but clearly removable. |

---

## 3. Why Level 3 measures BM25 — and why that's wrong for "quality"

`test_context_recall.py` measures **retriever** quality: it runs
`agentic_rag.wiki.search` (BM25 + 1-hop link expansion) and checks the
ground-truth page appears in the top 8. That is a legitimate *retrieval*
sanity test, but:

1. It never calls a model — it says nothing about **answer quality** (faithfulness, relevancy, hallucination).
2. It largely duplicates `test_corpus_selfcheck.py` (aggregate floors + neutrality hashing).
3. The retriever is deterministic pure Python — if it works once, it works forever; it's not where RAG quality risk lives. Risk lives in **what the LLM writes**.

So: **L3 should measure the model's answers** — Faithfulness (does the answer stay grounded in the pages read?), Answer Relevancy (does it answer the question?), plus Contextual Precision/Recall if we add ground truth, and citation validity (cite-or-die working end-to-end with a real model). Retriever checks stay deterministic and move to the self-check.

---

## 4. DeepEval vs RAGAS — research findings (2026, web)

Both frameworks compute the same core metrics (faithfulness, answer relevancy,
context precision, context recall) via **LLM-as-judge**; the differences are
workflow and integration.

| Dimension | **DeepEval** | **RAGAS** |
|---|---|---|
| Core model | **pytest-native**: `assert_test(test_case, metrics)` with per-metric thresholds → hard pass/fail in CI | **dataset-first**: build a Dataset, `evaluate()` over all rows → score table / pandas DataFrame, no native exit code |
| Metrics | `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`, `GEval` (custom criteria in plain English) | `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `factual_correctness`, `context_entity_recall` |
| Ground truth needed | contextual precision/recall + correctness only; faithfulness/relevancy are reference-free | same |
| Judge model | Swappable per metric (`model=`); default reads its own env config | Swappable via langchain LLM/embeddings wrappers (naturally reuses this project's `ChatOpenAI(base_url=…)` pattern) |
| Synthetic data | `Synthesizer` → golden test cases | `TestsetGenerator` → knowledge-graph based test sets (more battle-tested for scale) |
| CI gate | `deepeval test run` / plain pytest — fails the run below threshold | You roll your own `assert mean >= X` |
| Dependencies | `deepeval` (its own stack) | `ragas` + `datasets` + `pandas` |
| Best for | Engineers who want RAG quality gates inside their existing pytest suite | Data scientists iterating on retrieval at dataset scale, dashboards, A/B |

**Recommendation: DeepEval as primary.** Rationale:
1. This repo's whole eval pyramid is pytest — DeepEval drops in as ordinary
   `assert_test()` calls with `@requires_llm` skipping, no new mental model.
2. Its per-metric thresholds match the existing "pinned floor, never loosen"
   philosophy exactly (GROUNDED_MIN=0.7 / FABRICATED_MAX=0.4 today).
3. The hand-rolled `eval_judge.py` is already a mini-DeepEval — replacing it
   with a maintained library removes prompt-engineering drift.

**Watch-outs to spike first (Decision D1, §7):**
- **Custom provider judge**: your `.env` points `OPENAI_BASE_URL` at a proxy
  (OpenRouter/Azure/local). DeepEval's default config reads OpenAI env vars;
  custom judge needs the `model=` param or a small `DeepEvalBaseLLM` subclass.
  RAGAS judges are plain langchain `ChatOpenAI` → trivially uses your existing
  `judge_model()` pattern. **Spike: install `deepeval`, wire a custom judge to
  your base_url, run one faithfulness assertion. If friction > half a day,
  fall back to RAGAS** (or keep the hand-rolled judge — it already works).
- **Dependency conflicts**: DeepEval/RAGAS pin their own langchain-core/pydantic
  versions; `uv add --dry-run` first; run the full unit suite after install.
- **Cost**: judge calls = metrics × test cases × (1-3 LLM calls each). Keep the
  curated L3 set ≤ ~15 queries; use the configured model (temp 0); cache results.

---

## 5. New real-LLM tests (your ask #3)

### 5.1 Level 2 — turn "tool order" acceptance into "outcome" acceptance

Extend `test_trajectory_real_llm.py` (or add `test_agents_real_outcomes.py`
with the same harness): after each task run, assert the **on-disk / output
effect**, not just tool call order:

| Task | New outcome assertions (real model, real agents) |
|---|---|
| query (2 tasks) | final answer non-empty; **≥1 citation; every citation slug in NavCapture** (cite-or-die holds with the real model); answer contains the query's key term |
| ingest sample.md | page created on disk with valid frontmatter; `health_check` = 0 issues on the copy; index contains new slug; log gained `ingest` entry |
| ingest contradiction-source.md | run ends at HITL interrupt (already asserted) — now also resume via `resume_auto(approve)` and verify the page carries the new claim + log/index updated |
| lint | `lint-report-YYYY-MM-DD.md` actually written under `wiki/` and contains the deterministic findings |
| fix (3 tasks) | the seeded defect is **gone**: `health_check` no longer reports `missing-frontmatter`/`broken-link`/`missing-related` for the target slug; `regenerate_index` ran last |

These catch a model that "calls the right tools in the right order" but writes
garbage — the exact gap today.

### 5.2 Level 3 — real query-agent answer quality via DeepEval (new file)

New file: `tests/levels/level3/test_answer_quality_real_llm.py`
(`@requires_llm`; skips cleanly without a key).

1. **Curated eval dataset** (~10-15 queries): reuse the 15 `CURATED_QUERIES`
   ground-truth slugs; add a hand-written `expected_answer` (gold answer) per
   query + the set of pages that must be read. Ground truth lives in
   `eval_corpus.py` or a new `eval_dataset.py` fixture.
2. **Run the REAL query agent** over a tmp copy of `eval_wiki` (fresh
   `thread_id` per query, `init_shared_tools` on the copy — same isolation
   discipline as L2).
3. **Capture per query**: `input` = question, `actual_output` = final answer,
   `retrieval_context` = the content of pages the agent actually navigated
   (from NavCapture — this is exactly the "context" a judge needs),
   `expected_output` = gold answer.
4. **Score with DeepEval**:
   - `FaithfulnessMetric` (no ground truth) — threshold 0.80 (tune at first real run; **never loosen after pinning**)
   - `AnswerRelevancyMetric` — threshold 0.70
   - `ContextualRecallMetric` + `ContextualPrecisionMetric` (ground truth) — report-only or gated, your call (D4)
   - `GEval` custom criterion: "every citation is a real page navigated this turn" — cheap citation-validity check the stock metrics don't cover
5. **Keep the anchor-separation tests** (grounded ≥0.7 / fabricated ≤0.4) but
   reimplement them on the framework so the old hand-rolled prompts retire
   cleanly (§6.2).

### 5.3 Level 3 — optional real-LLM contradiction variant

Add one `@requires_llm` test to `test_contradiction_handling.py`: real ingest
agent ingests `contradiction-source.md`, auto-resume `approve`, assert the
merged claim is on disk. Flagged **optional** — real-model HITL flows are the
flakiest tests here (more judge/agent calls, interrupt timing). Include only
if D4 budget allows.

### 5.4 L3 must keep its cheap deterministic core

The 0-LLM proxies (cite-or-die gate, confidence inference, key-term
containment) and the scripted HITL flows stay — they give fast offline
regression protection. Only the *real-judge* tier migrates to DeepEval; the
stub round-trip tests (`_judge` parsing, corrective retry) migrate to test
DeepEval's own metric objects instead.

---

## 6. Deletions (your ask #4)

### 6.1 `tests/levels/level3/test_context_recall.py` — DELETE
- Duplicates `test_corpus_selfcheck.py`:
  - aggregate recall floors (curated/hard) — identical logic in both files
  - neutrality hashing — self-check does it on the *committed* trees
- Unique value = the **21 per-query parametrized assertions** (`expected_slug
  in top-8`). **Before deleting, fold those into
  `test_corpus_selfcheck.py`** as `test_curated_query_retrieves_ground_truth`
  / `test_hard_query_retrieves_ground_truth` (same parametrize, same fixtures).
- Then L3 contains only answer-quality tests — matches the level's purpose.

### 6.2 `tests/fixtures/eval_judge.py` — DELETE *after* DeepEval migration
- Once L3's real-judge tier runs on DeepEval, the hand-rolled
  `FAITHFULNESS_SYSTEM` / `RELEVANCY_SYSTEM` prompts + `judge_faithfulness` /
  `judge_relevancy` + the stub tests in `test_faithfulness.py` /
  `test_answer_relevancy.py` that pin them become obsolete.
- Sequence: land DeepEval tier first → keep old judge as a
  cross-calibration check for one run (compare scores) → delete + remove stub
  tests. Don't delete early: it's currently the only offline judge harness.

### 6.3 `tests/acceptance/` — DELETE both files (see §2 table)
- `test_ingest_real_source.py` always skips (no `raw/*.md`; verified).
- `test_wiki_health.py` thin + overlapping; fold the one useful assertion
  (lint completes on real wiki) into the L2 real-LLM tier, then delete.

### 6.4 `wiki copy/` — DELETE
- Dead duplicate; corpus lives in `tests/fixtures/eval_wiki/`. (Repo-level
  clutter, not a test file — confirm with you before removing.)

### 6.5 NOT deleting (deliberate)
- `tests/integration/` — unique CLI coverage (CliRunner, typer, env patching).
  Note overlap with L2 scripted tool-contract tests; consolidation is
  optional follow-up, not this plan.
- `tests/levels/level1/test_trajectory_contract.py` — keep, just fix the
  "Level 2" mislabel in its docstring.
- `tests/levels/level2/trajectory_contract.py`, `test_tool_selection.py`,
  `test_state_consistency.py`, `test_turn_efficiency.py`,
  `test_contradiction_handling.py` — all keep (deterministic value, no dup).

---

## 7. Decision points for you (before implementation)

| # | Decision | Options | My default |
|---|---|---|---|
| **D1** | Eval framework | DeepEval / RAGAS / keep hand-rolled judge | **DeepEval, with a ½-day spike** for custom-base_url judge; RAGAS or hand-rolled as fallback |
| **D2** | `eval_judge.py` retirement | Delete after migration / keep as cross-check | Delete after one cross-calibration run |
| **D3** | Ground truth dataset | Hand-write gold answers for 10-15 queries / generate via RAGAS `TestsetGenerator` | Hand-write (small set, reviewable) |
| **D4** | L3 gating strictness | Hard gate (fails CI below threshold) / report-only for the first N runs then gate | Report-only on first run to measure baselines → pin floors → then hard gate |
| **D5** | Budget per run | ~10 queries × 4 metrics ≈ 40-120 judge calls | ~10 queries; temp 0; reuse `.env` model |
| **D6** | Non-test deletions | `wiki copy/`, `tests/acceptance/` | Delete with your OK |

---

## 8. Implementation phases (with verification gates)

- **Phase 0 — Spike (D1).** `uv add --dry-run deepeval`; install; wire a custom
  judge to `OPENAI_BASE_URL`; one `assert_test` with `FaithfulnessMetric` on a
  sample Q/A/context. *Gate: metric scores return + suite still imports.*
- **Phase 1 — Summaries.** Rewrite the module docstring of every file in
  `tests/levels/` + `tests/fixtures/` + `tests/integration/` +
  `tests/acceptance/` with the §2 template. *Gate: `pytest --collect-only`
  unchanged counts; docstrings render.*
- **Phase 2 — L3 retriever consolidation + deletions.** Fold per-query checks
  into `test_corpus_selfcheck.py`; delete `test_context_recall.py`,
  `tests/acceptance/` (after folding), `wiki copy/` (D6). *Gate:
  `uv run pytest tests/ -q` green; collect count drops by exactly the deleted
  tests.*
- **Phase 3 — L2 outcome assertions.** Extend `test_trajectory_real_llm.py`
  with the §5.1 on-disk/answer checks. *Gate: run with `OPENAI_API_KEY` set —
  all 8 tasks pass with outcomes, or the failures are real model gaps
  (report, don't loosen).*
- **Phase 4 — L3 DeepEval tier.** New `test_answer_quality_real_llm.py` +
  dataset fixture; migrate anchor tests; keep deterministic proxies. *Gate:
  runs green with key; skips without key; baselines recorded in docstring.*
- **Phase 5 — Cleanup.** Delete `eval_judge.py` + its stub tests (D2); update
  `docs/documentation.html` §10 table + `archive/spec.md` L2/L3 sections;
  update this plan to EXECUTED. *Gate: full suite green, `git status` shows
  only intended changes.*

---

## 9. Risks

- **Judge cost/nondeterminism**: thresholds are bands, not exact values; temp 0,
  cheap judge for routine runs, stronger for release gates (both frameworks
  allow swapping the judge).
- **DeepEval dependency conflicts** (pydantic/langchain pins) — spike first
  (Phase 0); `uv` resolves conflicts cleanly or we fall back.
- **Real-LLM flakiness** — every real tier must fail loudly and report, never
  auto-loosen; per-run isolation (fresh tmp copy + `thread_id`) already
  enforced; reuse it for the new tiers.
- **Data confidence**: gold answers for the eval dataset are hand-written and
  small — mark them as low-confidence until reviewed; prefer the 21 corpus
  queries whose ground truth is already pinned.
