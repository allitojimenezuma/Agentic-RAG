# Spec — Streamlit UI for all four agents (multi-page, HITL, durable chat history)

## Intent

Extend the existing query-only Streamlit frontend (`frontend/app.py` + `frontend/chat_driver.py`)
into a full multi-page UI that drives **all four agents** (query, ingest, lint, fix) in-process,
with durable per-agent chat history, and — for the first time — full interactive
human-in-the-loop (approve / reject / edit) for the ingest/lint/fix agents. The CLI and
`src/agentic_rag/` stay **byte-identical**: the UI reuses the same agent builders, the same
interrupt semantics, and the same `Command(resume=...)` decision shapes the CLI uses, so CLI
performance and behavior are untouched.

## Scope

### In scope
- `frontend/app.py` restructured into an `st.navigation` entry point with 4 pages
  (`app_pages/query.py`, `app_pages/ingest.py`, `app_pages/lint.py`, `app_pages/fix.py`).
- New pure-Python, unit-testable modules:
  - `frontend/agent_driver.py` — generic streaming driver for HITL-capable agents
    (ingest/lint/fix): multi-mode streaming, interrupt detection, `Command(resume=...)` resume,
    decision-building helpers.
  - `frontend/history_store.py` — durable JSONL transcript store (per agent + thread),
    survives app restarts.
- New Streamlit-shell modules:
  - `frontend/agents.py` — `@st.cache_resource` agent builders + per-agent config factory.
  - `frontend/ui_common.py` — session-state init, sidebar thread manager, HITL action
    renderer + decision widgets, the shared turn/HITL-rerun loop.
- Full interactive HITL in the UI:
  - ingest: `delete_wiki_page` (approve/reject), `flag_contradiction` (approve/**edit**/reject).
  - lint + fix: `delete_wiki_page` (approve/reject).
  - Edit path takes a new `proposed_resolution` text input (mirrors `cli.py` ingest `e` path).
- Ingest page: natural-language chat input **plus** a picker of files already present under
  `settings.raw_sources_path` (message becomes `Ingest <relpath>`, exactly like the CLI).
  The UI **never writes to `raw/`** — the user uploads raw documents there themselves.
- Durable history: transcripts persisted to `frontend/history/<agent>/<thread_id>.jsonl`
  (gitignored); sidebar thread selector + "New chat" + delete per agent page.
- New tests: `tests/unit/test_agent_driver.py`, `tests/unit/test_history_store.py`,
  `tests/unit/test_ui_common.py`, `tests/unit/test_app_smoke.py` (Streamlit `AppTest`).
- README "Web UI (Streamlit)" section updated.
- `.gitignore`: add `frontend/history/`.

### Out of scope
- Any change to `src/agentic_rag/` (agents, tools, cli, middleware, wiki engine) — CLI stays
  byte-identical. This is a hard acceptance gate.
- Status/log dashboard page (user decision: no dashboard; `agentic-rag status`/`log` remain CLI-only).
- HTTP server, auth, multi-user/concurrent turns (single-user, single-browser, same as today).
- Uploading/writing raw documents from the UI (user uploads to `raw/` on disk).
- Changing the CLI's `MemorySaver` checkpointer — durable history is implemented at the
  frontend layer (JSONL), not by swapping the checkpointer.
- Persisting agent thread memory across restarts (in-process `MemorySaver` only). After a
  restart, a thread's *bubbles* reload but the agent has no memory of prior turns — the same
  semantics as the CLI's fresh `uuid4()` thread per invocation. UI keeps it simple: each
  submitted message is a new turn in the current thread; agent memory accumulates for the
  life of the Streamlit process.
- `st.chat_input(accept_file=...)` / uploads inside chat. No new dependencies beyond the
  existing `[ui]` group (`streamlit>=1.40`).

## Conventions

- Language: Python 3.11+. Package import root `agentic_rag` (`src/` layout, unchanged).
  `frontend/` stays a top-level directory of Streamlit scripts + helper modules (not a
  package). Import pattern `from frontend.chat_driver import …` (tests already do this; pytest
  runs from repo root, `frontend/` is a namespace package).
- Every Streamlit script (`frontend/app.py` and each `frontend/app_pages/*.py`) starts with the
  existing 4-line repo-root bootstrap:
  ```python
  _ROOT = Path(__file__).resolve().parent.parent
  if str(_ROOT) not in sys.path:
      sys.path.insert(0, str(_ROOT))
  ```
  (app.py's already has it; app_pages files need it so `from frontend.ui_common import …`
  works both under `st.navigation` and standalone `AppTest.from_file`.)
- Style: `from __future__ import annotations`; PEP 8; module-level `logging.getLogger(__name__)`;
  pydantic `BaseModel` events — match `frontend/chat_driver.py`.
- Test command (executors and gate run this): `uv sync --extra ui && uv run pytest`. The full
  existing suite must stay green (currently 261 passed / 2 skipped).
- Env via `Settings()` exactly as the CLI does. Required: `OPENAI_API_KEY`.
- Agent builds: reuse `build_query_agent` / `build_ingest_agent` / `build_lint_agent` /
  `build_fix_agent(settings)` from `src/agentic_rag/agents/*.py`. Never bypass
  `agents/factory.py::build_agent` or its middleware.
- Do NOT add new dependencies to `pyproject.toml`. `streamlit` is already in `[ui]`.
- AppTest smoke tests must **never hit the LLM**: they only render pages (agent builds are
  API-free), and drive HITL widgets through an inline stub page + `ScriptedChatModel`
  (`tests/fixtures/fake_llm.py`), never through the real pages' chat input.
- Streamlit version pinned by the `[ui]` group; installed 1.60.0. `AppTest` available at
  `from streamlit.testing.v1 import AppTest`.

## Interfaces

### Reused from `frontend/chat_driver.py` (UNCHANGED — do not edit)
`ToolStart`, `ToolEnd`, `AnswerToken`, `StreamEvent = ToolStart | ToolEnd | AnswerToken | FinalAnswer`
and `async def stream_query(...)`. Query page keeps using `stream_query` (cite-or-die
`FinalAnswer` path). No edits to `chat_driver.py`; its tests stay green.

### New module `frontend/agent_driver.py` (NEW, pure Python, no streamlit import)

```python
from frontend.chat_driver import ToolStart, ToolEnd, AnswerToken, StreamEvent  # reuse

class InterruptEvent(BaseModel):
    kind: Literal["interrupt"] = "interrupt"
    actions: list[dict]          # parsed action_requests: [{"name": str, "args": dict}]

class FinalMessage(BaseModel):
    kind: Literal["final_message"] = "final_message"
    text: str                    # last assistant message content

AgentEvent = StreamEvent | InterruptEvent | FinalMessage

ALLOWED_DECISIONS: dict[str, list[str]] = {
    "delete_wiki_page": ["approve", "reject"],
    "flag_contradiction": ["approve", "edit", "reject"],
}   # mirrors middleware configs in agents/ingest.py and agents/fix.py

def extract_interrupts(state: dict) -> list[dict]
    # state.get("__interrupt__") -> list of Interrupt. Tolerant parse (mirror cli.py):
    #   raw = getattr(i, "value", i); actions = raw.get("action_requests", []) if hasattr(raw,"get") else []
    #   any exception -> []. Flatten across interrupts. Never raises.

def build_decisions(choice: str, actions: list[dict], *, feedback: str = "",
                    index: int = 0, new_resolution: str = "") -> list[dict]
    # choice in {"approve","reject","edit"}; exactly mirrors cli.py ingest/fix HITL shapes:
    #   approve: [{"type":"approve"}] * max(len(actions),1)
    #   reject:  [{"type":"reject","feedback": feedback}] * max(len(actions),1)
    #   edit:    [{"type":"approve"}] * len(actions) with decisions[index] replaced by
    #            {"type":"edit","edited_action":{"name": actions[index]["name"],
    #             "args": {**actions[index]["args"], "proposed_resolution": new_resolution}}}
    # actions assumed non-empty for "edit" (UI only shows edit for flag_contradiction).

async def stream_turn(agent, message: str, config: dict, agent_name: str) -> AsyncGenerator[AgentEvent, None]
    # config is the full pinned per-agent config dict (see agents.py::agent_config).
    # 1. Reset per-turn tool-call accumulation state (copy the translation logic from
    #    chat_driver.py: _tool_call_key / _on_tool_args_fragment semantics — index-keyed,
    #    name remembered on first chunk, ToolStart deferred until name known).
    # 2. Drive: async for mode, chunk in agent.astream(
    #        {"messages": [{"role": "user", "content": message}]},
    #        config, stream_mode=["messages", "values"]):
    #    - mode "messages": exactly chat_driver's translation: ToolMessage -> ToolEnd(name,
    #      output=str(content)[:500]); AIMessageChunk with tool_call_chunks -> accumulate
    #      + ToolStart; plain content -> AnswerToken + free_text accumulation.
    #    - mode "values": IGNORE everything except an "__interrupt__" key. If present:
    #      for each pending action emit ToolEnd(name=action["name"], output="⏸ awaiting human approval")
    #      (so every ToolStart has a matching ToolEnd), then emit ONE InterruptEvent(actions).
    #      Record that an interrupt fired this turn. (Interrupts are captured from values
    #      snapshots DURING streaming — get_state() afterwards does NOT expose them.)
    # 3. After the stream ends: if an interrupt fired, yield NOTHING further (no FinalMessage).
    #    Else extract the last assistant message: state = agent.get_state(config);
    #    last AI message content from state.values["messages"]; fallback "".join(free_text)
    #    if no AI message; yield FinalMessage(text). Never raises.
    # Re-raise agent exceptions (Streamlit shell catches).

async def resume_turn(agent, decisions: list[dict], config: dict, agent_name: str) -> AsyncGenerator[AgentEvent, None]
    # Same translation as stream_turn, driven with:
    #   from langgraph.types import Command
    #   agent.astream(Command(resume={"decisions": decisions}), config, stream_mode=["messages","values"])
    # May yield another InterruptEvent (multi-interrupt chains) or FinalMessage.
```

Pinned decisions:
- `agent_driver.py` imports the shared event types from `chat_driver.py` (reuse, no duplicate
  models) but does **not** touch `stream_query` or its `_nav_capture` logic (query-only).
- The generic driver has **no** cite-or-die finalization — that is query-only. Its terminal
  event is `FinalMessage` (equivalent of `result["messages"][-1].content`, what the CLI echoes).
- Interrupts are detected from `values` snapshots only; `get_state` after the stream is used
  only for final-message extraction.
- Synthetic `ToolEnd("⏸ awaiting human approval")` guarantees the chip lifecycle closes.

### New module `frontend/history_store.py` (NEW, pure Python, stdlib only)

```python
class HistoryStore:
    def __init__(self, root: Path) -> None          # root pinned to <repo>/frontend/history
    def append(self, agent: str, thread_id: str, role: str, content: str) -> None
        # append one JSON line {"role","content"} to root/<agent>/<thread_id>.jsonl (mkdir parents)
    def load(self, agent: str, thread_id: str) -> list[dict]   # [{"role","content"}, ...]
        # missing file -> []; corrupt lines skipped (tolerant); never raises
    def list_threads(self, agent: str) -> list[str]  # thread ids, sorted by file mtime desc
    def delete(self, agent: str, thread_id: str) -> None       # remove file if present
    def new_thread_id(self) -> str                               # str(uuid.uuid4())
```

Pinned: store root = `Path(__file__).resolve().parent.parent / "frontend" / "history"`.
`.gitignore` gains `frontend/history/`. JSONL line format `{"role": "user"|"assistant", "content": str}` — exactly the shape of `st.session_state` messages today.

### New module `frontend/agents.py` (NEW, streamlit-aware shell)

```python
@st.cache_resource
def get_settings() -> Settings          # Settings(); app.py's top-level try/except guard runs first

@st.cache_resource
def get_query_agent() -> object         # build_query_agent(get_settings())
@st.cache_resource
def get_ingest_agent() -> object        # build_ingest_agent(get_settings())
@st.cache_resource
def get_lint_agent() -> object          # build_lint_agent(get_settings())
@st.cache_resource
def get_fix_agent() -> object           # build_fix_agent(get_settings())

def agent_config(agent: str, thread_id: str) -> dict   # EXACTLY mirrors cli.py, per agent:
    # "query":  {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit}
    # "ingest": {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.ingest_recursion_limit}
    # "lint"/"fix": {"configurable": {"thread_id": thread_id}}   (CLI omits recursion_limit for these)
```

### New module `frontend/ui_common.py` (NEW, streamlit-aware shell)

```python
AGENTS: dict[str, str] = {"query": "Wiki Q&A", "ingest": "Ingest", "lint": "Lint", "fix": "Fix"}
# plus Material icon per agent (e.g. forum / upload_file / health_and_safety / construction)

def init_page(agent: str) -> tuple[str, list]
    # ensure st.session_state keys f"{agent}_thread_id" (str uuid) and f"{agent}_messages" (list of
    # {"role","content"}) — one namespace per agent; returns (thread_id, messages)
def render_history(messages: list) -> None          # st.chat_message bubbles
def sidebar(agent: str, store: HistoryStore) -> None # thread selector (list_threads), select -> load +
    # swap thread_id/messages; "🧹 New chat" button (init_page fresh); delete-selected button
def render_actions(actions: list[dict]) -> None     # one st.container(border=True) card per action:
    # name + args summary (page_slug/slug first, then remaining args, truncated); mirror cli.py display
def run_turn(agent: str, agent_name: str, message: str, store: HistoryStore) -> None
    # THE shared turn shell (streamlit-dependent; verified via AppTest, not unit tests):
    # 1. if st.session_state[f"{agent}_pending"] holds a user decision -> resume path:
    #    decisions from build_decisions(...), resume_turn(...) into chips+answer placeholders;
    #    on InterruptEvent -> render_actions again and store new pending; on FinalMessage ->
    #    persist transcript + clear pending.
    # 2. else (fresh submit) -> stream_turn(...) into chips+answer placeholders; on
    #    InterruptEvent -> persist partial transcript, store pending ({"actions": ..., "turn_text": ...}),
    #    render_actions + decision widgets (Approve all / Reject all + feedback / Edit resolution
    #    text_input for flag_contradiction only); on FinalMessage -> final render + persist + clear pending.
    #    Buttons write the chosen decision into session_state and call st.rerun().
    # The rerun protocol: a turn never re-invokes the agent after an interrupt; the pending
    # marker in session_state gates exactly one resume per decision.
def render_final(agent_name: str, final: FinalMessage) -> str
    # markdown string for the terminal bubble; ingest/lint/fix just show the text.
```

### Restructured `frontend/app.py` (ENTRY POINT — rewritten)

- Keep the sys.path bootstrap + `Settings()` try/except (`st.error(...)` + `st.stop()`) at top.
- `st.set_page_config(page_title="Agentic RAG", layout="wide")`.
- `st.navigation([st.Page("app_pages/query.py", title="Wiki Q&A", icon=":material/forum:", default=True),
  st.Page("app_pages/ingest.py", title="Ingest", icon=":material/upload_file:"),
  st.Page("app_pages/lint.py", title="Lint", icon=":material/health_and_safety:"),
  st.Page("app_pages/fix.py", title="Fix", icon=":material/construction:")], position="top")`
  then `.run()`. Page paths are relative to the main script dir (`frontend/`).
- NO agent building, NO chat logic in app.py — only bootstrap, guard, navigation.

### New pages `frontend/app_pages/*.py`

- `query.py` — moves the existing query chat here (chat_driver.stream_query + structured
  `QueryAnswer` render + `reset_chat`), wired to durable history (`HistoryStore`) + sidebar.
  Keeps the existing per-turn `_nav_capture` reset contract. Rendering identical to today's
  app.py. Also a "New chat" affordance.
- `ingest.py` — chat input + a raw-source picker: `st.selectbox("Source in raw/", <relative
  paths of files under settings.raw_sources_path, recursive>)` + "Run" — message becomes
  `Ingest <relpath>`; free text goes through as-is. Uses `get_ingest_agent()`,
  `agent_config("ingest", tid)` (ingest_recursion_limit), `ui_common.run_turn("ingest", …)`.
  Full HITL incl. edit.
- `lint.py` — chat input + a `Run full health check` button sending exactly the CLI's pinned
  message: `"Run a full wiki health check. Report orphans, contradictions, missing links, and data gaps."`
  Uses `get_lint_agent()`, `agent_config("lint", tid)`, `run_turn("lint", …)`. HITL: approve/reject.
- `fix.py` — chat input + `st.pills` quick-actions (latest, missing-frontmatter, broken-link,
  missing-related, missing-index) that fill the input; message goes to the fix agent exactly as
  the CLI's `fix` command does (health_check context built inside the agent's tools/prompt —
  UI does NOT pre-run `health_check`; it only passes the message. NOTE: the CLI pre-runs
  `health_check` and injects issues into the message. The UI pins the same message-shaping:
  `health_check(settings.wiki_path)` -> "Fix these lint issues:\n- [kind] slug: detail"
  or "No issues", exactly mirroring cli.py's `fix` command message construction, in a small
  pure helper `build_fix_message(issue: str, wiki_path: Path) -> str` that lives in
  `frontend/agent_driver.py` so it is unit-testable).
  Uses `get_fix_agent()`, `agent_config("fix", tid)`, `run_turn("fix", …)`. HITL: approve/reject.

Pinned UI decisions:
- No `use_container_width` (deprecated); no custom HTML/CSS; native Streamlit elements only.
- `st.chat_input` per page; `submit_mode="disable"` during a running turn is allowed but not
  required (implementer's choice, keep simple).
- Raw picker reads ONLY (never writes to `raw/`).

## Tasks summary
1. `frontend/agent_driver.py` + unit tests (streaming, interrupts, resume, decisions).
2. `frontend/history_store.py` + unit tests + `.gitignore`.
3. `frontend/agents.py` + `frontend/ui_common.py` + `tests/unit/test_ui_common.py`.
4. Restructure `frontend/app.py` (st.navigation) + `app_pages/query.py` + `tests/unit/test_app_smoke.py`.
5. `app_pages/ingest.py` (raw picker + edit HITL) + README + smoke cases.
6. `app_pages/lint.py` + `app_pages/fix.py` + smoke cases.

## Acceptance
- `git diff --name-only HEAD` contains **no path under `src/`** — the CLI and agents are
  untouched (hard gate). `pyproject.toml` unchanged.
- `uv sync --extra ui && uv run pytest` green: all existing tests (261 passed / 2 skipped) plus
  the new `test_agent_driver.py`, `test_history_store.py`, `test_ui_common.py`, `test_app_smoke.py`.
- `uv run python -m py_compile` passes for every new/changed file.
- `uv run streamlit run frontend/app.py --server.headless=true --server.port=8501` launches and
  `st.navigation` shows the 4 pages (manual smoke; AppTest in CI).
- `AppTest.from_file` runs `frontend/app.py` and each `frontend/app_pages/*.py` without
  exceptions (fake `OPENAI_API_KEY` env; pages render only, no LLM calls).
- `test_agent_driver.py` proves, with a duck-typed fake agent: event order for a plain turn
  (ToolStart → ToolEnd → AnswerToken* → FinalMessage), interrupt turn (… → ToolEnd("⏸ awaiting
  human approval") → InterruptEvent(actions)), and a full resume cycle
  (stream_turn → InterruptEvent → resume_turn → FinalMessage). `extract_interrupts` tolerant
  parsing and `build_decisions` approve/reject/edit shapes are asserted.
- `test_history_store.py` proves append/load round-trip, per-agent isolation, mtime-desc
  `list_threads`, corrupt-line tolerance, delete, unique `new_thread_id`.
- `test_ui_common.py` proves via `AppTest.from_string` + `ScriptedChatModel` stub page: the
  HITL widget flow — actions render, clicking "Approve all" produces `[{"type":"approve"}]*N`
  and resumes, "Reject all" includes feedback, edit path produces the `edited_action` shape.
- Manual (real key) acceptance: ingest a contradiction → Approve / Reject / Edit all work;
  delete request → Approve / Reject works; durable history survives app restart.
- `.gitignore` contains `frontend/history/`.

## Open questions
- none (all product decisions resolved with the user: multi-page UI; full HITL; durable
  history; raw/ uploaded by user — UI only picks existing files; no status/log dashboard).
