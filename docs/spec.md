# Spec — Streamlit streaming chat frontend (query agent)

## Intent
Add a minimal, single-file Streamlit chat frontend (`frontend/app.py`) that talks to the
existing **query agent** in-process: real token streaming of the final answer, live tool-call
activity, multi-turn conversation memory, and the structured `QueryAnswer` render (answer +
citations + confidence + suggestion). No HTTP server, no build step. Ingest/fix/lint and their
HITL interrupts are out of scope.

## Scope

### In scope
- One new runtime file `frontend/app.py` (Streamlit app script).
- One new testable module `frontend/chat_driver.py` — the in-process streaming adapter that
  drives the compiled query agent via `agent.astream(..., stream_mode="messages")` and yields a
  typed event stream consumed by the Streamlit shell.
- One new unit test `tests/unit/test_chat_driver.py`.
- New optional dependency group `[ui]` containing `streamlit` in `pyproject.toml`.
- A short "Run the UI" section in `README.md` (exact command).

### Out of scope (deferred)
- Any HTTP server (FastAPI/LangServe/uvicorn). The agent runs in the Streamlit process.
- Ingest, fix, lint agents and **all HITL** (`Command(resume=...)`, approve/reject/edit UI).
  These agents use `HumanInTheLoopMiddleware` interrupts that are a poor fit for Streamlit's
  rerun model.
- Multi-user / concurrent-request safety. The query agent's `_NAV_CAPTURE`
  (`tools/grounding.py`) and `_WIKI_PATH` (`tools/shared.py`) are module globals; this frontend
  is single-user, single-tab. (Pinned: `chat_driver` resets `_NAV_CAPTURE` per turn — see
  Interfaces — so sequential turns are correct; concurrent turns are explicitly unsupported.)
- Durable checkpointer / persisting threads across process restarts. The existing per-build
  `MemorySaver()` is reused; thread memory lives only for the running Streamlit process.
- Auth, deployment, theming, mobile layout.
- Streaming the `citations`/`confidence`/`suggestion` fields incrementally — only the `answer`
  field is streamed live (mode 2); the rest render once from the final `QueryAnswer`.

## Conventions
- Language: Python 3.11+. Package import root `agentic_rag`, `src/` layout (unchanged).
  `frontend/` is a **new top-level directory** (NOT under `src/`); it is a Streamlit script +
  helper module, not an installable package. Import the app package as
  `from agentic_rag.agents.query import build_query_agent`, etc.
- Env / deps via `uv`. `streamlit` is added under a new **optional** group so the core install
  is unchanged: install with `uv sync --extra ui`.
- Test command (executors and gate run this): `uv run pytest`. The new test lives at
  `tests/unit/test_chat_driver.py` and must pass alongside the existing suite. The existing
  suite must remain green (frontend is purely additive — no edits to `src/agentic_rag/`).
- Import style: `from __future__ import annotations`; PEP 8; module-level
  `logging.getLogger(__name__)` — match the existing `src/agentic_rag/` style.
- Agent build: reuse `build_query_agent(settings)` from `src/agentic_rag/agents/query.py`. Do
  **not** bypass `agents/factory.py::build_agent` or its middleware.
- Config dict shape (exactly, from `cli.py` query command):
  `{"configurable": {"thread_id": <str>}, "recursion_limit": settings.recursion_limit}`
  (default `recursion_limit=30`). `recursion_limit` is a **top-level** key, NOT inside
  `configurable`.
- Input state shape (exactly): `{"messages": [{"role": "user", "content": <str>}]}`
- `Settings` is constructed from env/`.env` exactly as the CLI does: `from agentic_rag.config
  import Settings; settings = Settings()`. Required env: `OPENAI_API_KEY`. The app must surface
  a clear error in the UI if `Settings()` raises (e.g. missing key).
- Test harness: reuse `tests/fixtures/fake_llm.py::ScriptedChatModel` (sync, non-streaming) for
  `test_chat_driver.py`. Because it is non-streaming, tool-call args arrive as one
  `AIMessageChunk` with `tool_call_chunks[i]["args"]` = full args JSON string in a single
  chunk. `chat_driver` must handle both this one-chunk path AND the real incremental path
  (multiple `tool_call_chunks` with partial `args` fragments).

## Interfaces

### New module `frontend/chat_driver.py` (NEW)

The in-process streaming adapter. Pure-Python and unit-testable (no Streamlit import here).

```python
from __future__ import annotations
from typing import AsyncGenerator, Literal
from pydantic import BaseModel
from agentic_rag.schemas.query import QueryAnswer

class ToolStart(BaseModel):
    kind: Literal["tool_start"] = "tool_start"
    name: str
    args: dict          # best-effort parsed args; {} if not parseable yet

class ToolEnd(BaseModel):
    kind: Literal["tool_end"] = "tool_end"
    name: str
    output: str         # tool result string (truncated to ~500 chars for display)

class AnswerToken(BaseModel):
    kind: Literal["answer_token"] = "answer_token"
    text: str           # delta to append to the streaming answer bubble

class FinalAnswer(BaseModel):
    kind: Literal["final"] = "final"
    answer: QueryAnswer # the validated, cite-or-die-filtered QueryAnswer

StreamEvent = ToolStart | ToolEnd | AnswerToken | FinalAnswer

def extract_answer_so_far(accumulated_args_json: str) -> str:
    """Tolerant extraction of the `answer` field value from a (possibly partial)
    JSON string matching submit_query_answer's args schema, whose `answer` key is
    first. Reads from the opening quote after `"answer"` to the next unescaped
    quote, honouring `\\"` escapes. Returns the decoded substring accumulated so
    far (may be incomplete if the JSON is mid-stream). Returns "" if the answer
    field has not started or no opening quote is seen yet. Never raises."""

async def stream_query(
    agent,                      # compiled query agent (CompiledStateGraph) from build_query_agent
    message: str,
    thread_id: str,
    recursion_limit: int,
) -> AsyncGenerator[StreamEvent, None]:
    """Drive one query turn and yield StreamEvents in order.

    Per-turn setup (MUST run first, before any astream call):
      from agentic_rag.tools.grounding import new_nav_capture
      agent._nav_capture = new_nav_capture()   # fresh cite-or-die capture for this turn

    Then:
      config = {"configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit}
      async for chunk, metadata in agent.astream(
              {"messages": [{"role": "user", "content": message}]},
              config, stream_mode="messages"):
        ...

    Event rules:
    - AIMessageChunk with tool_call_chunks: for each tool_call_chunk, if its index/name is
      new this turn, emit ToolStart(name, best-effort args). Accumulate the args string per
      tool-call index. If the tool name is `submit_query_answer`, after updating the
      accumulated args run `extract_answer_so_far(accumulated)` and emit an AnswerToken with
      the NEW suffix (delta vs the last emitted answer text for this call). Handle the
      one-chunk path (full args in a single chunk) and the multi-chunk path (partial
      fragments) identically — delta = new_text[len(old_text):].
    - ToolMessage chunk (metadata langgraph_node == "tools", or chunk is a ToolMessage):
      emit ToolEnd(name=chunk.name, output=str(chunk.content)[:500]).
    - AIMessageChunk with plain content (no tool_call_chunks) and not a tool result: ignore
      (the query agent's final user-facing text is inside submit_query_answer's args, not in
      free content). Do not emit AnswerToken from free content.
    - After the stream ends: find the LAST ToolMessage with name == "submit_query_answer" in
      the agent's accumulated state for the thread
      (`agent.get_state(config).values.get("messages", [])`), parse its `.content` as
      `QueryAnswer.model_validate_json(...)`, and emit one FinalAnswer(answer=that). If no
      such ToolMessage is found, emit FinalAnswer with a QueryAnswer whose answer is the
      concatenation of all AnswerToken text emitted, citations=[], confidence="low",
      suggestion="(no structured answer produced)" — never raise.

    Error handling: if `agent.astream` raises, re-raise (the Streamlit shell catches and
    shows it). Do not swallow.

    Pure helper `extract_answer_so_far` is module-level and unit-tested directly.
    """
```

Pinned decisions:
- `extract_answer_so_far` is the ONLY place that parses partial tool-call JSON. It is pure
  (str -> str), never raises, and is the unit test's primary target.
- `stream_query` does NOT import streamlit. It only imports from `agentic_rag` and stdlib.
- The `FinalAnswer.answer` is the **cite-or-die-filtered** QueryAnswer (the tool already
  filtered citations via `validate_citations`; chat_driver just parses the tool's output JSON).
- `agent._nav_capture = new_nav_capture()` MUST be called at the start of every `stream_query`
  call so citations from turn N do not bleed into turn N+1 (the agent object is built once and
  reused across turns in the Streamlit app).

### New file `frontend/app.py` (NEW)

Streamlit shell. ~one screen of code. Responsibilities only:
1. `from agentic_rag.config import Settings` → `Settings()`; on exception, `st.error(...)` and
   `st.stop()`.
2. Build the query agent ONCE per process: `agent = build_query_agent(settings)`. Use
   `@st.cache_resource` so it survives reruns. Do NOT rebuild per turn.
3. Init `st.session_state` keys on first run:
   - `thread_id`: `str(uuid.uuid4())`
   - `messages`: `[]` (list of `{"role": "user"|"assistant", "content": str}` for the chat log)
4. Render `st.chat_message` bubbles for `st.session_state.messages`.
5. `st.chat_input("Ask the wiki…")` → on submit:
   - append user message to session_state and render it.
   - open an assistant `st.chat_message("assistant")` bubble.
   - inside it, render live tool activity as small status/chips while iterating
     `stream_query(...)`: for `ToolStart` show `🔍 {name}…` (and for `wiki_search` include the
     query arg, for `wiki_read_page` include the slug); for `ToolEnd` mark it done. Use
     `st.status(...)` or a simple expanding container — implementer's choice, keep it minimal.
   - accumulate `AnswerToken.text` and stream it live into the bubble via `st.write_stream`
     (or by updating a `st.empty()` placeholder as tokens arrive — implementer's choice).
   - on `FinalAnswer`: replace/finalize the bubble with the structured render mirroring the CLI
     `query` command: the answer text, then `Confidence: {confidence}`, then a `Citations:`
     bulleted list (`- {slug} - {title}{ (section: {section})}`), then `Suggestion: {suggestion}`
     if non-empty. Append the rendered assistant content to `st.session_state.messages`.
6. A "🧹 New chat" button (e.g. `st.sidebar.button`) that resets `st.session_state.thread_id`
   to a fresh `uuid.uuid4()` and clears `st.session_state.messages`. (Memory of the old thread
   remains in the in-memory checkpointer but is no longer referenced.)
7. Catch exceptions from `stream_query` and show them with `st.error(...)`; do not crash the app.

Pinned decisions:
- `app.py` contains NO business logic beyond wiring Streamlit primitives to `stream_query` and
  the structured renderer. All logic is in `chat_driver.py` so it is unit-testable.
- The structured final render in `app.py` must mirror `cli.py` query command's render block
  (answer + Confidence + Citations + Suggestion), so CLI and UI stay consistent.

### Touched existing file `pyproject.toml`
Add under `[project.optional-dependencies]` a new group:
```toml
ui = ["streamlit>=1.40"]
```
Do NOT add `streamlit` to the core `[project.dependencies]`.

### Touched existing file `README.md`
Add a short "## Web UI (Streamlit)" section with the exact commands:
```
uv sync --extra ui
uv run streamlit run frontend/app.py
```

## Tasks summary
High-level ordering only; the atomic breakdown lives in `progress.md`.
1. Add optional `ui` dep (streamlit) + README run instructions. [trivial]
2. Build + test `frontend/chat_driver.py` (streaming adapter + `extract_answer_so_far`).
3. Build `frontend/app.py` Streamlit shell wiring `stream_query` to the chat UI.

## Acceptance
- `uv sync --extra ui` installs streamlit; `uv run python -c "import streamlit"` succeeds.
- `uv run pytest` is green, including the new `tests/unit/test_chat_driver.py`, and no existing
  test is modified or broken.
- `frontend/chat_driver.py.stream_query` yields the correct event order for a scripted
  two-tool turn (wiki_search → submit_query_answer): a ToolStart for each tool, a ToolEnd for
  wiki_search, one or more AnswerToken whose concatenation equals the scripted `answer`, and a
  FinalAnswer whose `QueryAnswer` parses with non-navigated citations dropped (cite-or-die).
- `extract_answer_so_far` unit tests pass for: empty string, partial key, open-quote with
  incomplete text, complete text, text containing escaped quotes (`\"`), and text followed by
  more fields.
- `uv run python -m py_compile frontend/app.py` succeeds, and
  `uv run streamlit run frontend/app.py --server.headless=true --server.port=8501` launches
  without an import/compile error (manual smoke; the app need not be driven headlessly).
- No file under `src/agentic_rag/` is modified by any task.

## Open questions
- none
