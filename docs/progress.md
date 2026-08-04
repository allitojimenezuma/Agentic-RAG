# Progress

<!-- One line per task. Status markers: PENDING | RUNNING | COMPLETED | BLOCKED -->

- [x] [COMPLETED] T1: Add optional `[ui]` dep group (`streamlit>=1.40`) to `pyproject.toml` + "Web UI (Streamlit)" section to `README.md` [trivial] `pyproject.toml` `README.md`
- [x] [COMPLETED] T2: Build `frontend/chat_driver.py` streaming adapter (`stream_query` async generator + `extract_answer_so_far` tolerant partial-JSON helper) + unit test `tests/unit/test_chat_driver.py` using `ScriptedChatModel` `frontend/chat_driver.py` `tests/unit/test_chat_driver.py`
- [ ] [PENDING] T3: Build `frontend/app.py` Streamlit shell — cached `build_query_agent`, `session_state` thread_id + messages, `st.chat_input`, live tool chips + streamed answer tokens via `stream_query`, structured `QueryAnswer` final render, "New chat" reset `frontend/app.py`

## Interface handoffs
<!-- Populated by the orchestrator from each task's handoff contract block. Do not pre-fill beyond the stub. -->
- T2 exports: `StreamEvent = ToolStart | ToolEnd | AnswerToken | FinalAnswer` (pydantic models, `kind`-tagged), `extract_answer_so_far(accumulated_args_json: str) -> str`, `async def stream_query(agent, message: str, thread_id: str, recursion_limit: int) -> AsyncGenerator[StreamEvent, None]` in `frontend/chat_driver.py` (no streamlit import; resets `agent._nav_capture` per turn; emits FinalAnswer always, never raises on missing tool output)
