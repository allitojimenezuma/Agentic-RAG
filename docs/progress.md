# Progress

<!-- One line per task. Status markers: PENDING | RUNNING | COMPLETED | BLOCKED -->

- [x] [COMPLETED] T1: Add optional `[ui]` dep group (`streamlit>=1.40`) to `pyproject.toml` + "Web UI (Streamlit)" section to `README.md` [trivial] `pyproject.toml` `README.md`
- [ ] [PENDING] T2: Build `frontend/chat_driver.py` streaming adapter (`stream_query` async generator + `extract_answer_so_far` tolerant partial-JSON helper) + unit test `tests/unit/test_chat_driver.py` using `ScriptedChatModel` `frontend/chat_driver.py` `tests/unit/test_chat_driver.py`
- [ ] [PENDING] T3: Build `frontend/app.py` Streamlit shell — cached `build_query_agent`, `session_state` thread_id + messages, `st.chat_input`, live tool chips + streamed answer tokens via `stream_query`, structured `QueryAnswer` final render, "New chat" reset `frontend/app.py`

## Interface handoffs
<!-- Populated by the orchestrator from each task's handoff contract block. Do not pre-fill beyond the stub. -->
- (none yet)
