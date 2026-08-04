# Progress

<!-- One line per task. Status markers: PENDING | RUNNING | COMPLETED | BLOCKED -->

- [ ] [PENDING] T1: Build `frontend/agent_driver.py` — multi-mode streaming driver for HITL agents (`InterruptEvent`, `FinalMessage`, `extract_interrupts`, `build_decisions`, `stream_turn`, `resume_turn`; reuses `ToolStart/ToolEnd/AnswerToken` from `chat_driver.py`) + `tests/unit/test_agent_driver.py` `frontend/agent_driver.py` `tests/unit/test_agent_driver.py`
- [ ] [PENDING] T2: Build `frontend/history_store.py` — durable JSONL transcript store (`append/load/list_threads/delete/new_thread_id`) + `tests/unit/test_history_store.py`; add `frontend/history/` to `.gitignore` `frontend/history_store.py` `tests/unit/test_history_store.py` `.gitignore`
- [ ] [PENDING] T3: Build `frontend/agents.py` (cached `get_*_agent` + `agent_config` mirroring cli.py per-agent config shapes) + `frontend/ui_common.py` (session init, sidebar thread manager, HITL action renderer + decision widgets, `run_turn` rerun shell) + `tests/unit/test_ui_common.py` (pure asserts + AppTest HITL flow) `frontend/agents.py` `frontend/ui_common.py` `tests/unit/test_ui_common.py`
- [ ] [PENDING] T4: Restructure `frontend/app.py` into `st.navigation` entry (4 pages, `st.Page` paths relative to `frontend/`) + move query chat to `frontend/app_pages/query.py` (chat_driver + structured render + durable history via `HistoryStore`) + `tests/unit/test_app_smoke.py` (AppTest renders app.py + query page, no LLM) `frontend/app.py` `frontend/app_pages/query.py` `tests/unit/test_app_smoke.py`
- [ ] [PENDING] T5: Build `frontend/app_pages/ingest.py` — chat + raw/ file picker (`Ingest <relpath>` message) + full HITL (approve/reject/edit via `run_turn`) + `build_fix_message` helper in `agent_driver.py` with unit tests; extend `test_app_smoke.py`; update README "Web UI (Streamlit)" section `frontend/app_pages/ingest.py` `frontend/agent_driver.py` `tests/unit/test_agent_driver.py` `tests/unit/test_app_smoke.py` `README.md`
- [ ] [PENDING] T6: Build `frontend/app_pages/lint.py` (pinned health-check message + approve/reject HITL) + `frontend/app_pages/fix.py` (pills quick-actions + approve/reject HITL) + extend `test_app_smoke.py` `frontend/app_pages/lint.py` `frontend/app_pages/fix.py` `tests/unit/test_app_smoke.py`

## Interface handoffs
<!-- Populated by the orchestrator from each task's handoff contract block. Do not pre-fill beyond the stub. -->
- (none yet)
