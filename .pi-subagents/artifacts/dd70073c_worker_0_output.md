Done. Pushed to GitHub.

## Summary

**Files changed:**
- `src/agentic_rag/logging_config.py` (new)
- `src/agentic_rag/middleware/logging.py`
- `src/agentic_rag/cli.py`
- `src/agentic_rag/agents/factory.py`
- `src/agentic_rag/agents/ingest.py`
- `src/agentic_rag/agents/query.py`
- `src/agentic_rag/agents/lint.py`
- `src/agentic_rag/config.py`
- `.env.example`
- `tests/unit/test_logging.py` (new)

**Test results:** 115 passed, 2 skipped

**How to use:**

```bash
# Console only (default)
agentic-rag query "What is MLX?"

# With file logging
LOG_LEVEL=DEBUG LOG_DIR=./logs agentic-rag query "What is MLX?"
```

**Log levels:**
- `DEBUG`: Tool outputs, agent details, tool args
- `INFO`: Tool calls with duration, agent creation, command invocations
- `WARNING`: Non-critical issues
- `ERROR`: Tool failures, agent errors with full tracebacks

```