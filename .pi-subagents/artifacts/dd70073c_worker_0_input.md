# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement comprehensive logging for the agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

## Requirements
- Track every tool call (name, args, result, duration, errors)
- Track agent outputs (what each agent returns)
- Track CLI commands (which command was invoked)
- Track HITL interrupts and decisions
- Log levels: DEBUG, INFO, WARNING, ERROR
- Structured logging to both console and file

## Implementation

### 1. Create `src/agentic_rag/logging_config.py`
Set up Python logging:
```python
import logging
import sys
from pathlib import Path

def setup_logging(log_dir: Path = None, level: str = "INFO"):
    """Configure logging for the agentic_rag project.
    
    Args:
        log_dir: Directory for log files. If None, logs to console only.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("agentic_rag")
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "agentic_rag.log")
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s.%(funcName)s:%(lineno)d: %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
```

### 2. Update `src/agentic_rag/middleware/logging.py`
Replace the simple print-based middleware with proper logging:
```python
import logging
import time
from langchain.agents.middleware import wrap_tool_call

logger = logging.getLogger("agentic_rag.tools")

@wrap_tool_call
def audit_logging_middleware(request, handler):
    """Log every tool call with args, result, and duration."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    
    logger.info(f"TOOL CALL: {tool_name}({tool_args})")
    start_time = time.time()
    
    try:
        result = handler(request)
        duration = time.time() - start_time
        logger.info(f"TOOL RESULT: {tool_name} completed in {duration:.3f}s")
        logger.debug(f"TOOL OUTPUT: {tool_name} -> {str(result)[:500]}")
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"TOOL ERROR: {tool_name} failed after {duration:.3f}s: {e}")
        raise
```

### 3. Update `src/agentic_rag/cli.py`
Add logging setup and command-level logging:
- At the start of each command, log the command and arguments
- At the end, log completion status
- On errors, log the full traceback

### 4. Update `src/agentic_rag/agents/factory.py`
Add logging when agents are created:
```python
logger = logging.getLogger("agentic_rag.agents")

def build_agent(...):
    logger.info(f"Building agent with model={model}, tools={len(tools)} tools")
    logger.debug(f"Tools: {[t.name for t in tools]}")
    # ... rest of function
```

### 5. Update `src/agentic_rag/agents/ingest.py`, `query.py`, `lint.py`
Add logging for agent-specific operations:
- Agent creation
- HITL interrupts
- Command resume decisions

### 6. Update `.env.example` and `config.py`
Add a LOG_LEVEL setting:
```python
# In config.py
log_level: str = "INFO"
log_dir: Path | None = None  # None = console only
```

### 7. Create `tests/unit/test_logging.py`
Test that logging works correctly:
- Test setup_logging creates handlers
- Test middleware logs tool calls
- Test log levels work

After implementing all changes, run `pytest tests/ -v` and ensure all tests pass.

Work from /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Implement the requested change without widening scope

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```