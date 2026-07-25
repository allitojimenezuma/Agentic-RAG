# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Fix the broken imports in the agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

The tools were refactored to use factory functions (make_shared_tools, make_ingest_tools, etc.) instead of direct tool exports. Now tests and __init__.py have broken imports.

## Fix these files:

### 1. `src/agentic_rag/tools/__init__.py`
Update exports to use new factory functions:
```python
from agentic_rag.tools.shared import make_shared_tools
from agentic_rag.tools.ingest_tools import make_ingest_tools
from agentic_rag.tools.query_tools import make_query_tools
from agentic_rag.tools.lint_tools import make_lint_tools
```

### 2. `tests/unit/test_tools.py`
Update all tool tests. The tools now need wiki_path bound via factory functions. For each test:
- Create tools with `make_shared_tools(str(tmp_path))` or `make_ingest_tools(str(tmp_path))` etc.
- The tools no longer take wiki_path as an argument — it's already bound
- Call tools with just their remaining args (e.g., `tool_read_index()` instead of `tool_read_index(wiki_path=str(tmp_path))`)

### 3. `tests/integration/test_ingest_agent.py`
Update to use new tool factories. The FakeChatModel scripted tool calls need to match the new tool signatures (no wiki_path arg).

### 4. `tests/integration/test_query_agent.py`
Same — update tool imports and calls.

### 5. `tests/integration/test_lint_agent.py`
Same — update tool imports and calls.

### 6. `tests/integration/test_cli.py`
May need updates if it references old tool imports.

After fixing, run `pytest tests/ -q` and ensure all tests pass.

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