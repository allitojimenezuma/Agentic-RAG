# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Fix two issues in the agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG:

## Issue 1: Duplicate logs
The same info is logged in multiple places (middleware + tools + IO). Consolidate:
- **middleware/logging.py**: Keep as-is — logs tool call name, args, duration, result
- **tools/*.py**: REMOVE all logging — the middleware already covers this
- **io/*.py**: Keep DEBUG-level logs only — these are internal implementation details

So the fix is: remove `logger.info(...)` calls from `src/agentic_rag/tools/shared.py`, `src/agentic_rag/tools/ingest_tools.py`, `src/agentic_rag/tools/query_tools.py`, `src/agentic_rag/tools/lint_tools.py`. Keep the `logger = logging.getLogger(__name__)` line and DEBUG logs if any, but remove INFO logs that duplicate what middleware already logs.

## Issue 2: Agent hallucinates pages it didn't read
The query agent says it consulted `apple-silicon`, `modernbert`, `álvaro-jiménez-martínez` but only actually read `mlx`. Fix the query agent prompt to be strict about citations.

Update `src/agentic_rag/agents/prompts.py` — the `build_query_prompt` function:

Change the workflow step 5 to:
```
5. Return markdown answer with inline citations ONLY for pages you actually read using read_wiki_page. Do NOT cite pages you only saw in the index or in [[links]] — you must call read_wiki_page for each page you cite. At the end, list only the pages you actually read in the "Sources Consulted" table.
```

And add this rule:
```
- CRITICAL: Only cite pages you actually called read_wiki_page on. Never infer or hallucinate content from page names alone.
```

After fixing, run `pytest tests/ -q` to ensure all tests pass, then commit with message "fix: remove duplicate tool logs and prevent hallucinated citations" and push to origin.

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