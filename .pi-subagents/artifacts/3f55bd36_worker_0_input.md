# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Fix Phase 1 issues at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

Read PLAN.md §10 for index/log formats. Fix these issues:

## Blocker 1: Path traversal in wiki_io.py
`wiki_path / f"{slug}.md"` with slug `../../etc/passwd` resolves outside wiki_path.
Fix: validate slug at io layer. Add a `_validate_slug(slug)` function that:
- Rejects slugs containing `..`, `/`, `\`
- Rejects absolute paths
- Raises ValueError on invalid slugs
Call it in read_page, write_page, delete_page, page_exists, read_page_with_frontmatter.

## Blocker 2: index_manager.py section name parsing
`section_name.rstrip("s")` produces `"entitie"` instead of `"entity"` for "## Entities".
Fix: use proper singularization mapping:
```python
_SECTION_TO_TYPE = {
    "Entities": "entity",
    "Concepts": "concept", 
    "Sources": "source",
    "Comparisons": "comparison",
    "Overviews": "overview",
}
```

## Note 1: Source entry format
Source entries should use `sources/[slug].md` path format per §10, not `[slug].md`.

## Note 2: Display name casing
`_format_entry` re-derives display name from slug via `.title()`, losing original casing.
Fix: store original name in IndexEntry or preserve casing from the parsed entry.

After fixing, run `pytest tests/unit/ -v` and ensure all 57 tests still pass. Add new tests for the slug validation if needed.

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