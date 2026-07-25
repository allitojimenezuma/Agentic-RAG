# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement Phase 5 for an agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

Read PLAN.md §12.3 for acceptance test specs.

## 1. Create `tests/acceptance/test_wiki_health.py`
```python
import pytest
import os
from agentic_rag.config import Settings
from agentic_rag.agents.lint import build_lint_agent

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY")
def test_lint_real_wiki():
    """Run lint agent over real wiki/ — must complete without exceptions."""
    settings = Settings()
    agent = build_lint_agent(settings)
    config = {"configurable": {"thread_id": "acceptance-lint"}, "recursion_limit": settings.recursion_limit}
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Run a full wiki health check."}]
    }, config=config)
    
    # Should complete without exceptions
    assert result["messages"][-1].content
    assert len(result["messages"]) > 1
```

## 2. Create `tests/acceptance/test_ingest_real_source.py`
```python
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from agentic_rag.config import Settings
from agentic_rag.agents.ingest import build_ingest_agent
from agentic_rag.io.wiki_io import list_pages
from agentic_rag.io.index_manager import read_index

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY")
def test_ingest_real_source():
    """Ingest raw/cv.pdf into a temp copy of wiki/ — verify pages created."""
    # Create temp copy of wiki
    with tempfile.TemporaryDirectory() as tmp:
        wiki_copy = Path(tmp) / "wiki"
        shutil.copytree(Path("wiki"), wiki_copy)
        
        # Create a settings with temp wiki path
        settings = Settings(wiki_path=wiki_copy)
        agent = build_ingest_agent(settings)
        config = {"configurable": {"thread_id": "acceptance-ingest"}, "recursion_limit": settings.recursion_limit}
        
        # Find a raw source to ingest
        raw_sources = list(Path("raw").glob("*.md"))
        if not raw_sources:
            pytest.skip("No raw sources available")
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"Ingest {raw_sources[0]}"}]}
        , config=config)
        
        # Should complete
        assert result["messages"][-1].content
        
        # Check pages were created
        pages = list_pages(wiki_copy)
        assert len(pages) > 0
```

## 3. Update README.md with usage instructions
Read the current README.md and add:
- Installation: `pip install -e .`
- Configuration: copy .env.example to .env, set OPENAI_API_KEY
- Usage examples for each CLI command
- Architecture overview
- Testing instructions

## 4. Error handling pass
Check all modules for proper error handling:
- Source loader: handle file not found gracefully
- Wiki IO: handle permission errors
- Index/Log managers: handle malformed files
- CLI: handle missing .env gracefully

## 5. Run all tests
After all changes, run `pytest tests/ -v` and ensure everything passes.

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