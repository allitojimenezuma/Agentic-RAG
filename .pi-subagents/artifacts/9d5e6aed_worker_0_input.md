# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement Phase 4 for an agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

Read PLAN.md §11 for CLI specs. Read the langchain-middleware skill at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG/.agents/skills/langchain-middleware/SKILL.md for correct Command(resume=...) syntax.

## 1. Update `src/agentic_rag/cli.py`
Implement full Typer CLI with 5 commands:

```python
import typer
from uuid import uuid4
from pathlib import Path
from langgraph.types import Command
from agentic_rag.config import Settings

app = typer.Typer(help="Agentic RAG — LLM Wiki CLI")

@app.command()
def ingest(path: str):
    """Ingest a source file into the wiki. HITL prompts inline."""
    settings = Settings()
    from agentic_rag.agents.ingest import build_ingest_agent
    agent = build_ingest_agent(settings)
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": settings.recursion_limit}
    
    result = agent.invoke({"messages": [{"role": "user", "content": f"Ingest {path}"}]}, config=config)
    
    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]
        # Present interrupt to user
        print(f"\n⏸ Interrupt: {interrupt}")
        decision = input("Decision (approve/reject/edit): ").strip().lower()
        
        if decision == "approve":
            result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
        elif decision == "reject":
            feedback = input("Feedback (optional): ").strip()
            result = agent.invoke(Command(resume={"decisions": [{"type": "reject", "feedback": feedback}]}), config=config)
        elif decision == "edit":
            print("Edit not yet supported in CLI. Approving instead.")
            result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
        else:
            print("Invalid decision. Please enter approve, reject, or edit.")
    
    print(result["messages"][-1].content)

@app.command()
def query(question: str):
    """Query the wiki (read-only)."""
    settings = Settings()
    from agentic_rag.agents.query import build_query_agent
    agent = build_query_agent(settings)
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": settings.recursion_limit}
    
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    print(result["messages"][-1].content)

@app.command()
def lint():
    """Run wiki health check. Writes report to wiki/lint-report-YYYY-MM-DD.md."""
    settings = Settings()
    from agentic_rag.agents.lint import build_lint_agent
    agent = build_lint_agent(settings)
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": settings.recursion_limit}
    
    result = agent.invoke({"messages": [{"role": "user", "content": "Run a full wiki health check. Report orphans, contradictions, missing links, and data gaps."}]}, config=config)
    
    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]
        print(f"\n⏸ Interrupt: {interrupt}")
        decision = input("Decision (approve/reject): ").strip().lower()
        if decision == "approve":
            result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
        else:
            result = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config=config)
    
    print(result["messages"][-1].content)

@app.command()
def status():
    """Show wiki status: page counts, last log entry, quick orphan scan."""
    settings = Settings()
    from agentic_rag.io.wiki_io import list_pages
    from agentic_rag.io.log_manager import tail_log
    from agentic_rag.io.index_manager import read_index
    
    pages = list_pages(settings.wiki_path)
    index = read_index(settings.wiki_path)
    last_log = tail_log(settings.wiki_path, 1)
    
    print(f"Wiki pages: {len(pages)}")
    print(f"Index entries: {sum(len(v) for v in index.categories.values())}")
    if last_log:
        entry = last_log[0]
        print(f"Last log: [{entry.timestamp}] {entry.op} | {entry.title}")

@app.command()
def log(tail: int = typer.Option(10, help="Number of log entries to show")):
    """Tail the wiki log."""
    settings = Settings()
    from agentic_rag.io.log_manager import tail_log
    
    entries = tail_log(settings.wiki_path, tail)
    for entry in entries:
        print(f"[{entry.timestamp}] {entry.op} | {entry.title}")
        if entry.details:
            print(f"  {entry.details}")

if __name__ == "__main__":
    app()
```

## 2. Update `src/agentic_rag/main.py`
Simple entry point that calls app().

## 3. Create `tests/integration/test_cli.py`
Test CLI with CliRunner:
```python
from typer.testing import CliRunner
from agentic_rag.cli import app

runner = CliRunner()

def test_status_command(tmp_path, wiki_fixture):
    # Set up wiki fixture
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Wiki pages:" in result.output

def test_log_command(tmp_path, wiki_fixture):
    result = runner.invoke(app, ["log", "--tail", "5"])
    assert result.exit_code == 0
```

Note: For ingest/query/lint commands that need LLM, test with FakeChatModel if possible, or skip with marker.

## 4. Run all tests
After creating all files, run `pytest tests/ -v` and fix failures.

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