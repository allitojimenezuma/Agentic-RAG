# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement Phase 3 for an agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

Read PLAN.md §9 for agent specs. Read the langchain-fundamentals and langchain-middleware skills at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG/.agents/skills/ for correct create_agent and HITL syntax.

Create these files:

## 1. `src/agentic_rag/agents/factory.py`
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

def build_agent(model: str, tools: list, system_prompt: str, middleware: list = None):
    """Build a LangChain agent with create_agent."""
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware or [],
        checkpointer=MemorySaver(),
    )
```

## 2. `src/agentic_rag/agents/prompts.py`
Build system prompts, injecting AGENTS.md content:
```python
def build_ingest_prompt(agents_md: str) -> str:
    """Build system prompt for ingest agent."""
    return f"""You are the Ingest Agent for a persistent LLM-maintained wiki.

# Wiki Schema
{agents_md}

# Workflow
1. Call read_source(path) to get the source markdown.
2. Identify entities and concepts. For each:
   a. search_index(name) and read_wiki_page(slug) to check existing coverage.
   b. If a page exists and the source changes a factual claim that CONFLICTS with the page → call flag_contradiction(page_slug, existing_claim, new_claim, proposed_resolution). Wait for the human decision before writing.
   c. Otherwise create_page (new) or update_page (exists, non-conflicting update).
3. Update every Related section and add cross-links [[Page]].
4. Create a source summary page under sources/<slug>.md.
5. Call update_index for all created/updated pages.
6. Call append_log with op="ingest", title=source name, details=list of pages touched.

# Hard rules
- Never write outside wiki/.
- Never modify raw/.
- Never delete a page without the delete_wiki_page tool (HITL).
- Never ignore a contradiction — always call flag_contradiction.
- Always end by updating index and log."""

def build_query_prompt(agents_md: str) -> str:
    """Build system prompt for query agent."""
    return f"""You are the Query Agent for a persistent LLM wiki.
# Wiki Schema
{agents_md}

# Workflow
1. read_index → identify candidate pages.
2. search_index(question) → augment candidates.
3. read_wiki_page for each candidate; follow cross-links as needed.
4. Synthesize an answer with inline citations: "claim ([[/page-slug]])."
5. Return markdown answer + a "Sources consulted" list (page slugs + source titles).

# Rules
- Read-only. Do not call any write tool (none provided).
- If the wiki does not cover the question, say so explicitly and suggest sources to ingest."""

def build_lint_prompt(agents_md: str) -> str:
    """Build system prompt for lint agent."""
    return f"""You are the Lint Agent. Audit wiki health and WRITE A REPORT. Default to suggestions, not deletions.
# Wiki Schema
{agents_md}

# Workflow
1. read_all_pages + read_index.
2. For each page: find_inbound_links → detect orphans.
3. For each [[X]] link with no target file → missing page.
4. Compare overlapping claims across pages → contradictions / stale claims.
5. Suggest missing cross-references and data gaps (new questions/sources to investigate).
6. write_lint_report(report) to wiki/lint-report-YYYY-MM-DD.md.
7. If a page is clearly empty/duplicate and must be removed, call delete_wiki_page (HITL).

# Rules
- Prefer reporting over mutation.
- Never modify content pages directly.
- Cite page slugs + line references in findings."""
```

## 3. `src/agentic_rag/agents/ingest.py`
```python
from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_ingest_prompt
from agentic_rag.tools.ingest_tools import read_source, create_page, update_page, delete_wiki_page, update_index, append_log, flag_contradiction
from agentic_rag.tools.shared import read_index, read_wiki_page, search_index
from agentic_rag.schemas.agents_md import load_agents_md
from langchain.agents.middleware import HumanInTheLoopMiddleware

def build_ingest_agent(settings):
    agents_md = load_agents_md(settings.agents_md_path)
    tools = [read_source, read_index, search_index, read_wiki_page, create_page, update_page, delete_wiki_page, update_index, append_log, flag_contradiction]
    middleware = [
        HumanInTheLoopMiddleware(interrupt_on={
            "delete_wiki_page": {"allowed_decisions": ["approve", "reject"]},
            "flag_contradiction": {"allowed_decisions": ["approve", "edit", "reject"]},
        })
    ]
    return build_agent(
        model=settings.openai_model,
        tools=tools,
        system_prompt=build_ingest_prompt(agents_md),
        middleware=middleware,
    )
```

## 4. `src/agentic_rag/agents/query.py`
```python
from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_query_prompt
from agentic_rag.tools.query_tools import find_relevant_pages
from agentic_rag.tools.shared import read_index, read_wiki_page, search_index
from agentic_rag.schemas.agents_md import load_agents_md

def build_query_agent(settings):
    agents_md = load_agents_md(settings.agents_md_path)
    tools = [read_index, search_index, read_wiki_page, find_relevant_pages]
    return build_agent(
        model=settings.openai_model,
        tools=tools,
        system_prompt=build_query_prompt(agents_md),
    )
```

## 5. `src/agentic_rag/agents/lint.py`
```python
from agentic_rag.agents.factory import build_agent
from agentic_rag.agents.prompts import build_lint_prompt
from agentic_rag.tools.lint_tools import read_all_pages, find_inbound_links, extract_concepts, write_lint_report
from agentic_rag.tools.shared import read_index
from agentic_rag.schemas.agents_md import load_agents_md
from langchain.agents.middleware import HumanInTheLoopMiddleware

def build_lint_agent(settings):
    agents_md = load_agents_md(settings.agents_md_path)
    tools = [read_all_pages, read_index, find_inbound_links, extract_concepts, write_lint_report]
    middleware = [
        HumanInTheLoopMiddleware(interrupt_on={
            "delete_wiki_page": {"allowed_decisions": ["approve", "reject"]},
        })
    ]
    return build_agent(
        model=settings.openai_model,
        tools=tools,
        system_prompt=build_lint_prompt(agents_md),
        middleware=middleware,
    )
```

## 6. `src/agentic_rag/middleware/logging.py`
```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def audit_logging_middleware(request, handler):
    """Log every tool call with args to stdout."""
    print(f"[TOOL CALL] {request.tool_call['name']}({request.tool_call.get('args', {})})")
    result = handler(request)
    print(f"[TOOL RESULT] {request.tool_call['name']} -> {str(result)[:200]}")
    return result
```

## 7. `src/agentic_rag/middleware/guardrails.py`
```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def path_guard_middleware(request, handler):
    """Reject any tool call that tries to write outside wiki_path or touch raw/."""
    # Check for path arguments that might escape
    args = request.tool_call.get("args", {})
    for key in ["source_path", "path", "file_path"]:
        if key in args:
            val = str(args[key])
            if "raw/" in val or ".." in val:
                return f"ERROR: Path {val} is outside allowed wiki directory"
    return handler(request)
```

## 8. `src/agentic_rag/middleware/__init__.py`
Export middleware components.

## 9. `src/agentic_rag/agents/__init__.py`
Export agent builders.

## 10. Integration tests in `tests/integration/`
Create a FakeChatModel in `tests/fixtures/fake_llm.py` that returns scripted AIMessages with tool_calls.

Then create these integration tests:

### tests/integration/test_ingest_agent.py
- Feed a small raw/sample.md
- FakeChatModel scripted to call read_source → search_index → create_page ×2 → update_index → append_log
- Assert: pages created on disk, index updated, log entry appended

### tests/integration/test_query_agent.py  
- Ask "What is MLX?"
- FakeChatModel calls read_index → read_wiki_page("mlx") → returns answer with [[MLX]] citation
- Assert: no write tool called

### tests/integration/test_lint_agent.py
- Point at fixture wiki with orphan page
- FakeChatModel calls read_all_pages → find_inbound_links → write_lint_report
- Assert: report file exists

After creating all files, run `pytest tests/ -v` and fix failures. All unit tests (93) + new integration tests should pass.

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