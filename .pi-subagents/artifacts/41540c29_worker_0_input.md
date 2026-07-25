# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement Phase 0 scaffolding for an agentic RAG project per the PLAN.md in the repo root.

Create the following:
1. `pyproject.toml` with deps: langchain>=0.3, langchain-openai>=0.2, langgraph>=0.2, pydantic>=2, pydantic-settings>=2, markitdown[all]>=0.0.1, markdown-it-py>=3.0, typer>=0.12, python-dotenv>=1.0. Dev deps: pytest>=8, pytest-asyncio>=0.23, respx>=0.21. requires-python >=3.11. Script: agentic-rag = "agentic_rag.cli:app"
2. `src/agentic_rag/__init__.py` (empty or version)
3. `src/agentic_rag/config.py` — Pydantic BaseSettings with fields from PLAN.md §5: openai_api_key (str), openai_base_url (default "https://api.openai.com/v1"), openai_model (default "gpt-4.1-mini"), wiki_path (Path, default ./wiki), raw_sources_path (Path, default ./raw), agents_md_path (Path, default ./AGENTS.md), recursion_limit (int, default 12), hitl_enabled (bool, default True), retrieval_mode (str, default "index"), vector_db_path (Path|None, default None), markitdown_llm_describe_images (bool, default False). Config: env_file=".env", env_file_encoding="utf-8".
4. `src/agentic_rag/paths.py` — path resolution helpers
5. Directory layout: src/agentic_rag/{schemas,io,tools,agents,middleware}/ each with __init__.py
6. `.env.example` per PLAN.md §5
7. `config/config.yaml.example`
8. `AGENTS.md` — default wiki schema per PLAN.md §6 (page types, naming conventions, cross-reference format, frontmatter spec, update rules, index entry format, log prefix format, hard rules)
9. `README.md` skeleton
10. `tests/conftest.py` (empty or basic fixtures)
11. `tests/fixtures/` directory
12. `raw/` directory with a `.gitkeep`
13. Update `.gitignore` if needed (add __pycache__, *.pyc, .env, .venv, etc.)

After creating all files, run `pip install -e .` and report success/failure.

Do NOT modify existing files (IDEA.md, PLAN.md, wiki/). Work from /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG

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