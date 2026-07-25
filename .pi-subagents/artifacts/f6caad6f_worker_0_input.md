# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement Phase 2 for an agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

Read PLAN.md §9 for full tool specs. Create these files:

## 1. `src/agentic_rag/tools/shared.py`
Shared tools used by multiple agents:
```python
from langchain_core.tools import tool

@tool
def read_index(wiki_path: str) -> str:
    """Read the wiki index.md and return its full content. Shows all entities, concepts, sources, and comparisons with summaries."""
    # calls index_manager.read_index, returns formatted string

@tool
def read_wiki_page(wiki_path: str, slug: str) -> str:
    """Read a wiki page by slug. Returns the full markdown content including frontmatter. Use this to get detailed information about any entity, concept, or source."""
    # calls wiki_io.read_page, returns content

@tool
def search_index(wiki_path: str, query: str) -> str:
    """Search the wiki index by keyword. Returns matching entries with their slugs, types, and summaries. Use to find relevant pages before reading them."""
    # calls index_manager.find_in_index, returns formatted results
```

## 2. `src/agentic_rag/tools/ingest_tools.py`
Tools for the ingest agent:
```python
@tool
def read_source(source_path: str) -> str:
    """Read and convert a source file to markdown using MarkItDown. Supports pdf, docx, pptx, xlsx, html, csv, json, xml, ipynb, images, epub, and more."""
    # calls SourceLoader.load

@tool
def create_page(wiki_path: str, slug: str, page_type: str, title: str, content: str, sources: list[str] = [], tags: list[str] = []) -> str:
    """Create a new wiki page. Fails if the page already exists. Use update_page for existing pages."""
    # calls wiki_io.write_page with new Frontmatter

@tool
def update_page(wiki_path: str, slug: str, content: str, sources: list[str] = [], tags: list[str] = []) -> str:
    """Update an existing wiki page. Preserves frontmatter fields unless explicitly changed. Fails if the page does not exist."""
    # calls wiki_io.read_page_with_frontmatter then wiki_io.write_page

@tool
def delete_wiki_page(wiki_path: str, slug: str) -> str:
    """Delete a wiki page. This action requires human approval (HITL). Will be paused for confirmation."""
    # calls wiki_io.delete_page

@tool
def update_index(wiki_path: str, slug: str, page_type: str, summary: str, sources: list[str] = []) -> str:
    """Update the wiki index with a new or modified entry. Call this after creating or updating a page."""
    # calls index_manager.upsert_entry

@tool
def append_log(wiki_path: str, op: str, title: str, details: str = "") -> str:
    """Append an entry to the wiki log.md. Use op='ingest' for source ingestion, 'query' for queries, 'lint' for health checks."""
    # calls log_manager.append_log

@tool
def flag_contradiction(wiki_path: str, page_slug: str, existing_claim: str, new_claim: str, proposed_resolution: str) -> str:
    """Flag a contradiction between existing wiki content and new source material. This action requires human approval (HITL). Will be paused for decision."""
    # Returns contradiction details for HITL handling
```

## 3. `src/agentic_rag/tools/query_tools.py`
Tools for the query agent (read-only):
```python
@tool
def find_relevant_pages(wiki_path: str, query: str) -> str:
    """Find wiki pages relevant to a query. Combines index search with link traversal. Returns a list of slugs to read."""
    # Search index + follow [[links]] to find related pages
```
Plus import read_index, read_wiki_page, search_index from shared.

## 4. `src/agentic_rag/tools/lint_tools.py`
Tools for the lint agent:
```python
@tool
def read_all_pages(wiki_path: str) -> str:
    """Read ALL wiki pages. Returns a dict of slug→content. Use sparingly - this is expensive for large wikis."""
    # calls wiki_io.list_pages then reads each

@tool
def find_inbound_links(wiki_path: str, slug: str) -> str:
    """Find all pages that link to a given slug via [[slug]] or [[slug|alias]] syntax. Use to detect orphan pages."""
    # grep all pages for [[slug]]

@tool
def extract_concepts(content: str) -> str:
    """Extract concept names from page content. Returns headings and [[link]] targets found in the content."""
    # calls markdown_parser.extract_headings + extract_links

@tool
def write_lint_report(wiki_path: str, report: str) -> str:
    """Write a lint report to wiki/lint-report-YYYY-MM-DD.md with today's date."""
    # writes report file
```

## 5. `src/agentic_rag/tools/__init__.py`
Export all tools for easy import.

## 6. `tests/unit/test_tools.py`
Test each tool against a temp wiki fixture:
- read_index: returns formatted index content
- read_wiki_page: reads existing page, errors on missing
- search_index: keyword match returns results
- read_source: loads a markdown file
- create_page: creates new page, errors if exists
- update_page: updates existing, errors if missing
- delete_wiki_page: removes file
- update_index: adds entry to index
- append_log: appends with correct prefix
- find_relevant_pages: returns matching slugs
- read_all_pages: returns all pages
- find_inbound_links: finds [[link]] references
- write_lint_report: creates report file

Use pytest fixtures with a temporary wiki directory. Each tool takes wiki_path as first arg.

After creating all files, run `pytest tests/unit/ -v` and fix any failures.

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