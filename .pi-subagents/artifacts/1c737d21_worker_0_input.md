# Task for worker

You are a delegated subagent running from a fork of the parent session. Treat the inherited conversation as reference-only context, not a live thread to continue. Do not continue or answer prior messages as if they are waiting for a reply. Your sole job is to execute the task below and return a focused result for that task using your tools.

Task:
Implement Phase 1 for an agentic RAG project at /Users/alvarojimenezmartinez/Proyectos/LangChain-RAG.

Read PLAN.md sections §7-§8 for full specs. Create these files:

## 1. `src/agentic_rag/schemas/wiki.py`
Pydantic models:
- `Frontmatter`: slug (str), type (str), title (str), sources (list[str]), updated (date), tags (list[str])
- `IndexEntry`: slug (str), summary (str), type (str), sources (list[str]), updated (date)
- `Index`: categories (dict[str, list[IndexEntry]]) — keys: entities, concepts, sources, comparisons, overviews
- `LogEntry`: timestamp (datetime), op (str), title (str), details (str)
- `Heading`: level (int), text (str)
- `Link`: target (str), alias (str|None)

## 2. `src/agentic_rag/schemas/agents_md.py`
- `load_agents_md(path: Path) -> str`: reads AGENTS.md file, returns content. If missing, return sensible default string matching §6.

## 3. `src/agentic_rag/schemas/extraction.py`
Pydantic models for structured output (used by ingest agent later):
- `Entity`: name (str), type (str), summary (str), sources (list[str])
- `Concept`: name (str), summary (str), related_entities (list[str])
- `Contradiction`: page_slug (str), existing_claim (str), new_claim (str), proposed_resolution (str)
- `ExtractionResult`: entities (list[Entity]), concepts (list[Concept]), contradictions (list[Contradiction])

## 4. `src/agentic_rag/io/source_loader.py`
Wrap markitdown.MarkItDown:
```python
class SourceLoader:
    def __init__(self, settings): ... # optionally init with llm_client if markitdown_llm_describe_images
    def load(self, source: str) -> str: ... # convert path to markdown
```

## 5. `src/agentic_rag/io/wiki_io.py`
Filesystem ops on wiki/:
- `list_pages(wiki_path) -> list[Path]`: recursive .md, excludes index.md/log.md
- `read_page(wiki_path, slug) -> str`: raw markdown
- `read_page_with_frontmatter(wiki_path, slug) -> tuple[Frontmatter, str]`: parse frontmatter
- `write_page(wiki_path, slug, content, frontmatter) -> Path`: atomic write (temp+rename), creates parent dirs, serializes frontmatter
- `delete_page(wiki_path, slug)`: remove file
- `page_exists(wiki_path, slug) -> bool`

## 6. `src/agentic_rag/io/markdown_parser.py`
Using markdown-it-py:
- `extract_links(content: str) -> list[Link]`: find all [[Target]] and [[Target|alias]]
- `extract_headings(content: str) -> list[Heading]`
- `parse_frontmatter(content: str) -> Frontmatter`: YAML frontmatter between --- delimiters
- `serialize_frontmatter(fm: Frontmatter) -> str`: serialize to YAML string with --- delimiters
- `slugify(name: str) -> str`: "3D Gaussian Splatting" → "3d-gaussian-splatting"

## 7. `src/agentic_rag/io/index_manager.py`
- `read_index(wiki_path) -> Index`: parse wiki/index.md into Index model
- `upsert_entry(wiki_path, entry: IndexEntry)`: add or update entry in correct category
- `remove_entry(wiki_path, slug: str)`: remove entry by slug
- `write_index(wiki_path, index: Index)`: atomic write
- `find_in_index(wiki_path, query: str) -> list[IndexEntry]`: substring/keyword match on summary

## 8. `src/agentic_rag/io/log_manager.py`
- `append_log(wiki_path, entry: LogEntry)`: append with prefix `## [YYYY-MM-DD HH:MM] <op> | <title>`
- `tail_log(wiki_path, n: int = 10) -> list[LogEntry]`: parse trailing entries

## 9. Tests — create ALL these test files in tests/unit/:

### tests/unit/test_source_loader.py
- Test loading a markdown file
- Test non-existent file raises error

### tests/unit/test_wiki_io.py
- write→read roundtrip
- list_pages excludes index.md/log.md
- page_exists
- delete_page

### tests/unit/test_markdown_parser.py
- extract_links finds [[A]] and [[A|alias]]
- extract_headings depth
- parse_frontmatter YAML roundtrip
- slugify tests

### tests/unit/test_index_manager.py
- Parse a sample index.md
- upsert adds entry
- remove_entry
- find_in_index keyword match

### tests/unit/test_log_manager.py
- append creates correct prefix
- tail returns last N entries

### tests/unit/test_agents_md_loader.py
- Loads existing AGENTS.md
- Returns default when missing

Use pytest fixtures. Create test fixtures in tests/fixtures/ as needed (sample wiki pages, sample index.md, sample log.md).

After creating all files, run `pytest tests/unit/ -v` and fix any failures until all pass.

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