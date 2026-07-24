"""Typer CLI for agentic-rag: ingest, query, lint, status, log."""

from __future__ import annotations

from uuid import uuid4

import typer
from langgraph.types import Command

app = typer.Typer(help="Agentic RAG — LLM Wiki CLI")


@app.command()
def ingest(path: str = typer.Argument(..., help="Path to the source file to ingest")):
    """Ingest a source file into the wiki. HITL prompts inline."""
    from pathlib import Path

    from agentic_rag.config import Settings

    # Validate source file exists
    if not Path(path).is_file():
        typer.echo(f"Error: Source file not found: {path}", err=True)
        raise typer.Exit(1)

    try:
        settings = Settings()
    except Exception as e:
        typer.echo(f"Error loading settings: {e}", err=True)
        raise typer.Exit(1)

    from agentic_rag.agents.ingest import build_ingest_agent

    agent = build_ingest_agent(settings)
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": settings.recursion_limit,
    }

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"Ingest {path}"}]}, config=config
        )
    except Exception as e:
        typer.echo(f"Error during ingestion: {e}", err=True)
        raise typer.Exit(1)

    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]
        typer.echo(f"\n⏸ Interrupt: {interrupt}")
        decision = typer.prompt("Decision (approve/reject/edit)")

        if decision == "approve":
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}]}), config=config
            )
        elif decision == "reject":
            feedback = typer.prompt("Feedback (optional)", default="")
            result = agent.invoke(
                Command(
                    resume={"decisions": [{"type": "reject", "feedback": feedback}]}
                ),
                config=config,
            )
        elif decision == "edit":
            typer.echo("Edit not yet supported in CLI. Approving instead.")
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}]}), config=config
            )
        else:
            typer.echo("Invalid decision. Please enter approve, reject, or edit.")

    typer.echo(result["messages"][-1].content)


@app.command()
def query(question: str = typer.Argument(..., help="Question to ask the wiki")):
    """Query the wiki (read-only)."""
    try:
        from agentic_rag.config import Settings

        settings = Settings()
    except Exception as e:
        typer.echo(f"Error loading settings: {e}", err=True)
        raise typer.Exit(1)

    from agentic_rag.agents.query import build_query_agent

    agent = build_query_agent(settings)
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": settings.recursion_limit,
    }

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}, config=config
        )
    except Exception as e:
        typer.echo(f"Error during query: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(result["messages"][-1].content)


@app.command()
def lint():
    """Run wiki health check. Writes report to wiki/lint-report-YYYY-MM-DD.md."""
    try:
        from agentic_rag.config import Settings

        settings = Settings()
    except Exception as e:
        typer.echo(f"Error loading settings: {e}", err=True)
        raise typer.Exit(1)

    from agentic_rag.agents.lint import build_lint_agent

    agent = build_lint_agent(settings)
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": settings.recursion_limit,
    }

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Run a full wiki health check. Report orphans, contradictions, missing links, and data gaps.",
                    }
                ]
            },
            config=config,
        )
    except Exception as e:
        typer.echo(f"Error during lint: {e}", err=True)
        raise typer.Exit(1)

    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]
        typer.echo(f"\n⏸ Interrupt: {interrupt}")
        decision = typer.prompt("Decision (approve/reject)")

        if decision == "approve":
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}]}), config=config
            )
        else:
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "reject"}]}), config=config
            )

    typer.echo(result["messages"][-1].content)


@app.command()
def status():
    """Show wiki status: page counts, last log entry, quick orphan scan."""
    try:
        from agentic_rag.config import Settings

        settings = Settings()
    except Exception as e:
        typer.echo(f"Error loading settings: {e}", err=True)
        raise typer.Exit(1)

    from agentic_rag.io.index_manager import read_index
    from agentic_rag.io.log_manager import tail_log
    from agentic_rag.io.wiki_io import list_pages

    pages = list_pages(settings.wiki_path)
    index = read_index(settings.wiki_path)
    last_log = tail_log(settings.wiki_path, 1)

    typer.echo(f"Wiki pages: {len(pages)}")
    total_entries = sum(len(v) for v in index.categories.values())
    typer.echo(f"Index entries: {total_entries}")
    if last_log:
        entry = last_log[0]
        typer.echo(f"Last log: [{entry.timestamp}] {entry.op} | {entry.title}")


@app.command(name="log")
def log_cmd(
    tail: int = typer.Option(10, help="Number of log entries to show"),
):
    """Tail the wiki log."""
    try:
        from agentic_rag.config import Settings

        settings = Settings()
    except Exception as e:
        typer.echo(f"Error loading settings: {e}", err=True)
        raise typer.Exit(1)

    from agentic_rag.io.log_manager import tail_log

    entries = tail_log(settings.wiki_path, tail)
    for entry in entries:
        typer.echo(f"[{entry.timestamp}] {entry.op} | {entry.title}")
        if entry.details:
            typer.echo(f"  {entry.details}")


if __name__ == "__main__":
    app()
