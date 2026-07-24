"""Typer CLI for agentic-rag: ingest, query, lint, status, log."""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(help="Agentic RAG — LLM-maintained wiki system")


@app.command()
def ingest(path: str = typer.Argument(..., help="Path to the source file to ingest")):
    """Ingest a source file into the wiki."""
    typer.echo(f"Ingesting: {path}")


@app.command()
def query(question: str = typer.Argument(..., help="Question to ask the wiki")):
    """Query the wiki for an answer with citations."""
    typer.echo(f"Querying: {question}")


@app.command()
def lint():
    """Run a health-check on the wiki."""
    typer.echo("Running wiki health check...")


@app.command()
def status():
    """Show wiki statistics."""
    typer.echo("Wiki status...")


@app.command()
def log(tail: int = typer.Option(10, "--tail", "-n", help="Number of log entries to show")):
    """Show recent log entries."""
    typer.echo(f"Last {tail} log entries...")
