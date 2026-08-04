"""Typer CLI for agentic-rag: ingest, query, lint, status, log."""

from __future__ import annotations

import logging
import traceback
from uuid import uuid4

import typer
from langgraph.types import Command

app = typer.Typer(help="Agentic RAG — LLM Wiki CLI")
logger = logging.getLogger("agentic_rag.cli")


@app.command()
def ingest(input_text: str = typer.Argument(..., help="File path to ingest OR natural language to update/create")):
    """Ingest a source file or update/create pages from natural language."""
    from pathlib import Path

    from agentic_rag.config import Settings
    from agentic_rag.logging_config import setup_logging

    settings = Settings()
    setup_logging(log_dir=settings.log_dir, level=settings.log_level)

    logger.info(f"INGEST command invoked: input={input_text}")

    # Auto-detect mode: file path vs natural language
    input_path = Path(input_text)
    if input_path.is_file():
        user_message = f"Ingest {input_text}"
        logger.info("Mode: file ingest")
    else:
        user_message = input_text
        logger.info("Mode: update/create from text")

    try:
        settings = Settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        typer.echo(f"Error loading settings: {e}", err=True)
        raise typer.Exit(1)

    from agentic_rag.agents.ingest import build_ingest_agent

    agent = build_ingest_agent(settings)
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": settings.ingest_recursion_limit,
    }

    try:
        logger.info("Invoking ingest agent")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]}, config=config
        )
        logger.info("Ingest agent completed")
        if hasattr(agent, "_token_tracker"):
            agent._token_tracker.log_summary()
    except Exception as e:
        logger.error(f"Ingest agent failed: {e}\n{traceback.format_exc()}")
        typer.echo(f"Error during ingestion: {e}", err=True)
        raise typer.Exit(1)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        if isinstance(interrupts, (list, tuple)):
            interrupt = interrupts[0]
        else:
            interrupt = interrupts
        logger.info(f"HITL interrupt: {interrupt}")

        raw = getattr(interrupt, "value", interrupt)
        try:
            actions = raw.get("action_requests", []) if hasattr(raw, "get") else []
        except Exception:
            actions = []

        # Display pending actions
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  {len(actions)} pending action{'s' if len(actions) > 1 else ''}")
        typer.echo(f"{'='*60}")
        for i, action in enumerate(actions, 1):
            name = action.get("name", "unknown") if isinstance(action, dict) else "unknown"
            args = action.get("args", {}) if isinstance(action, dict) else {}
            desc = args.get("page_slug", args.get("slug", "")) if isinstance(args, dict) else ""
            typer.echo(f"  [{i}] {name}({desc})")
        typer.echo(f"{'='*60}")
        typer.echo("  [a]pprove all  [e]dit one  [r]eject all")

        decision = typer.prompt("Decision")
        logger.info(f"HITL decision: {decision}")

        if decision in ("a", "approve"):
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}] * len(actions) if actions else [{"type": "approve"}]}),
                config=config,
            )
        elif decision in ("r", "reject"):
            feedback = typer.prompt("Feedback (optional)", default="")
            result = agent.invoke(
                Command(
                    resume={"decisions": [{"type": "reject", "feedback": feedback}] * len(actions) if actions else [{"type": "reject", "feedback": feedback}]}
                ),
                config=config,
            )
        elif decision in ("e", "edit"):
            if len(actions) == 1:
                idx = 0
            else:
                idx = int(typer.prompt(f"Which action to edit [1-{len(actions)}]")) - 1
            target = actions[idx] if isinstance(actions[idx], dict) else {}
            target_args = target.get("args", {}) if isinstance(target, dict) else {}
            typer.echo(
                f"Current proposed_resolution: {target_args.get('proposed_resolution', '')}"
            )
            new_resolution = typer.prompt("New resolution")
            decisions = [{"type": "approve"}] * len(actions)
            decisions[idx] = {
                "type": "edit",
                "edited_action": {
                    "name": target.get("name", "flag_contradiction") if isinstance(target, dict) else "flag_contradiction",
                    "args": {**target_args, "proposed_resolution": new_resolution},
                },
            }
            result = agent.invoke(
                Command(resume={"decisions": decisions}),
                config=config,
            )
        else:
            typer.echo("Invalid. Use a/e/r.")

    logger.info("INGEST command finished")
    typer.echo(result["messages"][-1].content)


@app.command()
def query(question: str = typer.Argument(..., help="Question to ask the wiki")):
    """Query the wiki (read-only)."""
    from agentic_rag.config import Settings
    from agentic_rag.logging_config import setup_logging

    settings = Settings()
    setup_logging(log_dir=settings.log_dir, level=settings.log_level)

    logger.info(f"QUERY command invoked: question={question}")

    from agentic_rag.agents.query import build_query_agent

    agent = build_query_agent(settings)
    config = {
        "configurable": {"thread_id": str(uuid4())},
        "recursion_limit": settings.recursion_limit,
    }

    try:
        logger.info("Invoking query agent")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}, config=config
        )
        logger.info("Query agent completed")
        # Log token usage summary
        if hasattr(agent, "_token_tracker"):
            agent._token_tracker.log_summary()
    except Exception as e:
        logger.error(f"Query agent failed: {e}\n{traceback.format_exc()}")
        typer.echo(f"Error during query: {e}", err=True)
        raise typer.Exit(1)

    logger.info("QUERY command finished")

    from agentic_rag.schemas.query import QueryAnswer
    from agentic_rag.tools.grounding import validate_citations

    # Structured render: locate the LAST submit_query_answer ToolMessage
    # (its .content is the validated QueryAnswer JSON from the cite-or-die tool).
    nav_capture = getattr(agent, "_nav_capture", None)
    submit_message = None
    for msg in reversed(result["messages"]):
        if getattr(msg, "name", None) == "submit_query_answer":
            submit_message = msg
            break

    if submit_message is not None and nav_capture is not None:
        qa = QueryAnswer.model_validate_json(submit_message.content)
        # Belt-and-suspenders: re-validate against the turn's navigated set.
        qa = validate_citations(qa, nav_capture.navigated)
        out = f"Answer:\n{qa.answer}\n\nConfidence: {qa.confidence}"
        if qa.citations:
            out += "\nCitations:"
            for citation in qa.citations:
                section = f" (section: {citation.section})" if citation.section else ""
                out += f"\n- {citation.slug} - {citation.title}{section}"
        if qa.suggestion:
            out += f"\nSuggestion: {qa.suggestion}"
        typer.echo(out)
    else:
        # Compat fallback: fake/plain agents (no submit tool call, no
        # _nav_capture) render the raw final message.
        typer.echo(result["messages"][-1].content)


@app.command()
def lint():
    """Run wiki health check. Writes report to wiki/lint-report-YYYY-MM-DD.md."""
    from agentic_rag.config import Settings
    from agentic_rag.logging_config import setup_logging

    settings = Settings()
    setup_logging(log_dir=settings.log_dir, level=settings.log_level)

    logger.info("LINT command invoked")

    from agentic_rag.agents.lint import build_lint_agent

    agent = build_lint_agent(settings)
    config = {
        "configurable": {"thread_id": str(uuid4())}
    }

    try:
        logger.info("Invoking lint agent")
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
        logger.info("Lint agent completed")
        # Log token usage summary
        if hasattr(agent, "_token_tracker"):
            agent._token_tracker.log_summary()
    except Exception as e:
        logger.error(f"Lint agent failed: {e}\n{traceback.format_exc()}")
        typer.echo(f"Error during lint: {e}", err=True)
        raise typer.Exit(1)

    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]
        logger.info(f"HITL interrupt: {interrupt}")
        typer.echo(f"\n⏸ Interrupt: {interrupt}")
        decision = typer.prompt("Decision (approve/reject)")

        logger.info(f"HITL decision: {decision}")
        if decision == "approve":
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}]}), config=config
            )
        else:
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "reject"}]}), config=config
            )

    logger.info("LINT command finished")
    typer.echo(result["messages"][-1].content)


@app.command()
def fix(
    issue: str = typer.Argument(
        ..., help="Optional filter (issue kind or page slug) or 'latest' to run health_check"
    )
):
    """Fix wiki lint issues found by the deterministic health check."""
    from agentic_rag.config import Settings
    from agentic_rag.logging_config import setup_logging

    settings = Settings()
    setup_logging(log_dir=settings.log_dir, level=settings.log_level)

    logger.info(f"FIX command invoked: issue={issue}")

    from agentic_rag.agents.fix import build_fix_agent
    from agentic_rag.lint.health import health_check

    # Run the deterministic health check -> structured LintReport.
    try:
        report = health_check(settings.wiki_path)
        issues = report.issues
        filter_mismatch = False
        if issue and issue != "latest":
            needle = issue.lower()
            issues = [
                i for i in issues
                if needle in i.kind.lower() or needle in i.slug.lower()
            ]
            if not issues and report.issues:
                # A filter that matches nothing is NOT a clean bill of health:
                # the user asked to fix and real issues exist. Warn, then fall
                # back to fixing all of them instead of sending "No issues".
                filter_mismatch = True
                issues = report.issues
                warning = (
                    f"Warning: no issues matched '{issue}'; "
                    f"fixing all {len(issues)} issues."
                )
                typer.echo(warning)
                logger.warning(warning)
        if issues:
            lines = ["Fix these lint issues:"]
            if filter_mismatch:
                lines.insert(
                    0,
                    f"No issues matched filter '{issue}' — fixing all {len(issues)} issues.",
                )
            for i in issues:
                lines.append(f"- [{i.kind}] {i.slug}: {i.detail}")
            user_message = "\n".join(lines)
        else:
            user_message = "No issues"
    except Exception as e:
        logger.warning(f"Health check failed, falling back to 'No issues': {e}")
        user_message = "No issues"

    agent = build_fix_agent(settings)
    config = {
        "configurable": {"thread_id": str(uuid4())}
    }

    try:
        logger.info("Invoking fix agent")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
        )
        logger.info("Fix agent completed")
        if hasattr(agent, "_token_tracker"):
            agent._token_tracker.log_summary()
    except Exception as e:
        logger.error(f"Fix agent failed: {e}\n{traceback.format_exc()}")
        typer.echo(f"Error during fix: {e}", err=True)
        raise typer.Exit(1)

    while "__interrupt__" in result:
        interrupts = result["__interrupt__"]
        # result["__interrupt__"] is a list/tuple of Interrupt objects
        if isinstance(interrupts, (list, tuple)):
            interrupt = interrupts[0]
        else:
            interrupt = interrupts

        # Interrupt.value is a dict with action_requests
        raw = getattr(interrupt, "value", interrupt)
        try:
            actions = raw.get("action_requests", []) if hasattr(raw, "get") else []
        except Exception:
            actions = []

        # Display pending actions (approve/reject only — no edit-command path)
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  {len(actions)} pending action{'s' if len(actions) > 1 else ''}")
        typer.echo(f"{'='*60}")
        for i, action in enumerate(actions, 1):
            name = action.get("name", "unknown") if isinstance(action, dict) else "unknown"
            args = action.get("args", {}) if isinstance(action, dict) else {}
            desc = args.get("slug", args.get("page_slug", "")) if isinstance(args, dict) else ""
            typer.echo(f"  [{i}] {name}({desc})")
        typer.echo(f"{'='*60}")
        typer.echo("  [a]pprove all  [r]eject all")

        decision = typer.prompt("Decision")
        logger.info(f"HITL decision: {decision}")

        if decision in ("a", "approve"):
            result = agent.invoke(
                Command(resume={"decisions": [{"type": "approve"}] * len(actions) if actions else [{"type": "approve"}]}),
                config=config,
            )
        elif decision in ("r", "reject"):
            feedback = typer.prompt("Feedback (optional)", default="")
            result = agent.invoke(
                Command(
                    resume={"decisions": [{"type": "reject", "feedback": feedback}] * len(actions) if actions else [{"type": "reject", "feedback": feedback}]}
                ),
                config=config,
            )
        else:
            typer.echo("Invalid. Use a/r.")

    logger.info("FIX command finished")
    typer.echo(result["messages"][-1].content)


@app.command()
def status():
    """Show wiki status: page counts, last log entry, quick orphan scan."""
    from agentic_rag.config import Settings
    from agentic_rag.logging_config import setup_logging

    settings = Settings()
    setup_logging(log_dir=settings.log_dir, level=settings.log_level)

    logger.info("STATUS command invoked")

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

    logger.info("STATUS command finished")


@app.command(name="log")
def log_cmd(
    tail: int = typer.Option(10, help="Number of log entries to show"),
):
    """Tail the wiki log."""
    from agentic_rag.config import Settings
    from agentic_rag.logging_config import setup_logging

    settings = Settings()
    setup_logging(log_dir=settings.log_dir, level=settings.log_level)

    logger.info(f"LOG command invoked: tail={tail}")

    from agentic_rag.io.log_manager import tail_log

    entries = tail_log(settings.wiki_path, tail)
    for entry in entries:
        typer.echo(f"[{entry.timestamp}] {entry.op} | {entry.title}")
        if entry.details:
            typer.echo(f"  {entry.details}")

    logger.info("LOG command finished")


if __name__ == "__main__":
    app()
