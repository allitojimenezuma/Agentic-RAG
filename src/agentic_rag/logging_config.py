"""Logging configuration for the agentic_rag project."""

import logging
import sys
from pathlib import Path


# Color codes per module for console output
_COLORS = {
    "agentic_rag.cli":          "\033[36m",   # cyan
    "agentic_rag.agents":       "\033[35m",   # magenta
    "agentic_rag.agents.ingest":"\033[35m",   # magenta
    "agentic_rag.agents.query": "\033[35m",   # magenta
    "agentic_rag.agents.lint":  "\033[35m",   # magenta
    "agentic_rag.tools":        "\033[33m",   # yellow
    "agentic_rag.tools.shared": "\033[33m",   # yellow
    "agentic_rag.tools.ingest_tools": "\033[93m",  # bright yellow
    "agentic_rag.tools.lint_tools":   "\033[93m",  # bright yellow
    "agentic_rag.io":           "\033[32m",   # green
    "agentic_rag.io.wiki_io":   "\033[32m",   # green
    "agentic_rag.io.index": "\033[32m",  # green
    "agentic_rag.io.log":   "\033[32m",  # green
    "agentic_rag.io.source_loader": "\033[32m",  # green
    "agentic_rag.tokens":       "\033[34m",   # blue
}
_RESET = "\033[0m"
_DEFAULT_COLOR = "\033[37m"  # white fallback


class ColoredFormatter(logging.Formatter):
    """Console formatter that colors log lines by logger name."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Find matching color by longest prefix
        color = _DEFAULT_COLOR
        name = record.name
        for prefix in sorted(_COLORS, key=len, reverse=True):
            if name == prefix or name.startswith(prefix + "."):
                color = _COLORS[prefix]
                break

        level = record.levelname
        msg = record.getMessage()

        # Color the level tag too
        level_colors = {
            "DEBUG":    "\033[37m",    # white/dim
            "INFO":     "\033[32m",    # green
            "WARNING":  "\033[33m",    # yellow
            "ERROR":    "\033[31m",    # red
            "CRITICAL": "\033[41m",    # red bg
        }
        level_color = level_colors.get(level, _RESET)

        return (
            f"[{self.formatTime(record, self.datefmt)}] "
            f"{level_color}{level:<8}\033[0m "
            f"{color}{record.name}\033[0m: "
            f"{msg}"
        )


def setup_logging(log_dir: Path = None, level: str = "INFO") -> logging.Logger:
    """Configure logging for the agentic_rag project.

    Args:
        log_dir: Directory for log files. If None, logs to console only.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        The configured root logger for agentic_rag.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("agentic_rag")
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger

    # Colored console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
    logger.addHandler(console_handler)

    # File handler (plain, no colors)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "agentic_rag.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s.%(funcName)s:%(lineno)d: %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger
