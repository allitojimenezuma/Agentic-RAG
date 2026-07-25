"""Logging configuration for the agentic_rag project."""

import logging
import sys
from pathlib import Path


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

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

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
