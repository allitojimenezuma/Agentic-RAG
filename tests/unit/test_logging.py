"""Tests for logging configuration."""

import logging

from agentic_rag.logging_config import setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = setup_logging(level="INFO")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "agentic_rag"

    def test_log_level_is_respected(self):
        """Test that log level setting works."""
        logger = setup_logging(level="WARNING")
        assert logger.level == logging.WARNING

    def test_log_level_case_insensitive(self):
        """Test that log level is case-insensitive."""
        logger = setup_logging(level="debug")
        assert logger.level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self):
        """Test that invalid level defaults to INFO."""
        logger = setup_logging(level="INVALID")
        assert logger.level == logging.INFO

    def test_no_duplicate_handlers(self):
        """Test that calling setup_logging twice doesn't add duplicate handlers."""
        logger1 = setup_logging(level="INFO")
        handler_count = len(logger1.handlers)
        logger2 = setup_logging(level="INFO")
        assert len(logger2.handlers) == handler_count
