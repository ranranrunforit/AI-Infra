"""
Structured logging utilities for the operator.
"""

import logging
import sys
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'training_job'):
            log_data['training_job'] = record.training_job
        if hasattr(record, 'namespace'):
            log_data['namespace'] = record.namespace
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation

        return json.dumps(log_data)


class SimpleFormatter(logging.Formatter):
    """
    Simple human-readable formatter for development.
    """

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging(level: str = 'INFO', structured: bool = True) -> None:
    """
    Configure logging for the operator.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Use structured JSON logging if True, simple format if False
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Set formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = SimpleFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set specific loggers
    logging.getLogger('kopf').setLevel(logging.WARNING)
    logging.getLogger('kubernetes').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class OperatorLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context about training jobs.
    """

    def process(self, msg, kwargs):
        """Add extra context to log records."""
        extra = kwargs.get('extra', {})

        if 'training_job' in self.extra:
            extra['training_job'] = self.extra['training_job']
        if 'namespace' in self.extra:
            extra['namespace'] = self.extra['namespace']

        kwargs['extra'] = extra
        return msg, kwargs
