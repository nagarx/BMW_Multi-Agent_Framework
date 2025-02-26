"""
Logging utility for the BMW Agents framework.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

import colorlog

# Constants
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Color scheme for different log levels
LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}


def setup_logger(
    name: str = "bmw_agents",
    level: int = DEFAULT_LOG_LEVEL,
    log_file: str = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Set up a logger with the given name and level.

    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        log_file: Path to log file (default: None, which means log to console only)
        use_colors: Whether to use colored output (default: True)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    if use_colors:
        formatter = colorlog.ColoredFormatter(
            DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT, log_colors=LOG_COLORS
        )
    else:
        formatter = logging.Formatter(
            DEFAULT_LOG_FORMAT.replace("%(log_color)s", ""), datefmt=DEFAULT_DATE_FORMAT
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create default logger
logger = setup_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Name of the logger

    Returns:
        Configured logger
    """
    return logging.getLogger(f"bmw_agents.{name}")


class OperationTimer:
    """Context manager for tracking operations and logging their start/end."""

    def __init__(self, logger: logging.Logger, operation_name: str) -> None:
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self) -> "OperationTimer":
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}")
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]
    ) -> None:
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.2f}s: {exc_val}")
