"""Structured logging configuration for JRO."""

import logging
import sys


def setup_logger(
    name: str = "jro",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger namespace.
        level: Logging level (default INFO).
        log_file: Optional path to append log entries.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
