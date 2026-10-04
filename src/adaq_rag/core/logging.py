"""Application logging configuration."""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure structured console logging for the application.

    Args:
        log_level: Log severity string (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: The configured application logger.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger = logging.getLogger("adaq_rag")
    logger.setLevel(numeric_level)
    return logger
