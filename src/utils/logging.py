"""
src/utils/logging.py
--------------------
Standardised logging setup for the pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    suppress_third_party: bool = True,
) -> None:
    """
    Configure root logger with console (and optional file) handlers.

    Args:
        level: Python logging level string ('DEBUG', 'INFO', 'WARNING', etc.).
        log_file: If provided, also write logs to this file.
        suppress_third_party: Silence noisy third-party loggers.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    if suppress_third_party:
        for noisy in ("urllib3", "pymongo", "gurobipy"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
