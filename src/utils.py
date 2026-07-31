"""Shared utility helpers: logging, reproducibility, and JSON I/O."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.config import LOGS_DIR, RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed all relevant random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create (or fetch) a configured logger that logs to stdout and a file.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    log_file:
        Optional file name (relative to ``LOGS_DIR``) to also write logs to.
        Defaults to ``pipeline.log``.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured (e.g. re-imported in a notebook).
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOGS_DIR / (log_file or "pipeline.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def save_json(data: dict[str, Any], path: Path) -> None:
    """Persist a dictionary to disk as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
