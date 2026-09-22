"""Logging. Rich on the console, plain UTF-8 to the project log file."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["setup_logging", "get_logger", "console", "add_job_log"]

console = Console(stderr=False, soft_wrap=False)
_CONFIGURED = False


def setup_logging(level: str = "INFO", *, quiet: bool = False) -> None:
    global _CONFIGURED
    root = logging.getLogger("visualresearcher")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not _CONFIGURED:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
            omit_repeated_times=False,
            markup=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    for handler in root.handlers:
        if isinstance(handler, RichHandler):
            handler.setLevel(logging.CRITICAL if quiet else logging.DEBUG)


def add_job_log(path: Path) -> logging.Handler:
    """Tee the log into ``projects/<name>/job.log`` so a run is auditable later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    logging.getLogger("visualresearcher").addHandler(handler)
    return handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"visualresearcher.{name}")
