"""Logging configuration for ML scripts and the backend.

Uses the standard :mod:`logging` module rather than ``print`` so that verbosity
is controllable, records carry timestamps and module names, and output can be
redirected to a file for a long unattended training run on Lightning.

Diagnostic scripts that exist purely to display a report to a human still print
their formatted tables, which is intentional: that output *is* the deliverable.
Everything else logs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

__all__ = ["configure_logging", "get_logger"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LEVEL_ENV_VAR = "QAS_LOG_LEVEL"
_DEFAULT_LEVEL = "INFO"

_configured = False


def configure_logging(
    level: str | int | None = None,
    *,
    log_file: Path | None = None,
    force: bool = False,
) -> None:
    """Configure root logging once per process.

    Args:
        level: Log level name or numeric value. When ``None``, the
            ``QAS_LOG_LEVEL`` environment variable is used, defaulting to
            ``INFO``.
        log_file: Optional file to receive a copy of all log records. Its parent
            directory is created if needed. Used for long training runs so the
            log survives a disconnected session.
        force: Reconfigure even if this function already ran. Off by default so
            that importing a module cannot clobber an application's logging
            setup.

    Raises:
        ValueError: If ``level`` is not a recognized level name.
    """
    global _configured
    if _configured and not force:
        return

    if level is None:
        level = os.environ.get(_LEVEL_ENV_VAR, "").strip() or _DEFAULT_LEVEL

    if isinstance(level, str):
        resolved = logging.getLevelNamesMapping().get(level.upper())
        if resolved is None:
            raise ValueError(
                f"Unknown log level {level!r}. Expected one of: "
                f"{', '.join(sorted(logging.getLevelNamesMapping()))}."
            )
        level = resolved

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        The requested :class:`logging.Logger`.
    """
    return logging.getLogger(name)
