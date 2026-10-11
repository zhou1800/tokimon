"""Executable discovery and logger installation shared by tool runtimes."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO


def resolve_executable(command: str, env_var: str, env: Mapping[str, str]) -> str | None:
    search_path = env.get("PATH")
    override = env.get(env_var)
    if override:
        resolved = shutil.which(override, path=search_path)
        if resolved:
            return resolved
        override_path = Path(override).expanduser()
        if override_path.is_file() and os.access(override_path, os.X_OK):
            return str(override_path)
        return None
    return shutil.which(command, path=search_path)


def configure_logger(
    name: str,
    formatter: Callable[[], logging.Formatter],
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if not getattr(logger, "_tokimon_configured", False) or stream is not None:
        logger.handlers.clear()
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(formatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        setattr(logger, "_tokimon_configured", True)
    return logger


def format_log_record(
    record: logging.LogRecord,
    *,
    timestamp: str,
    service: str,
    runtime: str,
    fields: tuple[str, ...],
    format_exception: Callable[[Any], str],
    native_fields: tuple[str, ...] = (),
) -> str:
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "service": service,
        "runtime": runtime,
        "severity": record.levelname.lower(),
        "message": record.getMessage(),
        "correlation_id": getattr(record, "correlation_id", "server"),
    }
    for key in fields:
        value = getattr(record, key, None)
        if value is not None:
            payload[key] = value if key in native_fields else str(value)
    if record.exc_info:
        payload["exception"] = format_exception(record.exc_info)
    return json.dumps(payload, ensure_ascii=False)
