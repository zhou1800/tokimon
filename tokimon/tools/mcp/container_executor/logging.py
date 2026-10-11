"""Shared metadata-only startup/request/job logging."""

import logging
from datetime import datetime, timezone

from tokimon.tools._runtime import configure_logger, format_log_record


class LogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return format_log_record(record, timestamp=datetime.now(timezone.utc).isoformat(),
                                 service="container-executor", runtime="mcp",
                                 fields=("job_id", "stage", "error_type", "errno", "exit_code"),
                                 format_exception=self.formatException)


def logger() -> logging.Logger:
    return configure_logger("tokimon.container-executor", LogFormatter)


def log_failure(stage: str, correlation_id: str, error: Exception) -> None:
    logger().error("startup_failed", extra={"stage": stage, "correlation_id": correlation_id,
                   "error_type": type(error).__name__, "errno": getattr(error, "errno", None),
                   "exit_code": getattr(error, "returncode", None)})
