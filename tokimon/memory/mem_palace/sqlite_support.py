"""SQLite capability checks shared by Mem Palace store adapters."""

from __future__ import annotations

import sqlite3


_MISSING_FTS_ERRORS = {
    "no such module: fts5",
    "no such table: memory_fts",
    "no such table: main.memory_fts",
    "no such function: bm25",
}


def is_missing_fts_error(error: sqlite3.OperationalError) -> bool:
    """Return whether SQLite reports a specifically unsupported FTS surface."""

    return str(error).strip().lower() in _MISSING_FTS_ERRORS
