"""Pytest configuration for benchmark tasks.

Benchmark task `tests/` directories are meant to be copied into an isolated task
workspace (see `benchmarks/harness.py`) where the corresponding `starter/` code
lives at the workspace root.

Collecting these task tests in-place from the repository can raise import errors
(e.g., `from summary import ...` expects `summary.py` at the project root) and can
break `pytest` runs over the repo.

This conftest prevents in-repo collection of benchmark task tests; the harness
still copies and runs them in task workspaces.
"""

from __future__ import annotations

collect_ignore_glob = ["*/tests/*"]
