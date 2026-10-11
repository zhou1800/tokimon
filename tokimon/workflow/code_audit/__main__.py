"""Module entrypoint for `python -m tokimon.workflow.code_audit`."""

from __future__ import annotations

from tokimon.workflow.code_audit.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
