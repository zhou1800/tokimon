"""Module entrypoint for document audit workflow support."""

from __future__ import annotations

from tokimon.workflow.document_audit.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
