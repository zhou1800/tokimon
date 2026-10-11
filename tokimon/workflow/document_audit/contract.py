"""Public workflow-package declarations for document audit."""

from __future__ import annotations

from pathlib import Path


WORKFLOW_ID = "document-audit"
SKILL_ID = "document-audit"
PYTHON_PACKAGE = "tokimon.workflow.document_audit"
WORKFLOW_STATUS = "implemented"

CANONICAL_ENTRYPOINT = "tokimon.workflow.document_audit.cli:main"
OPERATOR_ENTRYPOINT = ".venv/bin/python -m tokimon.workflow.document_audit"

JSON_ENCODING = "UTF-8"
JSON_ENSURE_ASCII = False

LOCAL_STATE_ROOT = Path(".tokimon") / "workflow" / WORKFLOW_ID
RUN_ROOT = LOCAL_STATE_ROOT / "runs"
LEDGER_ROOT = LOCAL_STATE_ROOT / "ledger"
CURRENT_LEDGER_PATH = LEDGER_ROOT / "current.json"
REPORT_DIR_NAME = "reports"
COMPILED_EVALUATIONS_NAME = "evaluations-from-reports.json"
REPORTS_SUMMARY_NAME = "reports-summary.json"
AUDIT_UNITS_NAME = "audit-units.json"
CLEAN_UNIT_LOCKS_NAME = "clean-unit-locks.json"
RECHECK_PLAN_NAME = "recheck-plan.json"

GUIDELINE_RULE_SOURCES = (
    Path("docs/guidelines/document-guidelines.md"),
    Path("docs/guidelines/c4-document-guidelines.md"),
)

RULE_ID_PATTERN = r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}\b"
RULE_LINE_PATTERN = r"^\s*-\s+\*\*(?P<rule_id>[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}):\*\*\s+(?P<description>.+?)\s*$"

COMMANDS = (
    "contract",
    "rules",
    "matrix",
    "clean-unit-locks",
    "recheck-plan",
    "verify-reports",
    "compile-reports",
    "ledger-summary",
    "record-findings",
)
FINDING_STATUSES = ("open", "fixed", "deferred", "rejected", "blocked", "stale")
TERMINAL_FINDING_STATUSES = ("deferred", "rejected", "blocked")
FINDING_SEVERITIES = ("must_fix", "should_fix", "optional")
REQUIRED_FINDING_FIELDS = ("rule_id", "file_path", "summary", "severity", "status", "evidence")
EVIDENCE_LINE_FIELDS = ("line", "start_line")
EVIDENCE_TEXT_FIELDS = ("text", "excerpt", "quote", "details")
CHECKLIST_STATUSES = ("pass", "fail", "not_applicable", "unchecked")
AUDIT_REPORT_STATUSES = ("pass", "fail", "not_applicable")
CURRENT_LEDGER_SCHEMA_VERSION = 3
LEGACY_LEDGER_SCHEMA_VERSIONS = (1, 2)
RUN_ARTIFACT_SCHEMA_VERSION = 3
CHECKLIST_MATRIX_SCHEMA_VERSION = 3
LEGACY_CHECKLIST_MATRIX_SCHEMA_VERSIONS = (1, 2)
AUDIT_REPORT_SCHEMA_VERSION = 3
LEGACY_AUDIT_REPORT_SCHEMA_VERSIONS = (1, 2)
AUDIT_UNITS_SCHEMA_VERSION = 1
CLEAN_UNIT_LOCKS_SCHEMA_VERSION = 1
RECHECK_PLAN_SCHEMA_VERSION = 1

WORKFLOW_READINESS = {
    "id": WORKFLOW_ID,
    "workflow_status": WORKFLOW_STATUS,
    "entrypoint": OPERATOR_ENTRYPOINT,
    "modes": (
        {
            "id": "matrix-reports-ledger",
            "role": "primary",
            "summary": (
                "Document audit can load live guideline rules, validate audit-unit recheck coverage, "
                "compile row evaluations, and update a compact current ledger from auditor-accepted records."
            ),
            "required_tool_capabilities": (),
        },
    ),
}
