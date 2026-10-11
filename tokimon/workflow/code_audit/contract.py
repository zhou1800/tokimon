"""Public workflow-package declarations for code audit."""

from __future__ import annotations

from pathlib import Path


WORKFLOW_ID = "code-audit"
SKILL_ID = "code-audit"
PYTHON_PACKAGE = "tokimon.workflow.code_audit"
WORKFLOW_STATUS = "implemented"

CANONICAL_ENTRYPOINT = "tokimon.workflow.code_audit.cli:main"
OPERATOR_ENTRYPOINT = ".venv/bin/python -m tokimon.workflow.code_audit"

JSON_ENCODING = "UTF-8"
JSON_ENSURE_ASCII = False

LOCAL_STATE_ROOT = Path(".tokimon") / "workflow" / WORKFLOW_ID
RUN_ROOT = LOCAL_STATE_ROOT / "runs"
LEDGER_ROOT = LOCAL_STATE_ROOT / "ledger"
CURRENT_LEDGER_PATH = LEDGER_ROOT / "current.json"
REPORT_DIR_NAME = "reports"
COMPILED_EVALUATIONS_NAME = "evaluations-from-reports.json"
REPORTS_SUMMARY_NAME = "reports-summary.json"
PASS_LOCKS_NAME = "pass-locks.json"
RECHECK_PLAN_NAME = "recheck-plan.json"

GUIDELINE_RULE_SOURCES = (
    Path("docs/guidelines/code-guidelines.md"),
    Path("docs/guidelines/unit-testing-guidelines.md"),
)

RULE_ID_PATTERN = r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}\b"
RULE_LINE_PATTERN = r"^\s*-\s+\*\*(?P<rule_id>[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)*-\d{3}):\*\*\s+(?P<description>.+?)\s*$"

COMMANDS = (
    "contract",
    "rules",
    "subjects",
    "matrix",
    "pass-locks",
    "recheck-plan",
    "verify-reports",
    "compile-reports",
    "record-findings",
    "ledger-summary",
)
AUDIT_STATUSES = ("pass", "fail", "not_applicable", "needs_human_review", "unchecked")
AUDIT_REPORT_STATUSES = ("pass", "fail", "not_applicable", "needs_human_review")
DURABLE_RECORD_STATUSES = ("fail", "needs_human_review")
OPEN_LEDGER_STATUSES = DURABLE_RECORD_STATUSES
RESOLVED_LEDGER_STATUSES = ("fixed", "stale")
LEDGER_STATUSES = (*OPEN_LEDGER_STATUSES, *RESOLVED_LEDGER_STATUSES)
RESOLUTION_STATUS_BY_EVALUATION = {"pass": "fixed", "not_applicable": "stale"}
REQUIRED_SUBJECT_FIELDS = ("audit_subject_id", "title")
SUBJECT_BOUNDARY_FIELDS = (
    "implementation_paths",
    "test_paths",
    "owning_specs",
    "verification_commands",
    "dependency_edges",
    "notes",
)
OPTIONAL_SUBJECT_FIELDS = (
    "test_paths",
    "owning_specs",
    "verification_commands",
    "dependency_edges",
    "notes",
    "subject_source",
)
EVIDENCE_LINE_FIELDS = ("line", "line_start", "start_line")
EVIDENCE_TEXT_FIELDS = ("text", "excerpt", "quote", "details")
CURRENT_LEDGER_SCHEMA_VERSION = 3
LEGACY_LEDGER_SCHEMA_VERSIONS = (1, 2)
RUN_ARTIFACT_SCHEMA_VERSION = 3
CHECKLIST_MATRIX_SCHEMA_VERSION = 3
LEGACY_CHECKLIST_MATRIX_SCHEMA_VERSIONS = (1, 2)
SUBJECTS_SCHEMA_VERSION = 1
EVALUATIONS_SCHEMA_VERSION = 3
AUDIT_REPORT_SCHEMA_VERSION = 3
LEGACY_AUDIT_REPORT_SCHEMA_VERSIONS = (1, 2)
PASS_LOCKS_SCHEMA_VERSION = 3
RECHECK_PLAN_SCHEMA_VERSION = 3

WORKFLOW_READINESS = {
    "id": WORKFLOW_ID,
    "workflow_status": WORKFLOW_STATUS,
    "entrypoint": OPERATOR_ENTRYPOINT,
    "modes": (
        {
            "id": "subject-matrix-ledger",
            "role": "primary",
            "summary": "Code audit validates audit subjects, writes rule/subject matrices, validates pass-lock recheck plans, verifies per-row structured reports, compiles row evaluations, and records auditor-accepted current row dispositions.",
            "required_tool_capabilities": (),
        },
    ),
}
