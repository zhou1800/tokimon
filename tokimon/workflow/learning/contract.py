"""Public workflow-package declarations for Learning."""

from __future__ import annotations

from pathlib import Path


PRODUCT_ID = "learning"
SKILL_ID = "learning"
WORKFLOW_ID = "learning"
PYTHON_PACKAGE = "tokimon.workflow.learning"
WORKFLOW_STATUS = "implemented"

CANONICAL_ENTRYPOINT = "tokimon.workflow.learning.cli:main"
OPERATOR_ENTRYPOINT = ".venv/bin/python -m tokimon.workflow.learning"

LOCAL_STATE_ROOT = Path(".tokimon") / "workflow" / WORKFLOW_ID
RUN_ROOT = LOCAL_STATE_ROOT / "runs"
INTAKE_PACKAGE_ROOT = LOCAL_STATE_ROOT / "intake-packages"

REQUIRED_TOOL_CAPABILITIES: tuple[str, ...] = ()

COMMANDS = (
    "contract",
    "start-reactive",
    "start-active",
    "start-acceptance",
    "create-reactive-candidate",
    "create-active-candidate",
    "validate-reactive-candidate",
    "validate-active-candidate",
    "record-review",
    "approve-intake",
    "create-intake-package",
    "record-activation",
    "validate-acceptance",
)

VALID_RUN_MODES = ("reactive", "active", "acceptance")
VALID_RUN_STATUSES = ("initialized", "in_progress", "accepted", "incomplete", "blocked")
VALID_CANDIDATE_STATUSES = ("pending", "accepted", "rejected")
VALID_REVIEW_DECISIONS = ("accept", "reject")
VALID_OPERATOR_INTAKE_STATUSES = ("pending", "approved", "rejected")
VALID_INTAKE_PACKAGE_STATUSES = ("not_created", "package_created", "superseded")
VALID_TARGET_SURFACES = ("skill", "mcp", "skill_and_mcp", "workflow", "existing_surface_update")

REACTIVE_TRIGGER_VALUES = (
    "user_correction",
    "tool_failure",
    "test_failure",
    "code_review_finding",
    "document_review_finding",
)
REACTIVE_EXCLUDED_TRIGGER_VALUES = (
    "product_decision",
    "general_repo_browsing",
    "high_star_repo_mining",
    "startup_memory_observation",
)
ACTIVE_EXCLUDED_TRIGGER_VALUES = (
    "open_ended_repo_learning",
    "github_score_as_authority",
    "high_star_repo_mining",
    "general_repo_browsing",
    "code_copy_without_use_case",
    "guidance_without_pressure_scenario",
    "guidance_without_source_spans",
)

REQUIRED_CANDIDATE_FIELDS = (
    "packet_id",
    "learning_mode",
    "trigger",
    "pressure_scenario",
    "before_observation",
    "after_observation",
    "delta_summary",
    "candidate_guidance",
    "acceptance_expectations",
    "scope",
    "negative_scope",
    "claim_boundary",
    "candidate_review_status",
    "operator_intake_status",
    "intake_package_status",
    "reviewer_records",
    "operator_approval",
)
REQUIRED_REACTIVE_EVIDENCE_FIELDS = (
    "failure_type",
    "failure_artifact",
    "failure_command",
    "failure_summary",
    "vagueness_resolution",
)
REQUIRED_ACTIVE_EVIDENCE_FIELDS = (
    "operator_learning_request",
    "target_capability",
    "target_question",
    "selected_sources",
    "source_spans",
    "baseline_pressure_scenario",
    "before_after_delta",
)
REQUIRED_ACTIVATION_EVIDENCE_FIELDS = (
    "activation_id",
    "intake_id",
    "target_surface",
    "activated_path",
    "product_development_evidence",
    "activation_summary",
    "created_at",
)
REQUIRED_PRODUCT_DEVELOPMENT_EVIDENCE_FIELDS = (
    "acceptance_status",
    "acceptance_summary",
    "accepted_artifacts",
    "verification_evidence",
)
REQUIRED_ACCEPTED_ARTIFACT_FIELDS = ("path", "role")
REQUIRED_VERIFICATION_EVIDENCE_FIELDS = ("kind", "reference", "outcome")

RUN_ID_PATTERN = r"^learning-\d{8}T\d{6}Z-[a-z0-9]+(?:-[a-z0-9]+)*$"
REQUIRED_INITIAL_ARTIFACTS = ("state.json", "acceptance-plan.json", "artifacts.json")
REQUIRED_DIRECTORIES = (
    "candidates",
    "validation-reports",
    "intake-packages",
    "activation-records",
)
REQUIRED_STATE_FIELDS = (
    "run_id",
    "workflow_id",
    "mode",
    "status",
    "trigger",
    "target_capability",
    "source",
    "created_at",
    "updated_at",
    "artifact_manifest",
    "acceptance_plan",
)

WORKFLOW_READINESS = {
    "id": WORKFLOW_ID,
    "workflow_status": WORKFLOW_STATUS,
    "entrypoint": OPERATOR_ENTRYPOINT,
    "modes": (
        {
            "id": "artifact-support",
            "role": "primary",
            "summary": "Learning deterministic candidate, intake package, activation, and acceptance support commands can run.",
            "required_tool_capabilities": (),
            "agent_connector_capabilities": (),
        },
    ),
}
