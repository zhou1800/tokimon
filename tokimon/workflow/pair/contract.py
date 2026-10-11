"""Public workflow-package declarations for Pair."""

from __future__ import annotations

from pathlib import Path


PRODUCT_ID = "pair"
SKILL_ID = "pair"
WORKFLOW_ID = "pair"
PYTHON_PACKAGE = "tokimon.workflow.pair"
WORKFLOW_STATUS = "support"

CANONICAL_ENTRYPOINT = "tokimon.workflow.pair.cli:main"
OPERATOR_ENTRYPOINT = ".venv/bin/python -m tokimon.workflow.pair"

LOCAL_STATE_ROOT = Path(".tokimon") / "workflow" / WORKFLOW_ID
RUN_ROOT = LOCAL_STATE_ROOT / "runs"
SETTINGS_SECTION = "pair"
DEFAULT_MAX_ROUNDS = 5

NATIVE_SUBAGENT_CAPABILITY = "native-subagent"
LOCAL_CLI_CAPABILITY = "local-cli"
CODEX_CLAUDE_PEER_CAPABILITIES = (
    NATIVE_SUBAGENT_CAPABILITY,
    LOCAL_CLI_CAPABILITY,
)
PEER_CAPABILITIES = CODEX_CLAUDE_PEER_CAPABILITIES
CODEX_CLAUDE_PEER_AGENTS = ("codex", "claude")
PEER_A_AGENT = "codex"
PEER_B_AGENTS = ("claude", "codex")

REQUIRED_TOOL_CAPABILITIES: tuple[str, ...] = ()

COMMANDS = ("contract", "init", "select-peers", "status", "validate")

VALID_STATUSES = (
    "initialized",
    "in_progress",
    "paused_external_clarification",
    "consensus",
    "no_consensus",
    "blocked",
)

TASK_BEHAVIORS = (
    "read_only",
    "shared_mutable_output",
    "unique_irreversible_side_effect",
    "independent_outputs",
)

TERMINAL_OUTCOMES = ("consensus", "no_consensus")

ROUND_SIGNALS = (
    "continue",
    "consensus",
    "no_consensus",
    "external_clarification",
    "blocked",
)

PAUSE_REASONS = ("external_clarification",)
EXTERNAL_CLARIFICATION_FIELDS = ("status", "requested_by", "question")

REQUIRED_INITIAL_ARTIFACTS = ("state.json", "transcript.md", "artifacts.json")
REQUIRED_DIRECTORIES = ("prompts", "rounds", "peers/a", "peers/b")
ROUND_NOTE_FILES = ("peer-a.md", "peer-b.md")
CONSENSUS_ARTIFACT = "consensus.md"
NO_CONSENSUS_ARTIFACT = "no_consensus.md"
RUN_ID_PATTERN = r"^pair-\d{8}T\d{6}Z-[a-z0-9]+(?:-[a-z0-9]+)*$"

REQUIRED_STATE_FIELDS = (
    "run_id",
    "workflow_id",
    "task_summary",
    "status",
    "created_at",
    "updated_at",
    "current_round_index",
    "peers",
    "peer_selection_config",
    "peer_selection_rationale",
    "task_behavior",
    "active_prompt_artifact",
    "transcript_artifact",
    "artifact_manifest",
    "external_clarification",
    "consensus_status",
    "final_delivery_bundle",
)

NO_CONSENSUS_FIELDS = (
    "unresolved_issue",
    "peer_a_position",
    "peer_b_position",
    "peer_a_evidence",
    "peer_b_evidence",
    "human_decision_needed",
    "continuation_options",
)

REQUIRED_CONSENSUS_STATUS_FIELDS = (
    "result",
    "rounds_completed",
    "final_text_artifact",
    "final_text_sha256",
    "independent_notes",
    "exchange",
    "peer_acceptances",
)

REQUIRED_CONSENSUS_INDEPENDENT_NOTE_FIELDS = (
    "round",
    "peer_a_artifact",
    "peer_b_artifact",
)

REQUIRED_CONSENSUS_EXCHANGE_FIELDS = (
    "round",
    "peer_a_received_artifact",
    "peer_b_received_artifact",
    "peer_a_response_artifact",
    "peer_b_response_artifact",
)

REQUIRED_CONSENSUS_PEER_ACCEPTANCE_FIELDS = (
    "accepted",
    "final_text_artifact",
    "final_text_sha256",
)

REQUIRED_MUTABLE_WORK_FIELDS = (
    "writer_peer",
    "reviewer_peer",
    "write_round",
    "review_round",
    "writer_evidence_artifact",
    "reviewer_evidence_artifact",
    "reviewer_approved",
)

WORKFLOW_READINESS = {
    "id": WORKFLOW_ID,
    "workflow_status": WORKFLOW_STATUS,
    "entrypoint": OPERATOR_ENTRYPOINT,
    "modes": (
        {
            "id": "artifact-support",
            "role": "primary",
            "summary": "Pair workflow deterministic state and artifact support commands can run.",
            "required_tool_capabilities": (),
            "agent_connector_capabilities": (),
        },
    ),
}
