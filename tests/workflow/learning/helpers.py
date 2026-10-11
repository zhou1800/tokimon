from __future__ import annotations

from pathlib import Path

from tokimon.workflow.learning import artifacts


def reactive_candidate(*, packet_id: str = "reactive-tool-failure") -> dict[str, object]:
    return {
        "packet_id": packet_id,
        "learning_mode": "reactive",
        "trigger": "tool_failure",
        "pressure_scenario": "A broker command fails while a repository review is blocking product-development.",
        "before_observation": "The browser broker health check reported an unreachable CDP endpoint.",
        "after_observation": "The browser broker health check succeeded after the bind address was corrected.",
        "delta_summary": "The reusable lesson is to inspect broker bind addresses before changing callers.",
        "candidate_guidance": "When a broker health check cannot reach CDP, verify the session process bind address first.",
        "acceptance_expectations": "A product-development run can turn the evidence into an activated broker troubleshooting capability.",
        "scope": "remote browser broker session startup failures",
        "negative_scope": "not a general Docker networking rule",
        "claim_boundary": "Observed on a Tokimon remote-browser-broker session image.",
        "candidate_review_status": "pending",
        "operator_intake_status": "pending",
        "intake_package_status": "not_created",
        "reviewer_records": [],
        "operator_approval": {},
        "reactive_evidence": {
            "failure_type": "tool_failure",
            "failure_artifact": ".tokimon/workflow/product-development/learning-ability/repository-review.log",
            "failure_command": "browser_ensure",
            "failure_summary": "CDP health check failed before repository review could continue.",
            "vagueness_resolution": "The observed failure was a broker tool failure, not a product decision.",
        },
    }


def active_candidate(*, packet_id: str = "active-dreaming") -> dict[str, object]:
    return {
        "packet_id": packet_id,
        "learning_mode": "active",
        "trigger": "explicit_operator_learning_request",
        "pressure_scenario": "The operator asks Tokimon to learn a held-out capability from selected sources.",
        "before_observation": "Tokimon has no activated capability for the request.",
        "after_observation": "Tokimon can explain the source-backed pressure scenario and product-development handoff.",
        "delta_summary": "The reusable lesson is source-grounded product intent with negative scope.",
        "candidate_guidance": "Convert source-backed learning into a product-development intake package before activation.",
        "acceptance_expectations": "Product-development can activate the learned capability as a Skill, MCP, workflow, or existing-surface update.",
        "scope": "active learning requests with selected source spans",
        "negative_scope": "not open-ended whole-repository mining",
        "claim_boundary": "Applies only when exact source spans are present.",
        "candidate_review_status": "pending",
        "operator_intake_status": "pending",
        "intake_package_status": "not_created",
        "reviewer_records": [],
        "operator_approval": {},
        "active_evidence": {
            "operator_learning_request": "learn dreaming",
            "target_capability": "dreaming",
            "target_question": "How should Tokimon turn a learned capability into product behavior?",
            "selected_sources": ["/reference/repo/dreaming"],
            "source_spans": [
                {
                    "source": "/reference/repo/dreaming/README.md",
                    "span": "lines 10-20",
                    "summary": "The source describes pressure-tested capability activation.",
                }
            ],
            "baseline_pressure_scenario": "No activated capability exists for dreaming.",
            "before_after_delta": "Tokimon can now prepare a product-development intake package.",
        },
    }


def accepted(packet: dict[str, object]) -> dict[str, object]:
    artifacts.record_review(packet, reviewer_id="reviewer-a", decision="accept")
    artifacts.approve_intake(packet, operator_id="operator", approval_note="Approved for product-development intake.")
    return packet


def product_development_evidence(
    *,
    target_surface: str = "skill",
    activated_path: str = "skills/dreaming/skill.md",
) -> dict[str, object]:
    return {
        "acceptance_status": "accepted",
        "acceptance_summary": "Product-development accepted the activated Learning result.",
        "accepted_artifacts": [
            {
                "path": activated_path,
                "role": "activated_surface",
            }
        ],
        "verification_evidence": [
            {
                "kind": "unit_test",
                "reference": (
                    "python -m unittest discover -s tests/workflow/learning "
                    "-t . -p 'test_*.py'"
                ),
                "outcome": "passed",
            }
        ],
        "summary": "Accepted by product-development.",
        "target_surface": target_surface,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    artifacts.write_json(path, payload)
