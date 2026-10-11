from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import active_candidate, accepted, reactive_candidate
from tokimon.workflow.learning import validation


class LearningValidationTests(unittest.TestCase):
    def test_intake_gate_rejects_malformed_review_and_truthy_approval(self) -> None:
        for records in ([None], [{}], [{"reviewer_id": "peer", "decision": "reject", "reviewed_at": "now"}]):
            with self.subTest(records=records):
                packet = accepted(reactive_candidate())
                packet["reviewer_records"] = records
                result = validation.validate_candidate_packet(packet, require_intake_gate=True)
                self.assertFalse(result["ok"], result)
                self.assertTrue(any("reviewer_records" in error for error in result["errors"]))
        for approval in ("false", "true", 1, None):
            with self.subTest(approved=approval):
                packet = accepted(reactive_candidate())
                packet["operator_approval"]["approved"] = approval
                result = validation.validate_candidate_packet(packet, require_intake_gate=True)
                self.assertFalse(result["ok"], result)
                self.assertIn("operator_approval must record approved=true before intake package creation", result["errors"])

    def test_active_evidence_rejects_empty_or_non_string_selected_sources(self) -> None:
        for sources in ([None], [{}], [""], ["   "], ["valid", None]):
            with self.subTest(sources=sources):
                packet = active_candidate()
                packet["active_evidence"]["selected_sources"] = sources
                result = validation.validate_candidate_packet(packet)
                self.assertFalse(result["ok"], result)
                self.assertTrue(any("selected_sources" in error for error in result["errors"]))

    def test_reactive_candidate_validates_base_evidence(self) -> None:
        result = validation.validate_candidate_packet(reactive_candidate(), expected_mode="reactive")

        self.assertTrue(result["ok"], result["errors"])

    def test_active_candidate_validates_source_grounding(self) -> None:
        result = validation.validate_candidate_packet(active_candidate(), expected_mode="active")

        self.assertTrue(result["ok"], result["errors"])

    def test_intake_gate_requires_review_and_operator_approval(self) -> None:
        pending = validation.validate_candidate_packet(
            reactive_candidate(),
            expected_mode="reactive",
            require_intake_gate=True,
        )
        approved = validation.validate_candidate_packet(
            accepted(reactive_candidate()),
            expected_mode="reactive",
            require_intake_gate=True,
        )

        self.assertFalse(pending["ok"])
        self.assertIn("candidate_review_status must be accepted before intake package creation", pending["errors"])
        self.assertTrue(approved["ok"], approved["errors"])


if __name__ == "__main__":
    unittest.main()
