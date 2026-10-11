from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import (
    accepted,
    active_candidate,
    product_development_evidence,
    write_json,
)
from tokimon.workflow.learning import artifacts


class LearningAcceptanceStatusTests(unittest.TestCase):
    def test_acceptance_reports_missing_reactive_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", target_capability="dreaming", run_root=tmp)

            result = artifacts.validate_acceptance(run["run_id"], run_root=tmp)

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "incomplete")
            self.assertIn("reactive learning evidence is missing", result["errors"])

    def test_activation_record_rejects_missing_product_development_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", target_capability="dreaming", run_root=tmp)
            packet = accepted(active_candidate(packet_id="dreaming"))
            package_payload = artifacts.create_intake_package(
                packet,
                proposed_surface="skill",
                intake_id="dreaming-surface",
                run_id=run["run_id"],
                run_root=tmp,
            )
            self.assertTrue(
                package_payload["ok"],
                package_payload["errors"] if "errors" in package_payload else package_payload,
            )
            write_json(Path(run["run_path"]) / "candidates" / "active.json", packet)
            result = artifacts.record_activation(
                package_payload["package"],
                activated_path="skills/dreaming/skill.md",
                evidence={},
                run_id=run["run_id"],
                run_root=tmp,
            )

            self.assertFalse(result["ok"])
            self.assertIn("product_development_evidence must be a non-empty object", result["errors"])

    def test_activation_record_rejects_summary_only_product_development_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", target_capability="dreaming", run_root=tmp)
            packet = accepted(active_candidate(packet_id="dreaming"))
            package_payload = artifacts.create_intake_package(
                packet,
                proposed_surface="skill",
                intake_id="dreaming-surface",
                run_id=run["run_id"],
                run_root=tmp,
            )
            self.assertTrue(
                package_payload["ok"],
                package_payload["errors"] if "errors" in package_payload else package_payload,
            )
            write_json(Path(run["run_path"]) / "candidates" / "active.json", packet)
            result = artifacts.record_activation(
                package_payload["package"],
                activated_path="skills/dreaming/skill.md",
                evidence={"summary": "Accepted.", "target_surface": "skill"},
                run_id=run["run_id"],
                run_root=tmp,
            )

            self.assertFalse(result["ok"])
            self.assertIn("product_development_evidence.acceptance_status must be accepted", result["errors"])
            self.assertIn(
                "product_development_evidence.accepted_artifacts must be a non-empty list",
                result["errors"],
            )

    def test_activation_record_requires_run_local_intake_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", target_capability="dreaming", run_root=tmp)
            packet = accepted(active_candidate(packet_id="dreaming"))
            write_json(Path(run["run_path"]) / "candidates" / "active.json", packet)
            package = {
                "intake_id": "dreaming-surface",
                "packet_id": "dreaming",
                "source_candidate_id": "dreaming",
                "target_capability": "dreaming",
                "activation_candidate": {"proposed_surface": "skill"},
            }

            result = artifacts.record_activation(
                package,
                activated_path="skills/dreaming/skill.md",
                evidence=product_development_evidence(
                    target_surface="skill",
                    activated_path="skills/dreaming/skill.md",
                ),
                run_id=run["run_id"],
                run_root=tmp,
            )

            self.assertFalse(result["ok"])
            self.assertIn("intake package must exist in the run before activation recording", result["errors"])


if __name__ == "__main__":
    unittest.main()
