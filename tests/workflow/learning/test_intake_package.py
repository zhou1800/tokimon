from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import accepted, active_candidate
from tokimon.workflow.learning import artifacts


class LearningIntakePackageTests(unittest.TestCase):
    def test_run_intake_package_rejects_directory_and_filename_escapes(self) -> None:
        for relative in ("intake-packages", "intake-packages/boundary.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                run = artifacts.start_run(mode="active", target_capability="dreaming", run_root=Path(tmp) / "runs")
                target = Path(run["run_path"])
                link = target / relative
                victim = Path(tmp) / "victim"
                if link.is_dir():
                    link.rmdir()
                    victim.mkdir()
                    marker = victim / "marker"
                else:
                    marker = victim
                marker.write_text("unchanged")
                link.symlink_to(victim, target_is_directory=victim.is_dir())
                packet = accepted(active_candidate())
                with self.assertRaises(ValueError):
                    artifacts.create_intake_package(packet, proposed_surface="skill", intake_id="boundary", run_id=run["run_id"], run_root=Path(tmp) / "runs")
                self.assertEqual(marker.read_text(), "unchanged")
                self.assertEqual(packet["intake_package_status"], "not_created")

    def test_explicit_package_root_allows_its_own_contained_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "standalone"
            root.mkdir()
            (root / "boundary.json").symlink_to("package-copy.json")
            result = artifacts.create_intake_package(accepted(active_candidate()), proposed_surface="skill", intake_id="boundary", package_root=root)
            self.assertTrue(result["ok"], result)
            self.assertEqual(artifacts.read_json(root / "package-copy.json")["intake_id"], "boundary")
            victim = Path(tmp) / "victim"
            victim.write_text("unchanged")
            (root / "escape.json").symlink_to(victim)
            with self.assertRaises(ValueError):
                artifacts.create_intake_package(accepted(active_candidate()), proposed_surface="skill", intake_id="escape", package_root=root)
            self.assertEqual(victim.read_text(), "unchanged")

    def test_create_intake_package_requires_product_development(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = accepted(active_candidate(packet_id="dreaming"))
            payload = artifacts.create_intake_package(
                packet,
                proposed_surface="skill_and_mcp",
                intake_id="dreaming-surface",
                package_root=tmp,
            )

            self.assertTrue(payload["ok"], payload["errors"] if "errors" in payload else payload)
            package = artifacts.read_json(payload["artifact_path"])
            self.assertEqual(package["schema_id"], "learning.product_development_intake.v1")
            self.assertTrue(package["normal_product_development_required"])
            self.assertTrue(package["activation_not_permitted_by_learning_workflow"])
            self.assertEqual(package["intake_id"], "dreaming-surface")
            self.assertEqual(package["packet_id"], "dreaming")
            self.assertEqual(package["source_candidate_id"], "dreaming")
            self.assertEqual(package["activation_candidate"]["proposed_surface"], "skill_and_mcp")
            self.assertTrue(package["activation_candidate"]["surface_is_non_binding"])
            self.assertEqual(package["handoff"]["status"], "ready_for_product_development")
            self.assertIn("product-development", package["handoff"]["next_decision"])
            self.assertTrue(package["product_intent"])
            self.assertTrue(package["acceptance_expectations"])
            self.assertTrue(package["evidence_refs"]["source_evidence"])
            self.assertTrue(package["constraints"]["negative_scope"])
            self.assertEqual(packet["intake_package_status"], "package_created")

    def test_create_intake_package_rejects_unknown_surface_hypothesis(self) -> None:
        packet = accepted(active_candidate(packet_id="dreaming"))

        with self.assertRaises(ValueError):
            artifacts.create_intake_package(packet, proposed_surface="not-a-surface")


if __name__ == "__main__":
    unittest.main()
