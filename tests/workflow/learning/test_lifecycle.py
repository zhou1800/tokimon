from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import accepted, active_candidate, reactive_candidate
from tokimon.workflow.learning import artifacts


class LearningLifecycleTests(unittest.TestCase):
    def test_acceptance_scan_and_plan_allow_aliases_inside_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", run_root=tmp)
            target = Path(run["run_path"])
            artifacts.write_json(target / "packet-copy.json", accepted(reactive_candidate()))
            (target / "candidates/reactive.json").symlink_to("../packet-copy.json")
            plan = target / "acceptance-plan.json"
            plan.rename(target / "plan-copy.json")
            plan.symlink_to("plan-copy.json")
            result = artifacts.validate_acceptance(run["run_id"], run_root=tmp)
            self.assertTrue(result["reactive_ready"])
            self.assertEqual(artifacts.read_json(target / "plan-copy.json")["reactive_ready"], True)

    def test_acceptance_scan_rejects_a_looped_json_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", run_root=tmp)
            (Path(run["run_path"]) / "candidates/loop.json").symlink_to("loop.json")
            with self.assertRaisesRegex(ValueError, "cannot be resolved"):
                artifacts.validate_acceptance(run["run_id"], run_root=tmp)

    def test_acceptance_scans_reject_escaping_directories_and_json_entries(self) -> None:
        for directory in ("candidates", "intake-packages", "activation-records"):
            for kind in ("directory", "filename"):
                with self.subTest(directory=directory, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    run = artifacts.start_run(mode="acceptance", run_root=Path(tmp) / "runs")
                    target = Path(run["run_path"])
                    victim = Path(tmp) / "victim"
                    if kind == "directory":
                        victim.mkdir()
                        marker = victim / "outside.json"
                        link = target / directory
                        link.rmdir()
                    else:
                        marker = victim
                        link = target / directory / "outside.json"
                    marker.write_text("not JSON; must not be read")
                    link.symlink_to(victim, target_is_directory=victim.is_dir())
                    original_plan = (target / "acceptance-plan.json").read_bytes()
                    with self.assertRaisesRegex(ValueError, "owning directory"):
                        artifacts.validate_acceptance(run["run_id"], run_root=Path(tmp) / "runs")
                    self.assertEqual(marker.read_text(), "not JSON; must not be read")
                    self.assertEqual((target / "acceptance-plan.json").read_bytes(), original_plan)

    def test_acceptance_plan_filename_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", run_root=Path(tmp) / "runs")
            plan = Path(run["run_path"]) / "acceptance-plan.json"
            plan.unlink()
            victim = Path(tmp) / "victim"
            victim.write_text("unchanged")
            plan.symlink_to(victim)
            with self.assertRaises(ValueError):
                artifacts.validate_acceptance(run["run_id"], run_root=Path(tmp) / "runs")
            self.assertEqual(victim.read_text(), "unchanged")

    def test_review_approval_and_intake_package_update_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packet = accepted(reactive_candidate())
            payload = artifacts.create_intake_package(
                packet,
                proposed_surface="skill",
                intake_id="broker-cdp-bind",
                package_root=tmp,
            )

            self.assertTrue(payload["ok"], payload["errors"] if "errors" in payload else payload)
            self.assertEqual(packet["intake_package_status"], "package_created")
            self.assertTrue(Path(payload["artifact_path"]).is_file())
            self.assertEqual(payload["package"]["packet_id"], packet["packet_id"])

    def test_create_candidate_writes_validation_report_to_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="active", target_capability="dreaming", source="/ref", run_root=tmp)
            payload = artifacts.write_candidate(active_candidate(), mode="active", run_id=run["run_id"], run_root=tmp)

            self.assertTrue(payload["ok"], payload["errors"])
            self.assertTrue(Path(payload["packet_path"]).is_file())
            self.assertTrue(Path(payload["validation_report_path"]).is_file())

    def test_intake_approval_requires_accepted_review(self) -> None:
        packet = reactive_candidate()

        with self.assertRaises(ValueError):
            artifacts.approve_intake(packet, operator_id="operator", approval_note="too early")

        artifacts.record_review(packet, reviewer_id="reviewer-a", decision="reject")
        with self.assertRaises(ValueError):
            artifacts.approve_intake(packet, operator_id="operator", approval_note="rejected")


if __name__ == "__main__":
    unittest.main()
