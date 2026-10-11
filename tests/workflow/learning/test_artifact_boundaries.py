from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import accepted, reactive_candidate
from tokimon.workflow.learning import artifacts, contract


class LearningArtifactBoundaryTests(unittest.TestCase):
    def test_candidate_generated_paths_reject_directory_and_filename_escapes(self) -> None:
        for relative in ("candidates", "validation-reports", "state.json", "candidates/boundary.json", "validation-reports/boundary-validation.json"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "runs"
                run = artifacts.start_run(mode="reactive", trigger="tool_failure", run_root=root)
                target = Path(run["run_path"])
                link = target / relative
                victim = Path(tmp) / "victim"
                if link.is_dir():
                    link.rmdir()
                    victim.mkdir()
                    marker = victim / "marker"
                    marker.write_text("unchanged")
                else:
                    link.unlink(missing_ok=True)
                    victim.write_text("unchanged")
                    marker = victim
                link.symlink_to(victim, target_is_directory=victim.is_dir())
                with self.assertRaises(ValueError):
                    artifacts.write_candidate(reactive_candidate(packet_id="boundary"), mode="reactive", run_id=run["run_id"], run_root=root)
                self.assertEqual(marker.read_text(), "unchanged")
                if victim.is_dir():
                    self.assertEqual(list(victim.iterdir()), [marker])
                if relative.startswith("validation-reports"):
                    self.assertFalse((target / "candidates/boundary.json").exists())

    def test_candidate_write_allows_contained_state_and_output_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="reactive", trigger="tool_failure", run_root=tmp)
            target = Path(run["run_path"])
            (target / "state.json").rename(target / "state-copy.json")
            (target / "state.json").symlink_to("state-copy.json")
            (target / "candidates/boundary.json").symlink_to("../packet-copy.json")
            result = artifacts.write_candidate(reactive_candidate(packet_id="boundary"), mode="reactive", run_id=run["run_id"], run_root=tmp)
            self.assertTrue(result["ok"], result)
            self.assertEqual(artifacts.read_json(target / "packet-copy.json")["packet_id"], "boundary")

    def test_candidate_write_rejects_run_root_escape_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            root.mkdir()
            victim = Path(tmp) / "victim"
            victim.mkdir()
            marker = victim / "marker"
            marker.write_text("unchanged")
            (root / "linked").symlink_to(victim, target_is_directory=True)
            (root / "loop").symlink_to("loop", target_is_directory=True)
            for run_id in ("../victim", str(victim), "nested/run", "linked", "loop", ".", ".."):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    artifacts.write_candidate(reactive_candidate(), mode="reactive", run_id=run_id, run_root=root)
                self.assertEqual(marker.read_text(), "unchanged")
                self.assertEqual(list(victim.iterdir()), [marker])
            initialized = artifacts.start_run(mode="reactive", trigger="tool_failure", run_root=root)
            Path(initialized["run_path"]).rename(root / "legacy run.v1")
            safe = artifacts.write_candidate(reactive_candidate(), mode="reactive", run_id="legacy run.v1", run_root=root)
            self.assertTrue(safe["ok"], safe)
            self.assertTrue(Path(safe["packet_path"]).is_relative_to(root / "legacy run.v1"))

    def test_candidate_packet_is_local_until_intake_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            package_root = Path(tmp) / "intake-packages"
            run = artifacts.start_run(mode="reactive", trigger="tool_failure", run_root=run_root)
            created = artifacts.write_candidate(reactive_candidate(), mode="reactive", run_id=run["run_id"], run_root=run_root)

            self.assertTrue(created["ok"], created["errors"])
            self.assertTrue(Path(created["packet_path"]).is_file())
            self.assertFalse(package_root.exists())

            packaged = artifacts.create_intake_package(
                accepted(reactive_candidate()),
                proposed_surface="skill",
                package_root=package_root,
            )
            self.assertTrue(packaged["ok"], packaged["errors"] if "errors" in packaged else packaged)
            self.assertTrue(package_root.is_dir())

    def test_start_run_persists_initial_artifacts_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="reactive", trigger="tool_failure", run_root=tmp)
            target = Path(tmp) / run["run_id"]
            state = json.loads((target / "state.json").read_text(encoding="utf-8"))
            plan = json.loads((target / "acceptance-plan.json").read_text(encoding="utf-8"))
            manifest = json.loads((target / "artifacts.json").read_text(encoding="utf-8"))

            self.assertRegex(run["run_id"], contract.RUN_ID_PATTERN)
            self.assertTrue(set(contract.REQUIRED_STATE_FIELDS).issubset(state))
            for directory in contract.REQUIRED_DIRECTORIES:
                with self.subTest(directory=directory):
                    self.assertTrue((target / directory).is_dir())
            self.assertEqual(state["run_id"], run["run_id"])
            self.assertEqual(state["workflow_id"], "learning")
            self.assertEqual(state["mode"], "reactive")
            self.assertEqual(state["trigger"], "tool_failure")
            self.assertEqual(state["status"], "initialized")
            self.assertEqual(plan["run_id"], run["run_id"])
            self.assertEqual(manifest["run_id"], run["run_id"])
            self.assertEqual(manifest["workflow_id"], "learning")


if __name__ == "__main__":
    unittest.main()
