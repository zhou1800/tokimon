from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.workflow.learning.helpers import (
    accepted,
    active_candidate,
    product_development_evidence,
    reactive_candidate,
    write_json,
)
from tokimon.workflow.learning import artifacts


class LearningAcceptanceTests(unittest.TestCase):
    def test_activation_write_rejects_directory_and_filename_escapes(self) -> None:
        for kind in ("directory", "filename"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "runs"
                run = artifacts.start_run(mode="acceptance", run_root=root)
                package = self._write_accepted_candidate_with_package(run=run, run_root=str(root), packet=accepted(active_candidate()), intake_id="boundary", proposed_surface="skill")
                now = datetime(2026, 10, 2, 17, 10, tzinfo=timezone.utc)
                kwargs = {"activated_path": "skills/dreaming/skill.md", "evidence": product_development_evidence(), "run_id": run["run_id"], "run_root": root, "now": now}
                first = artifacts.record_activation(package, **kwargs)
                self.assertTrue(first["ok"], first)
                link = Path(first["record_path"])
                link.unlink()
                victim = Path(tmp) / "victim"
                if kind == "directory":
                    link = link.parent
                    link.rmdir()
                    victim.mkdir()
                    marker = victim / "marker"
                else:
                    marker = victim
                marker.write_text("unchanged")
                link.symlink_to(victim, target_is_directory=victim.is_dir())
                with self.assertRaises(ValueError):
                    artifacts.record_activation(package, **kwargs)
                self.assertEqual(marker.read_text(), "unchanged")
                if victim.is_dir():
                    self.assertEqual(list(victim.iterdir()), [marker])

    def _write_accepted_candidate_with_package(
        self,
        *,
        run: dict[str, object],
        run_root: str,
        packet: dict[str, object],
        intake_id: str,
        proposed_surface: str,
    ) -> dict[str, object]:
        package = artifacts.create_intake_package(
            packet,
            proposed_surface=proposed_surface,
            intake_id=intake_id,
            run_id=str(run["run_id"]),
            run_root=run_root,
        )
        self.assertTrue(package["ok"], package["errors"] if "errors" in package else package)
        run_path = Path(str(run["run_path"]))
        write_json(run_path / "candidates" / f"{packet['packet_id']}.json", packet)
        return package["package"]  # type: ignore[return-value]

    def test_acceptance_requires_reactive_active_and_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", target_capability="dreaming", run_root=tmp)
            run_path = Path(run["run_path"])
            write_json(run_path / "candidates" / "reactive.json", accepted(reactive_candidate()))
            package = self._write_accepted_candidate_with_package(
                run=run,
                run_root=tmp,
                packet=accepted(active_candidate(packet_id="dreaming")),
                intake_id="dreaming-surface",
                proposed_surface="skill",
            )
            activation = artifacts.record_activation(
                package,
                activated_path="skills/dreaming/skill.md",
                evidence=product_development_evidence(
                    target_surface="skill",
                    activated_path="skills/dreaming/skill.md",
                ),
                run_id=run["run_id"],
                run_root=tmp,
            )

            result = artifacts.validate_acceptance(run["run_id"], run_root=tmp)

            self.assertTrue(activation["ok"], activation["errors"] if "errors" in activation else activation)
            self.assertTrue(result["ok"], result["errors"])
            self.assertTrue(result["reactive_ready"])
            self.assertTrue(result["active_ready"])
            self.assertTrue(result["activation_ready"])
            self.assertEqual(result["status"], "accepted")

    def test_activation_records_accept_workflow_and_existing_surface_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = artifacts.start_run(mode="acceptance", target_capability="dreaming", run_root=tmp)
            workflow_package = self._write_accepted_candidate_with_package(
                run=run,
                run_root=tmp,
                packet=accepted(active_candidate(packet_id="dreaming-workflow")),
                intake_id="dreaming-workflow-surface",
                proposed_surface="workflow",
            )
            existing_package = self._write_accepted_candidate_with_package(
                run=run,
                run_root=tmp,
                packet=accepted(active_candidate(packet_id="dreaming-existing")),
                intake_id="dreaming-existing-surface",
                proposed_surface="existing_surface_update",
            )

            workflow_activation = artifacts.record_activation(
                workflow_package,
                activated_path="tokimon/workflow/dreaming/",
                evidence=product_development_evidence(
                    target_surface="workflow",
                    activated_path="tokimon/workflow/dreaming/",
                ),
                run_id=run["run_id"],
                run_root=tmp,
            )
            existing_activation = artifacts.record_activation(
                existing_package,
                activated_path="skills/learning/skill.md",
                evidence=product_development_evidence(
                    target_surface="existing_surface_update",
                    activated_path="skills/learning/skill.md",
                ),
                run_id=run["run_id"],
                run_root=tmp,
            )

            self.assertTrue(workflow_activation["ok"], workflow_activation["errors"] if "errors" in workflow_activation else workflow_activation)
            self.assertTrue(existing_activation["ok"], existing_activation["errors"] if "errors" in existing_activation else existing_activation)


if __name__ == "__main__":
    unittest.main()
