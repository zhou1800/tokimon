from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
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
from tokimon.workflow.learning import cli as learning_cli


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class LearningCliContractTests(unittest.TestCase):
    def test_main_defaults_to_contract_text(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = learning_cli.main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("workflow_id: learning", stdout.getvalue())

    def test_parser_exposes_documented_commands(self) -> None:
        parser = learning_cli.build_parser()

        self.assertEqual(
            subcommand_names(parser),
            {
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
            },
        )

        args = parser.parse_args(["start-active", "--target-capability", "dreaming", "--source", "/ref/repo", "--json"])
        self.assertEqual(args.command, "start-active")
        self.assertEqual(args.target_capability, "dreaming")
        self.assertTrue(args.json)

        package_args = parser.parse_args(
            ["create-intake-package", "--packet", "/tmp/candidate.json", "--proposed-surface", "workflow", "--json"]
        )
        self.assertEqual(package_args.command, "create-intake-package")
        self.assertEqual(package_args.proposed_surface, "workflow")

        activation_args = parser.parse_args(
            [
                "record-activation",
                "--intake-package",
                "/tmp/package.json",
                "--activated-path",
                "skills/dreaming/skill.md",
                "--evidence",
                "/tmp/evidence.json",
                "--run-id",
                "learning-20260607T000000Z-dreaming",
                "--json",
            ]
        )
        self.assertEqual(activation_args.command, "record-activation")
        self.assertEqual(activation_args.run_id, "learning-20260607T000000Z-dreaming")

    def test_module_contract_json(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.learning", "contract", "--json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["contract"]["workflow_id"], "learning")
        self.assertIn("skill_and_mcp", payload["contract"]["valid_target_surfaces"])
        self.assertEqual(
            payload["contract"]["required_product_development_evidence_fields"],
            ["acceptance_status", "acceptance_summary", "accepted_artifacts", "verification_evidence"],
        )
        self.assertEqual(payload["contract"]["required_accepted_artifact_fields"], ["path", "role"])
        self.assertEqual(
            payload["contract"]["required_verification_evidence_fields"],
            ["kind", "reference", "outcome"],
        )

    def test_module_start_active_uses_supplied_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.learning",
                    "start-active",
                    "--target-capability",
                    "dreaming",
                    "--source",
                    "/ref/repo",
                    "--slug",
                    "dreaming",
                    "--run-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertIn("dreaming", payload["run_id"])
            self.assertTrue((Path(payload["run_path"]) / "state.json").is_file())

    def test_module_record_activation_uses_supplied_run_id(self) -> None:
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
            evidence_path = Path(tmp) / "activation-evidence.json"
            write_json(
                evidence_path,
                product_development_evidence(
                    target_surface="skill",
                    activated_path="skills/dreaming/skill.md",
                ),
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.learning",
                    "record-activation",
                    "--intake-package",
                    str(package_payload["artifact_path"]),
                    "--activated-path",
                    "skills/dreaming/skill.md",
                    "--evidence",
                    str(evidence_path),
                    "--run-id",
                    str(run["run_id"]),
                    "--run-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )

            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["run_id"], run["run_id"])
            self.assertTrue(Path(payload["record_path"]).is_file())
            self.assertEqual(payload["record"]["intake_id"], "dreaming-surface")


if __name__ == "__main__":
    unittest.main()
