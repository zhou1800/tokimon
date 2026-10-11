from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.doctor import cli as doctor_cli


def check_payload(*, ok: bool, status: str) -> dict[str, object]:
    return {
        "workflow_id": "doctor",
        "status": status,
        "ok": ok,
        "summary": {
            "checks": {"pass": 1 if ok else 0, "warn": 0, "fail": 0 if ok else 1},
            "readiness": {"ready": 1, "partial": 0, "blocked": 0},
        },
        "memory": {},
        "tools": {"host": {}, "items": []},
        "workflows": [],
        "readiness": [],
        "checks": [],
    }


def run_main_json(argv: list[str]) -> tuple[int, dict[str, object]]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = doctor_cli.main(argv)
    return status, json.loads(output.getvalue())


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class DoctorCliContractTests(unittest.TestCase):
    def test_doctor_cli_contract(self) -> None:
        parser = doctor_cli.build_parser()

        self.assertEqual(subcommand_names(parser), {"check", "contract"})

        default_args = parser.parse_args(["--json"])
        self.assertIsNone(default_args.command)
        self.assertTrue(default_args.json)

        check_args = parser.parse_args(["check", "--workspace-root", "/tmp/tokimon", "--json"])
        self.assertEqual(check_args.command, "check")
        self.assertEqual(check_args.workspace_root, "/tmp/tokimon")
        self.assertTrue(check_args.json)

        contract_args = parser.parse_args(["contract", "--json"])
        self.assertEqual(contract_args.command, "contract")
        self.assertTrue(contract_args.json)

    def test_default_check_json_uses_public_cli_boundary(self) -> None:
        payload = check_payload(ok=True, status="pass")

        with mock.patch.object(doctor_cli.checks, "run_doctor", return_value=payload) as run_doctor:
            status, emitted = run_main_json(["--workspace-root", "/tmp/tokimon", "--json"])

        self.assertEqual(status, 0)
        self.assertEqual(emitted["workflow_id"], "doctor")
        self.assertTrue(emitted["ok"])
        run_doctor.assert_called_once_with(workspace_root=Path("/tmp/tokimon"))

    def test_check_json_foundation_failure_returns_exit_2(self) -> None:
        payload = check_payload(ok=False, status="fail")

        with mock.patch.object(doctor_cli.checks, "run_doctor", return_value=payload):
            status, emitted = run_main_json(["check", "--workspace-root", "/tmp/tokimon", "--json"])

        self.assertEqual(status, 2)
        self.assertFalse(emitted["ok"])
        self.assertEqual(emitted["status"], "fail")

    def test_contract_json_public_command(self) -> None:
        status, emitted = run_main_json(["contract", "--json"])

        self.assertEqual(status, 0)
        self.assertTrue(emitted["ok"])
        contract_payload = emitted["contract"]
        self.assertEqual(contract_payload["workflow_id"], "doctor")
        self.assertEqual(contract_payload["commands"], ["check", "contract"])

    def test_module_entrypoint_contract_json(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(ROOT), env.get("PYTHONPATH", "")) if path
        )

        completed = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.doctor", "contract", "--json"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        emitted = json.loads(completed.stdout)
        self.assertTrue(emitted["ok"])
        self.assertEqual(emitted["contract"]["workflow_id"], "doctor")


if __name__ == "__main__":
    unittest.main()
