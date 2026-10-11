from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.pair import cli as pair_cli
from tokimon.workflow.pair import state as pair_state


def write_settings(
    path: Path,
    *,
    access: str = "local-cli",
    peer_b: str = "claude",
) -> None:
    path.write_text(
        json.dumps(
            {
                "pair": {
                    "access": access,
                    "peer_b": peer_b,
                    "cwd": "/workspace/tokimon",
                }
            }
        ),
        encoding="utf-8",
    )


def subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("parser has no subcommands")


class PairCliContractTests(unittest.TestCase):
    def test_status_and_select_peers_unreadable_state_return_existing_error_json(self) -> None:
        for command in ("status", "select-peers"):
            for shape in ("directory", "invalid-utf8"):
                with self.subTest(command=command, shape=shape), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp) / "runs"
                    run = pair_state.init_run("State reader boundary", state_root=root)
                    path = Path(run["state_path"])
                    if shape == "directory":
                        path.unlink()
                        path.mkdir()
                    else:
                        path.write_bytes(b"\xff")
                    settings = Path(tmp) / "settings.json"
                    write_settings(settings)
                    settings_before = settings.read_bytes()
                    before = {p: p.read_bytes() for p in Path(run["run_path"]).rglob("*") if p.is_file()}
                    args = [sys.executable, "-m", "tokimon.workflow.pair", command, "--run-id", run["run_id"], "--state-root", str(root), "--json"]
                    if command == "select-peers":
                        args.extend(["--settings", str(settings), "--task-behavior", "read_only"])
                    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    payload = json.loads(result.stdout)
                    self.assertEqual(set(payload), {"ok", "run_id", "error"})
                    self.assertFalse(payload["ok"])
                    self.assertTrue(payload["error"])
                    if shape == "directory":
                        self.assertIn("unable to read state.json", payload["error"])
                        self.assertTrue(path.is_dir())
                        self.assertEqual(list(path.iterdir()), [])
                    self.assertEqual(result.stderr, "")
                    self.assertEqual(settings.read_bytes(), settings_before)
                    self.assertEqual({p: p.read_bytes() for p in Path(run["run_path"]).rglob("*") if p.is_file()}, before)

    def test_validate_malformed_json_files_exits_2_with_errors_json(self) -> None:
        for artifact in ("state.json", "artifacts.json"):
            for shape in ("directory", "invalid-utf8"):
                with self.subTest(artifact=artifact, shape=shape), tempfile.TemporaryDirectory() as tmp:
                    run = pair_state.init_run("Malformed JSON CLI", state_root=tmp)
                    path = Path(run["run_path"]) / artifact
                    if shape == "directory":
                        path.unlink()
                        path.mkdir()
                    else:
                        path.write_bytes(b"\xff")
                    before = {p: p.read_bytes() for p in Path(run["run_path"]).rglob("*") if p.is_file()}
                    result = subprocess.run([sys.executable, "-m", "tokimon.workflow.pair", "validate", "--run-id", run["run_id"], "--state-root", tmp, "--json"], cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    payload = json.loads(result.stdout)
                    self.assertEqual(set(payload), {"ok", "run_id", "run_path", "errors", "checked"})
                    self.assertFalse(payload["ok"])
                    self.assertTrue(any(artifact in error for error in payload["errors"]), payload)
                    self.assertEqual(result.stderr, "")
                    self.assertEqual({p: p.read_bytes() for p in Path(run["run_path"]).rglob("*") if p.is_file()}, before)

    def test_select_peers_looped_task_directory_exits_2_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "runs"
            run = pair_state.init_run("Looped task checkout", state_root=run_root)
            state_file = Path(run["state_path"])
            settings = Path(tmp) / "settings.json"
            write_settings(settings)
            state_before = state_file.read_bytes()
            settings_before = settings.read_bytes()
            loop = Path(tmp) / "loop"
            loop.symlink_to("loop", target_is_directory=True)
            result = subprocess.run([sys.executable, "-m", "tokimon.workflow.pair", "select-peers", "--run-id", run["run_id"], "--state-root", str(run_root), "--settings", str(settings), "--cwd", str(loop), "--task-behavior", "read_only", "--json"], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(set(payload), {"ok", "run_id", "error"})
            self.assertFalse(payload["ok"])
            self.assertIn("cannot be resolved", payload["error"])
            self.assertEqual(result.stderr, "")
            self.assertEqual(state_file.read_bytes(), state_before)
            self.assertEqual(settings.read_bytes(), settings_before)

    def test_validate_rejected_run_paths_returns_structured_exit_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            root.mkdir()
            victim = Path(tmp) / "victim"
            victim.mkdir()
            marker = victim / "state.json"
            marker.write_text("unchanged")
            (root / "linked").symlink_to(victim, target_is_directory=True)
            (root / "loop").symlink_to("loop", target_is_directory=True)
            for run_id in ("../victim", str(victim), "nested/run", "linked", "loop", ".", ".."):
                with self.subTest(run_id=run_id):
                    result = subprocess.run([sys.executable, "-m", "tokimon.workflow.pair", "validate", "--run-id", run_id, "--state-root", str(root), "--json"], cwd=ROOT, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    payload = json.loads(result.stdout)
                    self.assertEqual(set(payload), {"ok", "run_id", "run_path", "errors", "checked"})
                    self.assertFalse(payload["ok"])
                    self.assertEqual(payload["run_id"], run_id)
                    self.assertEqual(payload["checked"], [])
                    self.assertTrue(payload["errors"])
                    self.assertEqual(result.stderr, "")
                    self.assertEqual(marker.read_text(), "unchanged")

    def test_select_peers_rejects_escaping_or_looped_run_before_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            root.mkdir()
            victim = Path(tmp) / "victim"
            victim.mkdir()
            state_file = victim / "state.json"
            state_file.write_text('{"run_id": "victim", "unchanged": true}')
            (root / "linked").symlink_to(victim, target_is_directory=True)
            (root / "loop").symlink_to("loop", target_is_directory=True)
            settings = Path(tmp) / "settings.json"
            write_settings(settings)
            for run_id in ("../victim", str(victim), "linked", "loop"):
                with self.subTest(run_id=run_id):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = pair_cli.main(["select-peers", "--run-id", run_id, "--state-root", str(root), "--settings", str(settings), "--task-behavior", "read_only", "--json"])
                    payload = json.loads(output.getvalue())
                    self.assertNotEqual(code, 0, payload)
                    self.assertFalse(payload["ok"])
                    self.assertEqual(state_file.read_text(), '{"run_id": "victim", "unchanged": true}')


    def test_run_uses_subject_checkout_and_requested_round_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_checkout = Path(tmp) / "subject-repository"
            task_checkout.mkdir()
            settings_path = Path(tmp) / "settings.json"
            write_settings(settings_path, access="native-subagent", peer_b="codex")
            original_settings = settings_path.read_bytes()
            state_root = Path(tmp) / "runs"
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            command = [sys.executable, "-m", "tokimon.workflow.pair"]
            initialized = subprocess.run(
                command + [
                    "init", "--task-summary", "Review subject", "--max-rounds", "3",
                    "--state-root", str(state_root), "--json",
                ],
                cwd=task_checkout,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            run = json.loads(initialized.stdout)
            selected = subprocess.run(
                command + [
                    "select-peers", "--run-id", run["run_id"], "--state-root", str(state_root),
                    "--settings", str(settings_path), "--cwd", str(task_checkout),
                    "--task-behavior", "read_only", "--json",
                ],
                cwd=task_checkout,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            state = json.loads(selected.stdout)["state"]
            self.assertEqual(state["max_rounds"], 3)
            self.assertEqual(
                [state["peers"][slot]["cwd"] for slot in ("a", "b")], [str(task_checkout)] * 2,
            )
            self.assertEqual(settings_path.read_bytes(), original_settings)

    def test_invalid_round_limit_reports_input_error_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable, "-m", "tokimon.workflow.pair", "init",
                    "--task-summary", "Invalid limit", "--max-rounds", "0", "--state-root", tmp, "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse(json.loads(result.stdout)["ok"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_main_defaults_to_contract_text(self) -> None:
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = pair_cli.main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("workflow_id: pair", stdout.getvalue())
        self.assertIn("entrypoint: .venv/bin/python -m tokimon.workflow.pair", stdout.getvalue())

    def test_pair_cli_contract(self) -> None:
        parser = pair_cli.build_parser()

        self.assertEqual(subcommand_names(parser), {"contract", "init", "select-peers", "status", "validate"})

        default_args = parser.parse_args([])
        self.assertIsNone(default_args.command)

        contract_args = parser.parse_args(["contract", "--json"])
        self.assertEqual(contract_args.command, "contract")
        self.assertTrue(contract_args.json)

        init_args = parser.parse_args(
            [
                "init",
                "--task-summary",
                "Pair plan",
                "--slug",
                "pair-plan",
                "--state-root",
                "/tmp/pair-runs",
                "--json",
            ]
        )
        self.assertEqual(init_args.command, "init")
        self.assertEqual(init_args.task_summary, "Pair plan")
        self.assertEqual(init_args.slug, "pair-plan")
        self.assertEqual(init_args.state_root, "/tmp/pair-runs")
        self.assertTrue(init_args.json)

        select_peers_args = parser.parse_args(
            [
                "select-peers",
                "--run-id",
                "pair-1",
                "--settings",
                "/tmp/settings.local.json",
                "--task-behavior",
                "read_only",
                "--json",
            ]
        )
        self.assertEqual(select_peers_args.command, "select-peers")
        self.assertEqual(select_peers_args.settings, "/tmp/settings.local.json")
        self.assertEqual(select_peers_args.task_behavior, "read_only")
        self.assertTrue(select_peers_args.json)

        status_args = parser.parse_args(["status", "--run-id", "pair-1", "--state-root", "/tmp/pair-runs", "--json"])
        self.assertEqual(status_args.command, "status")
        self.assertEqual(status_args.run_id, "pair-1")
        self.assertTrue(status_args.json)

        validate_args = parser.parse_args(["validate", "--run-id", "pair-1", "--json"])
        self.assertEqual(validate_args.command, "validate")
        self.assertEqual(validate_args.run_id, "pair-1")
        self.assertTrue(validate_args.json)

    def test_module_defaults_to_contract_text(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.pair"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("workflow_id: pair", result.stdout)

    def test_module_contract_json(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tokimon.workflow.pair", "contract", "--json"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["contract"]["workflow_id"], "pair")
        self.assertEqual(payload["contract"]["consensus_artifact"], "consensus.md")
        self.assertIn("peer_a_evidence", payload["contract"]["no_consensus_fields"])
        self.assertIn("final_text_sha256", payload["contract"]["consensus_status_fields"])
        self.assertIn("reviewer_evidence_artifact", payload["contract"]["mutable_work_fields"])

    def test_module_init_status_validate_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            init_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "init",
                    "--task-summary",
                    "Pair smoke test",
                    "--slug",
                    "smoke-test",
                    "--state-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            init_payload = json.loads(init_result.stdout)
            self.assertTrue(init_payload["ok"])
            run_id = init_payload["run_id"]
            self.assertIn("smoke-test", run_id)

            settings_path = Path(tmp) / "settings.local.json"
            write_settings(settings_path)

            select_peers_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "select-peers",
                    "--run-id",
                    run_id,
                    "--state-root",
                    tmp,
                    "--settings",
                    str(settings_path),
                    "--task-behavior",
                    "read_only",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            select_peers_payload = json.loads(select_peers_result.stdout)
            self.assertTrue(select_peers_payload["ok"])
            self.assertEqual(
                select_peers_payload["peer_selection_config"]["selected_peer_ids"],
                ["local-a-codex", "local-b-claude"],
            )

            status_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "status",
                    "--run-id",
                    run_id,
                    "--state-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            status_payload = json.loads(status_result.stdout)
            self.assertTrue(status_payload["ok"])
            self.assertEqual(status_payload["state"]["status"], "in_progress")
            self.assertEqual(status_payload["state"]["peers"]["a"]["agent"], "codex")

            validate_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "validate",
                    "--run-id",
                    run_id,
                    "--state-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            validate_payload = json.loads(validate_result.stdout)
            self.assertTrue(validate_payload["ok"])

    def test_public_cli_failure_json_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.local.json"
            write_settings(settings_path)
            missing_run_id = "pair-20260505T010203Z-missing-run"
            select_peers_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "select-peers",
                    "--run-id",
                    missing_run_id,
                    "--state-root",
                    tmp,
                    "--settings",
                    str(settings_path),
                    "--task-behavior",
                    "read_only",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(select_peers_result.returncode, 2)
            select_payload = json.loads(select_peers_result.stdout)
            self.assertFalse(select_payload["ok"])
            self.assertEqual(select_payload["run_id"], missing_run_id)
            self.assertIn("state.json", select_payload["error"])

            status_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "status",
                    "--run-id",
                    missing_run_id,
                    "--state-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(status_result.returncode, 2)
            status_payload = json.loads(status_result.stdout)
            self.assertFalse(status_payload["ok"])
            self.assertEqual(status_payload["run_id"], missing_run_id)

            validate_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "validate",
                    "--run-id",
                    missing_run_id,
                    "--state-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(validate_result.returncode, 2)
            validate_payload = json.loads(validate_result.stdout)
            self.assertFalse(validate_payload["ok"])
            self.assertIn("missing run directory", validate_payload["errors"][0])

    def test_select_peers_records_local_cli_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            init_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "init",
                    "--task-summary",
                    "Pair local CLI smoke test",
                    "--slug",
                    "local-cli-smoke",
                    "--state-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            run_id = json.loads(init_result.stdout)["run_id"]

            settings_path = Path(tmp) / "settings.local.json"
            write_settings(settings_path, access="local-cli")

            select_peers_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "select-peers",
                    "--run-id",
                    run_id,
                    "--state-root",
                    tmp,
                    "--settings",
                    str(settings_path),
                    "--task-behavior",
                    "read_only",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            select_peers_payload = json.loads(select_peers_result.stdout)

            self.assertTrue(select_peers_payload["ok"])
            self.assertEqual(
                select_peers_payload["peer_selection_config"]["selected_peer_ids"],
                ["local-a-codex", "local-b-claude"],
            )
            self.assertEqual(select_peers_payload["peers"]["a"]["capability"], "local-cli")
            self.assertEqual(select_peers_payload["peers"]["b"]["capability"], "local-cli")

    def test_select_peers_records_native_subagent_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            init_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "init",
                    "--task-summary",
                    "Pair native subagent smoke test",
                    "--slug",
                    "native-subagent-smoke",
                    "--state-root",
                    tmp,
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            run_id = json.loads(init_result.stdout)["run_id"]

            settings_path = Path(tmp) / "settings.local.json"
            write_settings(settings_path, access="native-subagent", peer_b="codex")

            select_peers_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tokimon.workflow.pair",
                    "select-peers",
                    "--run-id",
                    run_id,
                    "--state-root",
                    tmp,
                    "--settings",
                    str(settings_path),
                    "--task-behavior",
                    "read_only",
                    "--json",
                ],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
            payload = json.loads(select_peers_result.stdout)

        self.assertEqual(
            payload["peer_selection_config"]["selected_peer_ids"],
            ["native-a-codex", "native-b-codex"],
        )
        self.assertEqual(payload["peers"]["a"]["capability"], "native-subagent")
        self.assertEqual(payload["peers"]["b"]["capability"], "native-subagent")


if __name__ == "__main__":
    unittest.main()
