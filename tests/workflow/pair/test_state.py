from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tokimon.workflow.pair import contract, peer_config, state


def peer_selection() -> tuple[dict[str, object], dict[str, object]]:
    peers = {
        "a": {
            "id": "local-a-codex",
            "capability": "local-cli",
            "agent": "codex",
            "cwd": "/workspace/tokimon",
        },
        "b": {
            "id": "local-b-claude",
            "capability": "local-cli",
            "agent": "claude",
            "cwd": "/workspace/tokimon",
        },
    }
    selection = {
        "source_path": ".tokimon/settings.local.json",
        "settings_section": "pair",
        "selected_peer_ids": ["local-a-codex", "local-b-claude"],
        "selected_at": "2026-05-05T01:02:03Z",
    }
    return peers, selection


def write_artifact(run_path: Path, artifact: str, text: str) -> None:
    target = run_path / artifact
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def complete_consensus_run(
    tmp: str,
    *,
    task_behavior: str = "read_only",
    access: str = "local-cli",
    peer_b: str = "claude",
) -> tuple[dict[str, object], Path, dict[str, object]]:
    payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
    run_path = Path(payload["run_path"])

    write_artifact(run_path, "rounds/round-1/peer-a.md", "Peer A independent note.\n")
    write_artifact(run_path, "rounds/round-1/peer-b.md", "Peer B independent note.\n")
    write_artifact(run_path, "rounds/round-2/peer-a.md", "Peer A response after receiving Peer B.\n")
    write_artifact(run_path, "rounds/round-2/peer-b.md", "Peer B response after receiving Peer A.\n")

    final_text = "Consensus final text.\n"
    write_artifact(run_path, contract.CONSENSUS_ARTIFACT, final_text)
    final_hash = sha256_text(final_text)

    consensus_status: dict[str, object] = {
        "result": "consensus",
        "rounds_completed": 2,
        "final_text_artifact": contract.CONSENSUS_ARTIFACT,
        "final_text_sha256": final_hash,
        "independent_notes": {
            "round": 1,
            "peer_a_artifact": "rounds/round-1/peer-a.md",
            "peer_b_artifact": "rounds/round-1/peer-b.md",
        },
        "exchange": {
            "round": 2,
            "peer_a_received_artifact": "rounds/round-1/peer-b.md",
            "peer_b_received_artifact": "rounds/round-1/peer-a.md",
            "peer_a_response_artifact": "rounds/round-2/peer-a.md",
            "peer_b_response_artifact": "rounds/round-2/peer-b.md",
        },
        "peer_acceptances": {
            "a": {
                "accepted": True,
                "final_text_artifact": contract.CONSENSUS_ARTIFACT,
                "final_text_sha256": final_hash,
            },
            "b": {
                "accepted": True,
                "final_text_artifact": contract.CONSENSUS_ARTIFACT,
                "final_text_sha256": final_hash,
            },
        },
    }

    current_round_index = 2
    if task_behavior == "shared_mutable_output":
        write_artifact(run_path, "rounds/round-3/peer-b.md", "Peer B reviewed and approved the write.\n")
        consensus_status["mutable_work"] = {
            "writer_peer": "a",
            "reviewer_peer": "b",
            "write_round": 2,
            "review_round": 3,
            "writer_evidence_artifact": "rounds/round-2/peer-a.md",
            "reviewer_evidence_artifact": "rounds/round-3/peer-b.md",
            "reviewer_approved": True,
        }
        current_round_index = 3

    consensus_status["rounds_completed"] = current_round_index
    peers, selection = peer_config.select_peers({"access": access, "peer_b": peer_b, "cwd": tmp})
    state.update_state(
        payload["run_id"],
        state_root=tmp,
        status="consensus",
        current_round_index=current_round_index,
        peers=peers,
        peer_selection_config=selection,
        peer_selection_rationale="Selected from config.",
        task_behavior=task_behavior,
        consensus_status=consensus_status,
    )
    return payload, run_path, consensus_status


def rewrite_consensus_status(state_path: Path, consensus_status: dict[str, object] | None) -> None:
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["consensus_status"] = consensus_status
    state.write_json(state_path, state_payload)


class PairStateTests(unittest.TestCase):
    def test_load_state_directory_is_an_input_error_and_missing_file_stays_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("State read boundary", state_root=tmp)
            path = Path(payload["state_path"])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                state.load_state(payload["run_id"], tmp)
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "unable to read state.json"):
                state.load_state(payload["run_id"], tmp)
            self.assertTrue(path.is_dir())
            self.assertEqual(list(path.iterdir()), [])

    def test_validate_collects_required_json_directory_and_encoding_errors(self) -> None:
        for artifact in ("state.json", "artifacts.json"):
            for shape in ("directory", "invalid-utf8"):
                with self.subTest(artifact=artifact, shape=shape), tempfile.TemporaryDirectory() as tmp:
                    payload = state.init_run("Malformed JSON artifact", state_root=tmp)
                    path = Path(payload["run_path"]) / artifact
                    if shape == "directory":
                        path.unlink()
                        path.mkdir()
                    else:
                        path.write_bytes(b"\xff")
                    before = {p: p.read_bytes() for p in Path(payload["run_path"]).rglob("*") if p.is_file()}
                    result = state.validate_run(payload["run_id"], tmp)
                    self.assertFalse(result["ok"])
                    expected = f"missing file: {artifact}" if shape == "directory" else f"unable to read JSON in {artifact}:"
                    self.assertTrue(any(error.startswith(expected) for error in result["errors"]), result["errors"])
                    self.assertIn(artifact, result["checked"])
                    self.assertEqual({p: p.read_bytes() for p in Path(payload["run_path"]).rglob("*") if p.is_file()}, before)

    def test_validate_collects_json_read_oserror_without_mutating_run(self) -> None:
        for artifact in ("state.json", "artifacts.json"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as tmp:
                payload = state.init_run("Read failure", state_root=tmp)
                target = Path(payload["run_path"]) / artifact
                before = target.read_bytes()
                original_read_text = Path.read_text
                def read_text(path: Path, *args, **kwargs):
                    if path == target:
                        raise PermissionError("synthetic filesystem read denied")
                    return original_read_text(path, *args, **kwargs)
                with mock.patch.object(Path, "read_text", read_text):
                    result = state.validate_run(payload["run_id"], tmp)
                self.assertFalse(result["ok"])
                self.assertIn(f"unable to read JSON in {artifact}: synthetic filesystem read denied", result["errors"])
                self.assertEqual(target.read_bytes(), before)

    def test_state_reads_and_writes_reject_filename_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            payload = state.init_run("State ownership", state_root=root)
            original = state.load_state(payload["run_id"], root)
            link = Path(payload["state_path"])
            victim = Path(tmp) / "victim.json"
            victim.write_bytes(link.read_bytes())
            before = victim.read_bytes()
            link.unlink()
            link.symlink_to(victim)
            operations = (
                lambda: state.load_state(payload["run_id"], root),
                lambda: state.save_state(payload["run_id"], original, root),
                lambda: state.update_state(payload["run_id"], state_root=root, status="blocked"),
            )
            for operation in operations:
                with self.subTest(operation=operation), self.assertRaises(ValueError):
                    operation()
                self.assertEqual(victim.read_bytes(), before)
            result = state.validate_run(payload["run_id"], root)
            self.assertFalse(result["ok"])
            self.assertTrue(any("state.json: Pair artifact path must stay inside" in error for error in result["errors"]))

    def test_state_alias_within_run_remains_readable_writable_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Contained state alias", state_root=tmp)
            link = Path(payload["state_path"])
            link.rename(link.with_name("state-copy.json"))
            link.symlink_to("state-copy.json")
            state.update_state(payload["run_id"], state_root=tmp, status="blocked")
            self.assertEqual(state.load_state(payload["run_id"], tmp)["status"], "blocked")
            result = state.validate_run(payload["run_id"], tmp)
            self.assertTrue(result["ok"], result["errors"])

    def test_validate_rejects_escaping_required_artifacts_and_directories(self) -> None:
        for relative in (*contract.REQUIRED_INITIAL_ARTIFACTS, *contract.REQUIRED_DIRECTORIES):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "runs"
                payload = state.init_run("Artifact ownership", state_root=root)
                link = Path(payload["run_path"]) / relative
                victim = Path(tmp) / "victim"
                if link.is_dir():
                    link.rmdir()
                    victim.mkdir()
                    marker = victim / "marker"
                else:
                    link.unlink()
                    marker = victim
                marker.write_text("not JSON; must not be read")
                link.symlink_to(victim, target_is_directory=victim.is_dir())
                result = state.validate_run(payload["run_id"], root)
                self.assertFalse(result["ok"])
                self.assertTrue(any(relative + ": Pair artifact path must stay inside" in error for error in result["errors"]), result["errors"])
                self.assertEqual(marker.read_text(), "not JSON; must not be read")

    def test_validate_rejects_terminal_escape_and_note_symlink_loop(self) -> None:
        for relative in (contract.CONSENSUS_ARTIFACT, "rounds/round-1/peer-a.md"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as tmp:
                payload, target, _ = complete_consensus_run(str(Path(tmp) / "runs"))
                link = target / relative
                victim = Path(tmp) / "victim"
                victim.write_bytes(link.read_bytes())
                before = victim.read_bytes()
                link.unlink()
                if relative == contract.CONSENSUS_ARTIFACT:
                    link.symlink_to(victim)
                else:
                    link.symlink_to(link.name)
                result = state.validate_run(payload["run_id"], Path(tmp) / "runs")
                self.assertFalse(result["ok"])
                self.assertTrue(any("Pair artifact path" in error for error in result["errors"]), result["errors"])
                self.assertEqual(victim.read_bytes(), before)

    def test_validate_rejects_no_consensus_terminal_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            payload, target, _ = complete_consensus_run(str(root))
            state.update_state(payload["run_id"], state_root=root, status="no_consensus", no_consensus={field: "Recorded evidence" for field in contract.NO_CONSENSUS_FIELDS})
            victim = Path(tmp) / "victim.md"
            victim.write_text("No consensus evidence")
            (target / contract.NO_CONSENSUS_ARTIFACT).symlink_to(victim)
            result = state.validate_run(payload["run_id"], root)
            self.assertFalse(result["ok"])
            self.assertTrue(any("Pair artifact path must stay inside" in error for error in result["errors"]), result["errors"])
            self.assertEqual(victim.read_text(), "No consensus evidence")

    def test_run_paths_reject_traversal_symlink_escape_and_loops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            root.mkdir()
            victim = Path(tmp) / "victim"
            victim.mkdir()
            original = victim / "state.json"
            original.write_text('{"unchanged": true}')
            (root / "linked").symlink_to(victim, target_is_directory=True)
            (root / "loop").symlink_to("loop", target_is_directory=True)
            for run_id in ("../victim", str(victim), "nested/run", "linked", "loop", ".", ".."):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    state.load_state(run_id, root)
                self.assertEqual(original.read_text(), '{"unchanged": true}')
            self.assertEqual(state.run_path("legacy run.v1", root), root / "legacy run.v1")


    def test_round_limit_defaults_and_legacy_state_remain_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Round limit", state_root=tmp)
            self.assertEqual(payload["state"]["max_rounds"], 5)
            state_file = Path(payload["state_path"])
            older_state = payload["state"]
            older_state.pop("max_rounds")
            state.write_json(state_file, older_state)
            original = state_file.read_bytes()

            self.assertEqual(state.status_payload(payload["run_id"], tmp)["state"], older_state)
            self.assertTrue(state.validate_run(payload["run_id"], tmp)["ok"])
            self.assertEqual(state_file.read_bytes(), original)

    def test_invalid_round_limits_leave_no_partial_run(self) -> None:
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, "max_rounds must be a positive integer"):
                    state.init_run("Invalid limit", state_root=tmp, max_rounds=limit)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_clarification_requires_a_question_and_selected_requester(self) -> None:
        cases = (
            None,
            "not an object",
            {"status": "pending", "requested_by": "a", "question": ""},
            {"status": "", "requested_by": "a", "question": "Which scope?"},
            {"status": "pending", "requested_by": "", "question": "Which scope?"},
            {"status": "pending", "requested_by": "unselected-peer", "question": "Which scope?"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, _ = complete_consensus_run(tmp)
            original = state.load_state(payload["run_id"], tmp)
            for clarification in cases:
                with self.subTest(clarification=clarification):
                    malformed = deepcopy(original)
                    malformed.update(
                        status="paused_external_clarification", external_clarification=clarification,
                    )
                    with self.assertRaisesRegex(ValueError, "external_clarification"):
                        state.save_state(payload["run_id"], malformed, tmp)
                    state.write_json(Path(payload["state_path"]), malformed)
                    validation = state.validate_run(payload["run_id"], tmp)
                    self.assertFalse(validation["ok"], validation)

    def test_clarification_answer_survives_pause_and_resume(self) -> None:
        for access, peer_b in (
            ("native-subagent", "codex"), ("local-cli", "claude"),
        ):
            with self.subTest(access=access), tempfile.TemporaryDirectory() as tmp:
                payload = state.init_run("Clarify scope", state_root=tmp)
                peers, selection = peer_config.select_peers(
                    {"access": access, "peer_b": peer_b, "cwd": tmp},
                )
                question = {
                    "status": "pending", "requested_by": peers["b"]["id"], "question": "Which scope?",
                }
                state.update_state(
                    payload["run_id"], state_root=tmp, status="paused_external_clarification",
                    peers=peers, peer_selection_config=selection, task_behavior="read_only",
                    external_clarification=question,
                )
                self.assertTrue(state.validate_run(payload["run_id"], tmp)["ok"])
                answer = {**question, "status": "answered", "answer": "The current task only."}
                state.update_state(
                    payload["run_id"], state_root=tmp, status="in_progress",
                    external_clarification=answer,
                )
                self.assertEqual(
                    state.load_state(payload["run_id"], tmp)["external_clarification"], answer,
                )
                self.assertTrue(state.validate_run(payload["run_id"], tmp)["ok"])

    def test_extra_or_duplicate_peers_cannot_be_saved_or_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, _ = complete_consensus_run(tmp)
            original = state.load_state(payload["run_id"], tmp)
            for case in ("extra", "duplicate", "empty-id"):
                with self.subTest(case=case):
                    malformed = deepcopy(original)
                    if case == "extra":
                        malformed["peers"]["c"] = deepcopy(malformed["peers"]["a"])
                    elif case == "duplicate":
                        malformed["peers"]["b"]["id"] = malformed["peers"]["a"]["id"]
                    else:
                        malformed["peers"]["b"]["id"] = ""
                    malformed["peer_selection_config"]["selected_peer_ids"] = [
                        malformed["peers"][slot]["id"] for slot in ("a", "b")
                    ]
                    with self.assertRaises(ValueError):
                        state.save_state(payload["run_id"], malformed, tmp)
                    state.write_json(Path(payload["state_path"]), malformed)
                    self.assertFalse(state.validate_run(payload["run_id"], tmp)["ok"])

    def test_consensus_round_count_must_match_current_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, _ = complete_consensus_run(tmp)
            state.update_state(payload["run_id"], state_root=tmp, current_round_index=3)

            validation = state.validate_run(payload["run_id"], tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status.rounds_completed must match current_round_index", validation["errors"],
            )

    def test_exchange_cannot_claim_a_future_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, run_path, evidence = complete_consensus_run(tmp)
            evidence["exchange"]["round"] = 3
            for slot in ("a", "b"):
                artifact = f"rounds/round-3/peer-{slot}.md"
                write_artifact(run_path, artifact, "Synthetic later response.\n")
                evidence["exchange"][f"peer_{slot}_response_artifact"] = artifact
            rewrite_consensus_status(Path(payload["state_path"]), evidence)

            validation = state.validate_run(payload["run_id"], tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status.exchange.round must not exceed rounds_completed", validation["errors"],
            )

    def test_mutable_review_cannot_claim_a_future_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, evidence = complete_consensus_run(tmp, task_behavior="shared_mutable_output")
            evidence["mutable_work"]["review_round"] = 4
            rewrite_consensus_status(Path(payload["state_path"]), evidence)

            validation = state.validate_run(payload["run_id"], tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status.mutable_work.review_round must not exceed rounds_completed",
                validation["errors"],
            )

    def test_slug_and_run_id_are_stable(self) -> None:
        self.assertEqual(state.slugify_slug(" Pair: Plan Review! "), "pair-plan-review")
        self.assertEqual(state.slugify_slug(""), "pair-run")

        run_id = state.generate_run_id(
            "Pair Plan Review",
            now=datetime(2026, 5, 5, 1, 2, 3, tzinfo=timezone.utc),
        )

        self.assertEqual(run_id, "pair-20260505T010203Z-pair-plan-review")

    def test_init_run_creates_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run(
                "Pair plan review",
                slug="plan-review",
                state_root=tmp,
                now=datetime(2026, 5, 5, 1, 2, 3, tzinfo=timezone.utc),
            )

            run_path = Path(payload["run_path"])
            self.assertEqual(payload["run_id"], "pair-20260505T010203Z-plan-review")
            for directory in contract.REQUIRED_DIRECTORIES:
                self.assertTrue((run_path / directory).is_dir(), directory)
            for artifact in contract.REQUIRED_INITIAL_ARTIFACTS:
                self.assertTrue((run_path / artifact).is_file(), artifact)

            loaded = state.load_state(payload["run_id"], state_root=tmp)
            self.assertEqual(loaded["workflow_id"], "pair")
            self.assertEqual(loaded["status"], "initialized")
            self.assertIsNone(loaded["peer_selection_config"])

            manifest = json.loads((run_path / "artifacts.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_contract"]["round_note_files"], ["peer-a.md", "peer-b.md"])
            self.assertEqual(
                manifest["artifact_contract"]["terminal_artifacts"],
                {"consensus": "consensus.md", "no_consensus": "no_consensus.md"},
            )
            self.assertEqual(
                manifest["artifact_contract"]["required_consensus_status_fields"],
                list(contract.REQUIRED_CONSENSUS_STATUS_FIELDS),
            )
            self.assertEqual(
                manifest["artifact_contract"]["required_mutable_work_fields"],
                list(contract.REQUIRED_MUTABLE_WORK_FIELDS),
            )

            status = state.status_payload(payload["run_id"], state_root=tmp)
            self.assertTrue(status["ok"])
            self.assertEqual(status["state"]["run_id"], payload["run_id"])

            validation = state.validate_run(payload["run_id"], state_root=tmp)
            self.assertTrue(validation["ok"], validation["errors"])

    def test_validate_run_reports_missing_required_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            Path(payload["transcript_path"]).unlink()

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn("missing file: transcript.md", validation["errors"])

    def test_validate_run_requires_terminal_artifact_for_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            state_file = Path(payload["state_path"])
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            state_payload["status"] = "consensus"
            state.write_json(state_file, state_payload)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn("missing terminal artifact: consensus.md", validation["errors"])

    def test_validate_run_requires_consensus_evidence_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            run_path = Path(payload["run_path"])
            write_artifact(run_path, contract.CONSENSUS_ARTIFACT, "Consensus.\n")
            peers, selection = peer_selection()
            state.update_state(
                payload["run_id"],
                state_root=tmp,
                status="consensus",
                current_round_index=2,
                peers=peers,
                peer_selection_config=selection,
                peer_selection_rationale="Selected from config.",
                task_behavior="read_only",
            )

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status must be an object when status is consensus",
                validation["errors"],
            )

    def test_validate_run_rejects_one_round_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, consensus_status = complete_consensus_run(tmp)
            state_file = Path(payload["state_path"])
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            state_payload["current_round_index"] = 1
            consensus_status["rounds_completed"] = 1
            state_payload["consensus_status"] = consensus_status
            state.write_json(state_file, state_payload)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn("state current_round_index must be at least 2 for consensus", validation["errors"])
            self.assertIn("state consensus_status.rounds_completed must be at least 2", validation["errors"])

    def test_validate_run_rejects_consensus_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, consensus_status = complete_consensus_run(tmp)
            consensus_status["final_text_sha256"] = "0" * 64
            rewrite_consensus_status(Path(payload["state_path"]), consensus_status)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status.final_text_sha256 must match consensus.md SHA-256",
                validation["errors"],
            )

    def test_validate_run_rejects_missing_referenced_consensus_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, run_path, _ = complete_consensus_run(tmp)
            (run_path / "rounds/round-2/peer-b.md").unlink()

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "missing artifact for state consensus_status.exchange.peer_b_response_artifact: "
                "rounds/round-2/peer-b.md",
                validation["errors"],
            )

    def test_validate_run_rejects_round_one_notes_reused_as_exchange_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            run_path = Path(payload["run_path"])

            write_artifact(run_path, "rounds/round-1/peer-a.md", "Peer A independent note.\n")
            write_artifact(run_path, "rounds/round-1/peer-b.md", "Peer B independent note.\n")
            final_text = "Consensus final text.\n"
            write_artifact(run_path, contract.CONSENSUS_ARTIFACT, final_text)
            final_hash = sha256_text(final_text)

            peers, selection = peer_selection()
            state.update_state(
                payload["run_id"],
                state_root=tmp,
                status="consensus",
                current_round_index=2,
                peers=peers,
                peer_selection_config=selection,
                peer_selection_rationale="Selected from config.",
                task_behavior="read_only",
                consensus_status={
                    "result": "consensus",
                    "rounds_completed": 2,
                    "final_text_artifact": contract.CONSENSUS_ARTIFACT,
                    "final_text_sha256": final_hash,
                    "independent_notes": {
                        "round": 1,
                        "peer_a_artifact": "rounds/round-1/peer-a.md",
                        "peer_b_artifact": "rounds/round-1/peer-b.md",
                    },
                    "exchange": {
                        "round": 2,
                        "peer_a_received_artifact": "rounds/round-1/peer-b.md",
                        "peer_b_received_artifact": "rounds/round-1/peer-a.md",
                        "peer_a_response_artifact": "rounds/round-1/peer-a.md",
                        "peer_b_response_artifact": "rounds/round-1/peer-b.md",
                    },
                    "peer_acceptances": {
                        "a": {
                            "accepted": True,
                            "final_text_artifact": contract.CONSENSUS_ARTIFACT,
                            "final_text_sha256": final_hash,
                        },
                        "b": {
                            "accepted": True,
                            "final_text_artifact": contract.CONSENSUS_ARTIFACT,
                            "final_text_sha256": final_hash,
                        },
                    },
                },
            )

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status.exchange.peer_a_response_artifact "
                "must be distinct from independent note artifacts",
                validation["errors"],
            )
            self.assertIn(
                "state consensus_status.exchange.peer_b_response_artifact "
                "must be distinct from independent note artifacts",
                validation["errors"],
            )
            self.assertIn(
                "state consensus_status.exchange.peer_a_response_artifact "
                "must be rounds/round-2/peer-a.md",
                validation["errors"],
            )
            self.assertIn(
                "state consensus_status.exchange.peer_b_response_artifact "
                "must be rounds/round-2/peer-b.md",
                validation["errors"],
            )

    def test_validate_run_accepts_complete_read_only_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, _ = complete_consensus_run(tmp)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertTrue(validation["ok"], validation["errors"])

    def test_validate_run_requires_mutable_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, consensus_status = complete_consensus_run(tmp, task_behavior="shared_mutable_output")
            mutable_work = consensus_status["mutable_work"]
            self.assertIsInstance(mutable_work, dict)
            del mutable_work["reviewer_evidence_artifact"]
            rewrite_consensus_status(Path(payload["state_path"]), consensus_status)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status.mutable_work missing field: reviewer_evidence_artifact",
                validation["errors"],
            )

    def test_validate_run_rejects_mutable_same_writer_and_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, consensus_status = complete_consensus_run(tmp, task_behavior="shared_mutable_output")
            mutable_work = consensus_status["mutable_work"]
            self.assertIsInstance(mutable_work, dict)
            mutable_work["reviewer_peer"] = "a"
            mutable_work["reviewer_evidence_artifact"] = "rounds/round-2/peer-a.md"
            rewrite_consensus_status(Path(payload["state_path"]), consensus_status)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state consensus_status.mutable_work writer_peer and reviewer_peer must be distinct",
                validation["errors"],
            )

    def test_validate_run_accepts_complete_mutable_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload, _, _ = complete_consensus_run(tmp, task_behavior="shared_mutable_output")

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertTrue(validation["ok"], validation["errors"])

    def test_validate_run_requires_peer_selection_after_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            state_file = Path(payload["state_path"])
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            state_payload["status"] = "in_progress"
            state.write_json(state_file, state_payload)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn(
                "state peer_selection_config must be an object before Pair work starts",
                validation["errors"],
            )

    def test_validate_run_rejects_stale_peer_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            state_file = Path(payload["state_path"])
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            peers, selection = peer_selection()
            peers["a"]["capability"] = "mcp.remote-agent-broker"
            peers["b"]["capability"] = "remote-agent-broker"
            state_payload.update(
                {
                    "status": "in_progress",
                    "peers": peers,
                    "peer_selection_config": selection,
                    "peer_selection_rationale": "Manually restored stale state.",
                    "task_behavior": "read_only",
                }
            )
            state.write_json(state_file, state_payload)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

        allowed = ", ".join(contract.PEER_CAPABILITIES)
        self.assertFalse(validation["ok"])
        self.assertIn(f"state peers.a.capability must be {allowed}", validation["errors"])
        self.assertIn(f"state peers.b.capability must be {allowed}", validation["errors"])

    def test_validate_run_rejects_native_claude_peer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            state_file = Path(payload["state_path"])
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            peers, selection = peer_selection()
            peers["a"].update(
                {"id": "native-a-codex", "capability": "native-subagent"}
            )
            peers["b"].update(
                {"id": "native-b-claude", "capability": "native-subagent"}
            )
            selection["selected_peer_ids"] = [
                peers["a"]["id"],
                peers["b"]["id"],
            ]
            state_payload.update(
                {
                    "status": "in_progress",
                    "peers": peers,
                    "peer_selection_config": selection,
                    "peer_selection_rationale": "Manually restored native state.",
                    "task_behavior": "read_only",
                }
            )
            with self.assertRaisesRegex(
                ValueError,
                "state peers.b.agent must be codex for native-subagent capability",
            ):
                state.save_state(payload["run_id"], state_payload, state_root=tmp)
            state.write_json(state_file, state_payload)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

        self.assertFalse(validation["ok"])
        self.assertIn(
            "state peers.b.agent must be codex for native-subagent capability",
            validation["errors"],
        )

    def test_validate_run_rejects_mixed_peer_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            state_file = Path(payload["state_path"])
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            peers, selection = peer_selection()
            peers["a"].update(
                {"id": "native-a-codex", "capability": "native-subagent"}
            )
            selection["selected_peer_ids"] = [
                peers["a"]["id"],
                peers["b"]["id"],
            ]
            state_payload.update(
                {
                    "status": "in_progress",
                    "peers": peers,
                    "peer_selection_config": selection,
                    "peer_selection_rationale": "Manually restored mixed state.",
                    "task_behavior": "read_only",
                }
            )
            state.write_json(state_file, state_payload)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

        self.assertFalse(validation["ok"])
        self.assertIn(
            "state peers.a and peers.b capability must match",
            validation["errors"],
        )

    def test_validate_run_requires_no_consensus_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            run_path = Path(payload["run_path"])
            (run_path / contract.NO_CONSENSUS_ARTIFACT).write_text("No consensus.\n", encoding="utf-8")
            state_file = Path(payload["state_path"])
            state_payload = json.loads(state_file.read_text(encoding="utf-8"))
            state_payload["status"] = "no_consensus"
            state.write_json(state_file, state_payload)

            validation = state.validate_run(payload["run_id"], state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn("state no_consensus must be an object when status is no_consensus", validation["errors"])

    def test_validate_run_accepts_complete_no_consensus_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)
            run_path = Path(payload["run_path"])
            (run_path / contract.NO_CONSENSUS_ARTIFACT).write_text("No consensus.\n", encoding="utf-8")
            peers, selection = peer_selection()
            updated = state.update_state(
                payload["run_id"],
                state_root=tmp,
                status="no_consensus",
                peers=peers,
                peer_selection_config=selection,
                peer_selection_rationale="Selected from config.",
                task_behavior="read_only",
                no_consensus={
                    "unresolved_issue": "Which approach should ship?",
                    "peer_a_position": "Ship A.",
                    "peer_b_position": "Ship B.",
                    "peer_a_evidence": ["A test result"],
                    "peer_b_evidence": ["B review result"],
                    "human_decision_needed": "Choose A or B.",
                    "continuation_options": ["More rounds", "Different Peer"],
                },
            )

            self.assertEqual(updated["status"], "no_consensus")
            validation = state.validate_run(payload["run_id"], state_root=tmp)
            self.assertTrue(validation["ok"], validation["errors"])

    def test_update_state_rejects_invalid_status_and_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = state.init_run("Pair plan review", slug="plan-review", state_root=tmp)

            with self.assertRaises(ValueError):
                state.update_state(payload["run_id"], state_root=tmp, status="done")
            with self.assertRaises(ValueError):
                state.update_state(payload["run_id"], state_root=tmp, task_behavior="unknown")
            with self.assertRaises(ValueError):
                state.update_state(payload["run_id"], state_root=tmp, status="in_progress")

    def test_validate_run_reports_invalid_run_id_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_run = Path(tmp) / "not-a-pair-run"
            bad_run.mkdir()
            for directory in contract.REQUIRED_DIRECTORIES:
                (bad_run / directory).mkdir(parents=True)
            state.write_json(
                bad_run / "state.json",
                state.initial_state("not-a-pair-run", "Bad run id"),
            )
            state.write_json(bad_run / "artifacts.json", state.artifact_manifest("not-a-pair-run"))
            (bad_run / "transcript.md").write_text("Transcript.\n", encoding="utf-8")

            validation = state.validate_run("not-a-pair-run", state_root=tmp)

            self.assertFalse(validation["ok"])
            self.assertIn("run id does not match Pair run id format", validation["errors"])


if __name__ == "__main__":
    unittest.main()
