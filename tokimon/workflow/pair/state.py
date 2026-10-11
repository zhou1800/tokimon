"""Deterministic state and artifact helpers for Pair workflow runs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokimon.runtime import resolve_runtime_path, source_root
from tokimon.workflow.pair import contract


RUN_ID_RE = re.compile(contract.RUN_ID_PATTERN)
WORKSPACE_ROOT = source_root()


def slugify_slug(value: str | None) -> str:
    """Return a stable human-readable slug segment for a Pair run id."""
    if not value:
        return "pair-run"
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "pair-run"


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def timestamp_text(now: datetime | None = None) -> str:
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id_timestamp(now: datetime | None = None) -> str:
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generate_run_id(
    task_summary: str | None = None,
    *,
    slug: str | None = None,
    now: datetime | None = None,
) -> str:
    slug_part = slugify_slug(slug or task_summary)
    return f"{contract.WORKFLOW_ID}-{run_id_timestamp(now)}-{slug_part}"


def resolve_run_root(state_root: str | Path | None = None) -> Path:
    path = Path(state_root) if state_root is not None else contract.RUN_ROOT
    return resolve_runtime_path(path, workspace_root=WORKSPACE_ROOT)


def run_path(run_id: str, state_root: str | Path | None = None) -> Path:
    if (
        not isinstance(run_id, str) or not run_id.strip()
        or run_id in {".", ".."} or "/" in run_id or "\\" in run_id
        or Path(run_id).is_absolute()
    ):
        raise ValueError("run_id must be a non-empty path component")
    root = resolve_run_root(state_root)
    target = root / run_id
    try:
        contained = target.resolve().is_relative_to(root.resolve())
    except RuntimeError as exc:
        raise ValueError("run_id cannot resolve its configured path") from exc
    if not contained:
        raise ValueError("run_id escapes its configured run root")
    return target


def _owned_artifact_path(target: Path, relative: str | Path) -> Path:
    """Contain run-owned state and artifacts while allowing contained aliases."""
    path = target / relative
    try:
        contained = path.resolve().is_relative_to(target.resolve())
    except RuntimeError as exc:
        raise ValueError("Pair artifact path cannot be resolved") from exc
    if not contained:
        raise ValueError("Pair artifact path must stay inside the Pair run directory")
    return path


def state_path(run_id: str, state_root: str | Path | None = None) -> Path:
    return _owned_artifact_path(run_path(run_id, state_root), "state.json")


def _relative_artifact(path: str, kind: str) -> dict[str, str]:
    return {"kind": kind, "path": path}


def initial_state(
    run_id: str,
    task_summary: str,
    *,
    max_rounds: int = contract.DEFAULT_MAX_ROUNDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not _is_plain_int(max_rounds) or max_rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    stamp = timestamp_text(now)
    return {
        "run_id": run_id,
        "workflow_id": contract.WORKFLOW_ID,
        "task_summary": task_summary,
        "status": "initialized",
        "created_at": stamp,
        "updated_at": stamp,
        "current_round_index": 0,
        "max_rounds": max_rounds,
        "peers": {"a": None, "b": None},
        "peer_selection_config": None,
        "peer_selection_rationale": None,
        "task_behavior": None,
        "active_prompt_artifact": None,
        "transcript_artifact": "transcript.md",
        "artifact_manifest": "artifacts.json",
        "external_clarification": None,
        "consensus_status": None,
        "final_delivery_bundle": None,
    }


def artifact_manifest(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_contract": {
            "required_initial_artifacts": list(contract.REQUIRED_INITIAL_ARTIFACTS),
            "required_directories": list(contract.REQUIRED_DIRECTORIES),
            "round_note_files": list(contract.ROUND_NOTE_FILES),
            "terminal_artifacts": {
                "consensus": contract.CONSENSUS_ARTIFACT,
                "no_consensus": contract.NO_CONSENSUS_ARTIFACT,
            },
            "required_state_fields": list(contract.REQUIRED_STATE_FIELDS),
            "required_no_consensus_fields": list(contract.NO_CONSENSUS_FIELDS),
            "required_consensus_status_fields": list(contract.REQUIRED_CONSENSUS_STATUS_FIELDS),
            "required_consensus_independent_note_fields": list(
                contract.REQUIRED_CONSENSUS_INDEPENDENT_NOTE_FIELDS
            ),
            "required_consensus_exchange_fields": list(contract.REQUIRED_CONSENSUS_EXCHANGE_FIELDS),
            "required_consensus_peer_acceptance_fields": list(
                contract.REQUIRED_CONSENSUS_PEER_ACCEPTANCE_FIELDS
            ),
            "required_mutable_work_fields": list(contract.REQUIRED_MUTABLE_WORK_FIELDS),
        },
        "artifacts": [
            _relative_artifact("state.json", "state"),
            _relative_artifact("transcript.md", "transcript"),
            _relative_artifact("artifacts.json", "artifact_manifest"),
            _relative_artifact("prompts/", "prompt_directory"),
            _relative_artifact("rounds/", "round_directory"),
            _relative_artifact("peers/a/", "peer_workspace"),
            _relative_artifact("peers/b/", "peer_workspace"),
        ],
    }


def transcript_template(run_id: str, task_summary: str) -> str:
    return "\n".join(
        (
            f"# Pair Transcript: {run_id}",
            "",
            f"- workflow_id: {contract.WORKFLOW_ID}",
            f"- task_summary: {task_summary}",
            "",
            "Record Peer notes, exchanged objections, mediator routing events, and final outcome here.",
            "",
        )
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def init_run(
    task_summary: str,
    *,
    slug: str | None = None,
    max_rounds: int = contract.DEFAULT_MAX_ROUNDS,
    state_root: str | Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    run_id = generate_run_id(task_summary, slug=slug, now=now)
    state = initial_state(run_id, task_summary, max_rounds=max_rounds, now=now)
    target_run_path = run_path(run_id, state_root)
    target_run_path.mkdir(parents=True, exist_ok=False)
    for directory in contract.REQUIRED_DIRECTORIES:
        (target_run_path / directory).mkdir(parents=True, exist_ok=True)

    write_json(target_run_path / "state.json", state)
    write_json(target_run_path / "artifacts.json", artifact_manifest(run_id))
    (target_run_path / "transcript.md").write_text(transcript_template(run_id, task_summary), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_path": str(target_run_path),
        "state_path": str(target_run_path / "state.json"),
        "artifact_manifest_path": str(target_run_path / "artifacts.json"),
        "transcript_path": str(target_run_path / "transcript.md"),
        "state": state,
    }


def load_state(run_id: str, state_root: str | Path | None = None) -> dict[str, Any]:
    try:
        with state_path(run_id, state_root).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"unable to read state.json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("state.json must contain a JSON object")
    return payload


def save_state(
    run_id: str,
    state_payload: dict[str, Any],
    state_root: str | Path | None = None,
) -> dict[str, Any]:
    if state_payload.get("run_id") != run_id:
        raise ValueError("state run_id must match requested run id")
    if state_payload.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"state workflow_id must be {contract.WORKFLOW_ID}")
    if state_payload.get("status") not in contract.VALID_STATUSES:
        raise ValueError("state status is not a valid Pair status")
    errors: list[str] = []
    _validate_round_state(state_payload, errors)
    _validate_peer_selection_state(state_payload, errors)
    _validate_external_clarification_state(state_payload, errors)
    if errors:
        raise ValueError("; ".join(errors))
    write_json(state_path(run_id, state_root), state_payload)
    return state_payload


def update_state(
    run_id: str,
    *,
    state_root: str | Path | None = None,
    now: datetime | None = None,
    status: str | None = None,
    current_round_index: int | None = None,
    max_rounds: int | None = None,
    peers: dict[str, Any] | None = None,
    peer_selection_config: dict[str, Any] | None = None,
    peer_selection_rationale: str | None = None,
    task_behavior: str | None = None,
    active_prompt_artifact: str | None = None,
    external_clarification: dict[str, Any] | None = None,
    consensus_status: dict[str, Any] | None = None,
    final_delivery_bundle: dict[str, Any] | None = None,
    no_consensus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_state(run_id, state_root)
    if status is not None:
        if status not in contract.VALID_STATUSES:
            raise ValueError("status is not a valid Pair status")
        payload["status"] = status
    if current_round_index is not None:
        if current_round_index < 0:
            raise ValueError("current_round_index must be non-negative")
        payload["current_round_index"] = current_round_index
    if max_rounds is not None:
        payload["max_rounds"] = max_rounds
    if peers is not None:
        payload["peers"] = peers
    if peer_selection_config is not None:
        payload["peer_selection_config"] = peer_selection_config
    if peer_selection_rationale is not None:
        payload["peer_selection_rationale"] = peer_selection_rationale
    if task_behavior is not None:
        if task_behavior not in contract.TASK_BEHAVIORS:
            raise ValueError("task_behavior is not a valid Pair task behavior")
        payload["task_behavior"] = task_behavior
    if active_prompt_artifact is not None:
        payload["active_prompt_artifact"] = active_prompt_artifact
    if external_clarification is not None:
        payload["external_clarification"] = external_clarification
    if consensus_status is not None:
        payload["consensus_status"] = consensus_status
    if final_delivery_bundle is not None:
        payload["final_delivery_bundle"] = final_delivery_bundle
    if no_consensus is not None:
        payload["no_consensus"] = no_consensus
    payload["updated_at"] = timestamp_text(now)
    return save_state(run_id, payload, state_root)


def status_payload(run_id: str, state_root: str | Path | None = None) -> dict[str, Any]:
    state = load_state(run_id, state_root)
    target_run_path = run_path(run_id, state_root)
    return {
        "ok": True,
        "run_id": run_id,
        "run_path": str(target_run_path),
        "state": state,
        "artifacts": {
            "state": str(target_run_path / "state.json"),
            "artifact_manifest": str(target_run_path / "artifacts.json"),
            "transcript": str(target_run_path / "transcript.md"),
        },
    }


def _validate_json_file(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"missing file: {path.name}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        errors.append(f"unable to read JSON in {path.name}: {exc}")
        return None
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {path.name}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{path.name} must contain a JSON object")
        return None
    return payload


def _status_requires_peer_selection(status: Any) -> bool:
    return status in {"in_progress", "paused_external_clarification", "consensus", "no_consensus"}


def _allowed_peer_capability_text() -> str:
    return ", ".join(contract.PEER_CAPABILITIES)


def _validate_peer_capability(peer: dict[str, Any], slot: str, errors: list[str]) -> None:
    if peer.get("capability") not in contract.PEER_CAPABILITIES:
        errors.append(f"state peers.{slot}.capability must be {_allowed_peer_capability_text()}")


def _validate_peer_placement(
    peer_a: dict[str, Any],
    peer_b: dict[str, Any],
    errors: list[str],
) -> None:
    capability_a = peer_a.get("capability")
    capability_b = peer_b.get("capability")
    if capability_a not in contract.CODEX_CLAUDE_PEER_CAPABILITIES:
        errors.append("state peers.a.capability must be native-subagent or local-cli")
    if peer_a.get("agent") != contract.PEER_A_AGENT:
        errors.append("state peers.a.agent must be codex")

    if capability_a != capability_b:
        errors.append("state peers.a and peers.b capability must match")

    for slot, peer in (("a", peer_a), ("b", peer_b)):
        if peer.get("capability") == contract.NATIVE_SUBAGENT_CAPABILITY and peer.get("agent") != "codex":
            errors.append(
                f"state peers.{slot}.agent must be codex for native-subagent capability"
            )
        if (
            peer.get("capability") == contract.LOCAL_CLI_CAPABILITY
            and peer.get("agent") not in contract.CODEX_CLAUDE_PEER_AGENTS
        ):
            errors.append(f"state peers.{slot}.agent must be codex or claude for local-cli capability")


def _validate_peer_selection_state(state_payload: dict[str, Any], errors: list[str]) -> None:
    if not _status_requires_peer_selection(state_payload.get("status")):
        return

    selection = state_payload.get("peer_selection_config")
    if not isinstance(selection, dict):
        errors.append("state peer_selection_config must be an object before Pair work starts")
        return

    peers = state_payload.get("peers")
    if not isinstance(peers, dict):
        errors.append("state peers must be an object before Pair work starts")
        return
    if set(peers) != {"a", "b"}:
        errors.append("state peers must contain exactly a and b")

    peer_a = peers.get("a")
    peer_b = peers.get("b")
    if not isinstance(peer_a, dict) or not isinstance(peer_b, dict):
        errors.append("state peers.a and peers.b must be selected before Pair work starts")
        return
    peer_ids = [peer_a.get("id"), peer_b.get("id")]
    if any(not isinstance(peer_id, str) or not peer_id.strip() for peer_id in peer_ids):
        errors.append("state Peer identities must be non-empty strings")
    elif peer_ids[0] == peer_ids[1]:
        errors.append("state Peer identities must be distinct")
    _validate_peer_capability(peer_a, "a", errors)
    _validate_peer_capability(peer_b, "b", errors)
    _validate_peer_placement(peer_a, peer_b, errors)

    selected_peer_ids = selection.get("selected_peer_ids")
    if selected_peer_ids != [peer_a.get("id"), peer_b.get("id")]:
        errors.append("state peers must match peer_selection_config selected_peer_ids")

    if not selection.get("source_path"):
        errors.append("state peer_selection_config source_path is required")


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_round_state(state_payload: dict[str, Any], errors: list[str]) -> None:
    current_round = state_payload.get("current_round_index")
    if not _is_plain_int(current_round) or current_round < 0:
        errors.append("state current_round_index must be a non-negative integer")
    max_rounds = state_payload.get("max_rounds", contract.DEFAULT_MAX_ROUNDS)
    if not _is_plain_int(max_rounds) or max_rounds < 1:
        errors.append("state max_rounds must be a positive integer")


def _validate_external_clarification_state(state_payload: dict[str, Any], errors: list[str]) -> None:
    clarification = state_payload.get("external_clarification")
    if clarification is None:
        if state_payload.get("status") == "paused_external_clarification":
            errors.append("state external_clarification is required while paused for clarification")
        return
    if not isinstance(clarification, dict):
        errors.append("state external_clarification must be an object")
        return
    for field in contract.EXTERNAL_CLARIFICATION_FIELDS:
        value = clarification.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"state external_clarification.{field} must be a non-empty string")

    requesters = {"a", "b"}
    selection = state_payload.get("peer_selection_config")
    selected_ids = selection.get("selected_peer_ids") if isinstance(selection, dict) else None
    if isinstance(selected_ids, list):
        requesters.update(peer_id for peer_id in selected_ids if isinstance(peer_id, str))
    requested_by = clarification.get("requested_by")
    if isinstance(requested_by, str) and requested_by.strip() and requested_by not in requesters:
        errors.append("state external_clarification.requested_by must identify a selected Peer")


def _require_fields(payload: dict[str, Any], fields: tuple[str, ...], errors: list[str], label: str) -> None:
    for field in fields:
        if field not in payload:
            errors.append(f"{label} missing field: {field}")


def _run_relative_path(target_run_path: Path, value: Any, errors: list[str], label: str) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label} must be a relative artifact path")
        return None

    relative = Path(value)
    if relative.is_absolute():
        errors.append(f"{label} must be a relative artifact path")
        return None

    try:
        return _owned_artifact_path(target_run_path, relative).resolve()
    except ValueError as exc:
        errors.append(f"{label}: {exc}")
        return None


def _validate_non_empty_artifact(
    target_run_path: Path,
    value: Any,
    errors: list[str],
    label: str,
    *,
    expected_name: str | None = None,
) -> Path | None:
    path = _run_relative_path(target_run_path, value, errors, label)
    if path is None:
        return None

    if expected_name is not None and path.name != expected_name:
        errors.append(f"{label} must name {expected_name}")

    if not path.is_file():
        errors.append(f"missing artifact for {label}: {value}")
        return None
    if path.stat().st_size == 0:
        errors.append(f"empty artifact for {label}: {value}")
        return None
    return path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_consensus_artifact(target_run_path: Path, errors: list[str]) -> str | None:
    try:
        consensus_path = _owned_artifact_path(target_run_path, contract.CONSENSUS_ARTIFACT)
    except ValueError as exc:
        errors.append(str(exc))
        return None
    if not consensus_path.is_file():
        errors.append(f"missing terminal artifact: {contract.CONSENSUS_ARTIFACT}")
        return None
    if consensus_path.stat().st_size == 0:
        errors.append(f"empty terminal artifact: {contract.CONSENSUS_ARTIFACT}")
        return None
    return _sha256_file(consensus_path)


def _validate_independent_notes(
    consensus_status: dict[str, Any],
    target_run_path: Path,
    errors: list[str],
) -> dict[str, str]:
    independent_notes = consensus_status.get("independent_notes")
    if not isinstance(independent_notes, dict):
        errors.append("state consensus_status.independent_notes must be an object")
        return {}

    _require_fields(
        independent_notes,
        contract.REQUIRED_CONSENSUS_INDEPENDENT_NOTE_FIELDS,
        errors,
        "state consensus_status.independent_notes",
    )

    round_value = independent_notes.get("round")
    if not _is_plain_int(round_value) or round_value != 1:
        errors.append("state consensus_status.independent_notes.round must be 1")

    artifacts: dict[str, str] = {}
    note_fields = (
        ("a", "peer_a_artifact", contract.ROUND_NOTE_FILES[0]),
        ("b", "peer_b_artifact", contract.ROUND_NOTE_FILES[1]),
    )
    for slot, field, expected_name in note_fields:
        if field not in independent_notes:
            continue
        artifact = independent_notes[field]
        _validate_non_empty_artifact(
            target_run_path,
            artifact,
            errors,
            f"state consensus_status.independent_notes.{field}",
            expected_name=expected_name,
        )
        if isinstance(artifact, str):
            artifacts[slot] = artifact
    return artifacts


def _validate_exchange(
    consensus_status: dict[str, Any],
    independent_artifacts: dict[str, str],
    target_run_path: Path,
    errors: list[str],
) -> None:
    exchange = consensus_status.get("exchange")
    if not isinstance(exchange, dict):
        errors.append("state consensus_status.exchange must be an object")
        return

    _require_fields(
        exchange,
        contract.REQUIRED_CONSENSUS_EXCHANGE_FIELDS,
        errors,
        "state consensus_status.exchange",
    )

    round_value = exchange.get("round")
    if not _is_plain_int(round_value) or round_value < 2:
        errors.append("state consensus_status.exchange.round must be at least 2")
    rounds_completed = consensus_status.get("rounds_completed")
    if _is_plain_int(round_value) and _is_plain_int(rounds_completed) and round_value > rounds_completed:
        errors.append("state consensus_status.exchange.round must not exceed rounds_completed")

    received_checks = (
        ("peer_a_received_artifact", independent_artifacts.get("b"), contract.ROUND_NOTE_FILES[1]),
        ("peer_b_received_artifact", independent_artifacts.get("a"), contract.ROUND_NOTE_FILES[0]),
    )
    for field, expected, expected_name in received_checks:
        if field not in exchange:
            continue
        _validate_non_empty_artifact(
            target_run_path,
            exchange[field],
            errors,
            f"state consensus_status.exchange.{field}",
            expected_name=expected_name,
        )
        if expected is not None and exchange.get(field) != expected:
            errors.append(f"state consensus_status.exchange.{field} must match the other Peer independent note")

    independent_note_paths = {
        artifact for artifact in independent_artifacts.values() if isinstance(artifact, str)
    }
    response_round_path = None
    if _is_plain_int(round_value) and round_value >= 2:
        response_round_path = f"rounds/round-{round_value}"

    response_fields = (
        ("peer_a_response_artifact", contract.ROUND_NOTE_FILES[0]),
        ("peer_b_response_artifact", contract.ROUND_NOTE_FILES[1]),
    )
    for field, expected_name in response_fields:
        if field not in exchange:
            continue
        artifact = exchange[field]
        _validate_non_empty_artifact(
            target_run_path,
            artifact,
            errors,
            f"state consensus_status.exchange.{field}",
            expected_name=expected_name,
        )
        if not isinstance(artifact, str):
            continue
        if artifact in independent_note_paths:
            errors.append(
                f"state consensus_status.exchange.{field} must be distinct from independent note artifacts"
            )
        if response_round_path is not None:
            expected_artifact = f"{response_round_path}/{expected_name}"
            if artifact != expected_artifact:
                errors.append(f"state consensus_status.exchange.{field} must be {expected_artifact}")


def _validate_peer_acceptances(
    consensus_status: dict[str, Any],
    target_run_path: Path,
    final_text_sha256: str | None,
    errors: list[str],
) -> None:
    peer_acceptances = consensus_status.get("peer_acceptances")
    if not isinstance(peer_acceptances, dict):
        errors.append("state consensus_status.peer_acceptances must be an object")
        return

    for slot in ("a", "b"):
        acceptance = peer_acceptances.get(slot)
        label = f"state consensus_status.peer_acceptances.{slot}"
        if not isinstance(acceptance, dict):
            errors.append(f"{label} must be an object")
            continue

        _require_fields(
            acceptance,
            contract.REQUIRED_CONSENSUS_PEER_ACCEPTANCE_FIELDS,
            errors,
            label,
        )

        if acceptance.get("accepted") is not True:
            errors.append(f"{label}.accepted must be true")
        if acceptance.get("final_text_artifact") != contract.CONSENSUS_ARTIFACT:
            errors.append(f"{label}.final_text_artifact must be {contract.CONSENSUS_ARTIFACT}")
        if final_text_sha256 is not None and acceptance.get("final_text_sha256") != final_text_sha256:
            errors.append(f"{label}.final_text_sha256 must match consensus_status.final_text_sha256")

        evidence_artifact = acceptance.get("evidence_artifact")
        if evidence_artifact is not None:
            expected_name = contract.ROUND_NOTE_FILES[0] if slot == "a" else contract.ROUND_NOTE_FILES[1]
            _validate_non_empty_artifact(
                target_run_path,
                evidence_artifact,
                errors,
                f"{label}.evidence_artifact",
                expected_name=expected_name,
            )


def _validate_mutable_work(
    state_payload: dict[str, Any],
    target_run_path: Path,
    errors: list[str],
) -> None:
    consensus_status = state_payload["consensus_status"]
    mutable_work = consensus_status.get("mutable_work")
    if not isinstance(mutable_work, dict):
        errors.append("state consensus_status.mutable_work must be an object for shared_mutable_output")
        return

    _require_fields(
        mutable_work,
        contract.REQUIRED_MUTABLE_WORK_FIELDS,
        errors,
        "state consensus_status.mutable_work",
    )

    writer_peer = mutable_work.get("writer_peer")
    reviewer_peer = mutable_work.get("reviewer_peer")
    for field, value in (("writer_peer", writer_peer), ("reviewer_peer", reviewer_peer)):
        if value not in {"a", "b"}:
            errors.append(f"state consensus_status.mutable_work.{field} must be a or b")
    if writer_peer in {"a", "b"} and reviewer_peer in {"a", "b"} and writer_peer == reviewer_peer:
        errors.append("state consensus_status.mutable_work writer_peer and reviewer_peer must be distinct")
    write_round = mutable_work.get("write_round")
    review_round = mutable_work.get("review_round")
    if not _is_plain_int(write_round) or write_round < 1:
        errors.append("state consensus_status.mutable_work.write_round must be a positive integer")
    if not _is_plain_int(review_round) or review_round < 1:
        errors.append("state consensus_status.mutable_work.review_round must be a positive integer")
    if _is_plain_int(write_round) and _is_plain_int(review_round) and review_round <= write_round:
        errors.append("state consensus_status.mutable_work.review_round must be after write_round")
    rounds_completed = consensus_status.get("rounds_completed")
    for field, value in (("write_round", write_round), ("review_round", review_round)):
        if _is_plain_int(value) and _is_plain_int(rounds_completed) and value > rounds_completed:
            errors.append(f"state consensus_status.mutable_work.{field} must not exceed rounds_completed")

    if mutable_work.get("reviewer_approved") is not True:
        errors.append("state consensus_status.mutable_work.reviewer_approved must be true")

    artifact_checks = (
        ("writer_evidence_artifact", writer_peer),
        ("reviewer_evidence_artifact", reviewer_peer),
    )
    for field, peer_slot in artifact_checks:
        if field not in mutable_work:
            continue
        expected_name = None
        if peer_slot == "a":
            expected_name = contract.ROUND_NOTE_FILES[0]
        if peer_slot == "b":
            expected_name = contract.ROUND_NOTE_FILES[1]
        _validate_non_empty_artifact(
            target_run_path,
            mutable_work[field],
            errors,
            f"state consensus_status.mutable_work.{field}",
            expected_name=expected_name,
        )


def _validate_consensus_state(state_payload: dict[str, Any], target_run_path: Path, errors: list[str]) -> None:
    current_round_index = state_payload.get("current_round_index")
    if _is_plain_int(current_round_index) and current_round_index < 2:
        errors.append("state current_round_index must be at least 2 for consensus")

    actual_final_sha256 = _validate_consensus_artifact(target_run_path, errors)

    consensus_status = state_payload.get("consensus_status")
    if not isinstance(consensus_status, dict):
        errors.append("state consensus_status must be an object when status is consensus")
        return

    _require_fields(
        consensus_status,
        contract.REQUIRED_CONSENSUS_STATUS_FIELDS,
        errors,
        "state consensus_status",
    )

    if consensus_status.get("result") != "consensus":
        errors.append("state consensus_status.result must be consensus")

    rounds_completed = consensus_status.get("rounds_completed")
    if not _is_plain_int(rounds_completed) or rounds_completed < 2:
        errors.append("state consensus_status.rounds_completed must be at least 2")
    elif rounds_completed != current_round_index:
        errors.append("state consensus_status.rounds_completed must match current_round_index")

    if consensus_status.get("final_text_artifact") != contract.CONSENSUS_ARTIFACT:
        errors.append(f"state consensus_status.final_text_artifact must be {contract.CONSENSUS_ARTIFACT}")

    recorded_final_sha256 = consensus_status.get("final_text_sha256")
    if actual_final_sha256 is not None and recorded_final_sha256 != actual_final_sha256:
        errors.append("state consensus_status.final_text_sha256 must match consensus.md SHA-256")

    independent_artifacts = _validate_independent_notes(consensus_status, target_run_path, errors)
    _validate_exchange(consensus_status, independent_artifacts, target_run_path, errors)
    _validate_peer_acceptances(consensus_status, target_run_path, actual_final_sha256, errors)

    if state_payload.get("task_behavior") == "shared_mutable_output":
        _validate_mutable_work(state_payload, target_run_path, errors)


def validate_run(run_id: str, state_root: str | Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    try:
        target_run_path = run_path(run_id, state_root)
    except ValueError as exc:
        return {
            "ok": False, "run_id": run_id,
            "run_path": str((Path(state_root) if state_root is not None else contract.RUN_ROOT) / str(run_id)),
            "errors": [str(exc)], "checked": [],
        }

    if not target_run_path.exists():
        errors.append(f"missing run directory: {target_run_path}")
        return {
            "ok": False,
            "run_id": run_id,
            "run_path": str(target_run_path),
            "errors": errors,
            "checked": [],
        }

    checked: list[str] = []
    checked.append("run_id_format")
    if not RUN_ID_RE.match(run_id):
        errors.append("run id does not match Pair run id format")

    for directory in contract.REQUIRED_DIRECTORIES:
        checked.append(directory)
        try:
            path = _owned_artifact_path(target_run_path, directory)
            if not path.is_dir():
                errors.append(f"missing directory: {directory}")
        except ValueError as exc:
            errors.append(f"{directory}: {exc}")

    artifact_paths: dict[str, Path] = {}
    for artifact in contract.REQUIRED_INITIAL_ARTIFACTS:
        checked.append(artifact)
        try:
            path = _owned_artifact_path(target_run_path, artifact)
            artifact_paths[artifact] = path
            if not path.is_file():
                errors.append(f"missing file: {artifact}")
        except ValueError as exc:
            errors.append(f"{artifact}: {exc}")

    state = _validate_json_file(artifact_paths["state.json"], errors) if "state.json" in artifact_paths else None
    manifest = _validate_json_file(artifact_paths["artifacts.json"], errors) if "artifacts.json" in artifact_paths else None

    if state is not None:
        for field in contract.REQUIRED_STATE_FIELDS:
            if field not in state:
                errors.append(f"state missing field: {field}")
        if state.get("workflow_id") != contract.WORKFLOW_ID:
            errors.append(f"state workflow_id must be {contract.WORKFLOW_ID}")
        if state.get("run_id") != run_id:
            errors.append("state run_id must match requested run id")
        if state.get("status") not in contract.VALID_STATUSES:
            errors.append("state status is not a valid Pair status")
        _validate_round_state(state, errors)
        _validate_peer_selection_state(state, errors)
        _validate_external_clarification_state(state, errors)

        if state.get("status") == "consensus":
            _validate_consensus_state(state, target_run_path, errors)
        if state.get("status") == "no_consensus":
            try:
                terminal = _owned_artifact_path(target_run_path, contract.NO_CONSENSUS_ARTIFACT)
                if not terminal.is_file():
                    errors.append(f"missing terminal artifact: {contract.NO_CONSENSUS_ARTIFACT}")
            except ValueError as exc:
                errors.append(str(exc))
            no_consensus = state.get("no_consensus")
            if not isinstance(no_consensus, dict):
                errors.append("state no_consensus must be an object when status is no_consensus")
            else:
                for field in contract.NO_CONSENSUS_FIELDS:
                    if field not in no_consensus:
                        errors.append(f"state no_consensus missing field: {field}")

    if manifest is not None:
        if manifest.get("workflow_id") != contract.WORKFLOW_ID:
            errors.append(f"artifact manifest workflow_id must be {contract.WORKFLOW_ID}")
        if manifest.get("run_id") != run_id:
            errors.append("artifact manifest run_id must match requested run id")
        if not isinstance(manifest.get("artifacts"), list):
            errors.append("artifact manifest artifacts must be a list")
        artifact_contract = manifest.get("artifact_contract")
        if not isinstance(artifact_contract, dict):
            errors.append("artifact manifest artifact_contract must be an object")
        else:
            expected_contract_fields = {
                "required_initial_artifacts",
                "required_directories",
                "round_note_files",
                "terminal_artifacts",
                "required_state_fields",
                "required_no_consensus_fields",
                "required_consensus_status_fields",
                "required_consensus_independent_note_fields",
                "required_consensus_exchange_fields",
                "required_consensus_peer_acceptance_fields",
                "required_mutable_work_fields",
            }
            for field in expected_contract_fields:
                if field not in artifact_contract:
                    errors.append(f"artifact manifest artifact_contract missing field: {field}")

    return {
        "ok": not errors,
        "run_id": run_id,
        "run_path": str(target_run_path),
        "errors": errors,
        "checked": checked,
    }
