"""Current ledger for code-audit findings and human-review items."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokimon.workflow import audit_project as project
from tokimon.workflow.code_audit import contract


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project.root() / candidate


def display_path(path: Path) -> str:
    return str(repo_path(path).resolve())


def normalize_subject(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"`([^`]+)`", r"\1", lowered)
    lowered = re.sub(r"[^a-z0-9./_-]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered[:120] or "finding"


def make_fingerprint(scope_id: str, rule_id: str, audit_subject_id: str, normalized: str) -> str:
    return ":".join((scope_id, rule_id, audit_subject_id, normalize_subject(normalized)))


def empty_ledger() -> dict[str, Any]:
    return {
        "schema_version": contract.CURRENT_LEDGER_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "project_id": project.current().project_id,
        "updated_at": None,
        "entries": {},
        "stats": {},
    }


def load_ledger(path: Path | str) -> dict[str, Any]:
    ledger_path = repo_path(path)
    if not ledger_path.exists():
        return empty_ledger()
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ledger must be a JSON object")
    schema_version = payload.get("schema_version")
    supported_versions = (contract.CURRENT_LEDGER_SCHEMA_VERSION, *contract.LEGACY_LEDGER_SCHEMA_VERSIONS)
    if schema_version not in supported_versions:
        raise ValueError(f"unsupported ledger schema version: {schema_version}")
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected workflow_id in ledger: {payload.get('workflow_id')}")
    payload.setdefault("entries", {})
    if schema_version == contract.CURRENT_LEDGER_SCHEMA_VERSION:
        if payload.get("project_id") != project.current().project_id:
            raise ValueError("ledger belongs to a different audit project")
    entries = payload["entries"]
    if not isinstance(entries, Mapping):
        raise ValueError("ledger entries must be a JSON object")
    for fingerprint, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"ledger entry {fingerprint!r} must be a JSON object")
        for field in ("status", "rule_id", "audit_subject_id"):
            if field not in entry:
                raise ValueError(f"ledger entry {fingerprint!r} missing {field}")
        if not isinstance(entry.get("evidence", []), Iterable):
            raise ValueError(f"ledger entry {fingerprint!r} evidence must be iterable")
    return payload


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=contract.JSON_ENSURE_ASCII, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _has_positive_line(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def validate_evidence(evidence: Any) -> None:
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence) or not evidence:
        raise ValueError("evaluation missing exact evidence")
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"evidence item {index} must be an object")
        if not str(item.get("path", "")).strip():
            raise ValueError(f"evidence item {index} missing path")
        if not any(_has_positive_line(item.get(field)) for field in contract.EVIDENCE_LINE_FIELDS):
            raise ValueError(f"evidence item {index} missing positive line, line_start, or start_line")
        if not any(str(item.get(field, "")).strip() for field in contract.EVIDENCE_TEXT_FIELDS):
            raise ValueError(f"evidence item {index} missing text, excerpt, quote, or details")


def _read_evaluations_payload(path: Path | str | None) -> Any:
    if path is None:
        return []
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _extract_evaluations(payload: Any) -> list[dict[str, Any]]:
    evaluations = payload.get("evaluations") if isinstance(payload, Mapping) else payload
    if evaluations is None and isinstance(payload, Mapping):
        evaluations = payload.get("rows")
    if not isinstance(evaluations, list):
        raise ValueError("evaluations input must be a list or contain evaluations/rows")
    return [dict(evaluation) for evaluation in evaluations]


def load_evaluations(path: Path | str | None) -> list[dict[str, Any]]:
    return _extract_evaluations(_read_evaluations_payload(path))


def load_accepted_evaluations(path: Path | str) -> list[dict[str, Any]]:
    payload = _read_evaluations_payload(path)
    if isinstance(payload, Mapping) and payload.get("audit_authority") is False:
        raise ValueError("accepted evaluations input must not be a compile-reports support artifact")
    if isinstance(payload, Mapping) and "project" in payload:
        project.validate_binding(payload)
    return _extract_evaluations(payload)


def _evaluation_status(evaluation: Mapping[str, Any]) -> str:
    status = str(evaluation.get("status", "")).strip()
    if status not in contract.AUDIT_STATUSES:
        raise ValueError(f"invalid audit status: {status}")
    if status == "unchecked":
        raise ValueError("unchecked is not allowed in evaluations")
    return status


def validate_evaluation(
    evaluation: Mapping[str, Any],
    *,
    known_rule_ids: set[str] | None = None,
    known_subject_ids: set[str] | None = None,
    expected_scope_id: str | None = None,
    expected_run_id: str | None = None,
) -> str:
    rule_id = str(evaluation.get("rule_id", "")).strip()
    subject_id = str(evaluation.get("audit_subject_id", "")).strip()
    if not rule_id:
        raise ValueError("evaluation missing required field: rule_id")
    if not subject_id:
        raise ValueError("evaluation missing required field: audit_subject_id")
    if known_rule_ids is not None and rule_id not in known_rule_ids:
        raise ValueError(f"evaluation references unknown rule_id: {rule_id}")
    if known_subject_ids is not None and subject_id not in known_subject_ids:
        raise ValueError(f"evaluation references unknown audit_subject_id: {subject_id}")

    status = _evaluation_status(evaluation)
    if "project" in evaluation:
        project.validate_binding(evaluation)
    scope_id = str(evaluation.get("scope_id", "")).strip()
    run_id = str(evaluation.get("run_id", "")).strip()
    if not scope_id:
        raise ValueError("evaluation missing required field: scope_id")
    if not run_id:
        raise ValueError("evaluation missing required field: run_id")
    if expected_scope_id is not None and scope_id != expected_scope_id:
        raise ValueError(f"evaluation scope_id mismatch: expected {expected_scope_id}; got {scope_id}")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"evaluation run_id mismatch: expected {expected_run_id}; got {run_id}")
    evidence = evaluation.get("evidence", [])
    verification = evaluation.get("verification", [])
    summary = str(evaluation.get("summary", "")).strip()
    reason = str(evaluation.get("reason", "")).strip()
    review_prompt = str(evaluation.get("review_prompt", "")).strip()

    if status == "pass" and not evidence and not verification:
        raise ValueError("pass evaluations require evidence or verification")
    if status == "fail":
        if not summary:
            raise ValueError("fail evaluations require summary")
        validate_evidence(evidence)
    if status == "not_applicable" and not reason:
        raise ValueError("not_applicable evaluations require reason")
    if status == "needs_human_review":
        validate_evidence(evidence)
        if not review_prompt:
            raise ValueError("needs_human_review evaluations require review_prompt")
    return status


def _stats(entries: Mapping[str, Mapping[str, Any]], scope_id: str | None = None) -> dict[str, Any]:
    selected = [
        entry
        for entry in entries.values()
        if scope_id is None or entry.get("scope_id") == scope_id
    ]
    by_status = {status: 0 for status in contract.LEDGER_STATUSES}
    by_status.update(Counter(str(entry["status"]) for entry in selected))
    by_rule = Counter(str(entry["rule_id"]) for entry in selected)
    by_subject = Counter(str(entry["audit_subject_id"]) for entry in selected)
    return {
        "total": len(selected),
        "open_total": sum(by_status[status] for status in contract.OPEN_LEDGER_STATUSES),
        "resolved_total": sum(by_status[status] for status in contract.RESOLVED_LEDGER_STATUSES),
        "by_status": dict(by_status),
        "by_rule_id": dict(sorted(by_rule.items())),
        "by_audit_subject_id": dict(sorted(by_subject.items())),
    }


def _ledger_health(entries: Mapping[str, Mapping[str, Any]], scope_id: str | None = None) -> dict[str, Any]:
    affected: list[dict[str, Any]] = []
    missing_paths: set[str] = set()
    open_missing_paths: set[str] = set()
    for fingerprint, entry in entries.items():
        if scope_id is not None and entry.get("scope_id") != scope_id:
            continue
        entry_missing: set[str] = set()
        for evidence in entry.get("evidence", []):
            if not isinstance(evidence, Mapping):
                continue
            path = str(evidence.get("path", "")).strip()
            if path and not repo_path(path).exists():
                entry_missing.add(path)
        if not entry_missing:
            continue
        is_open = entry.get("status") in contract.OPEN_LEDGER_STATUSES
        missing_paths.update(entry_missing)
        if is_open:
            open_missing_paths.update(entry_missing)
        affected.append(
            {
                "fingerprint": fingerprint,
                "status": entry.get("status"),
                "missing_paths": sorted(entry_missing),
            }
        )
    return {
        "healthy": not open_missing_paths,
        "missing_evidence_path_count": len(missing_paths),
        "open_missing_evidence_path_count": len(open_missing_paths),
        "affected_entry_count": len(affected),
        "affected_entries": affected,
    }


def summarize_ledger(path: Path | str, scope_id: str | None = None) -> dict[str, Any]:
    ledger = load_ledger(path)
    entries = ledger.get("entries", {})
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "ledger_path": str(path),
        "scope_id": scope_id,
        "project_id": ledger.get("project_id"),
        "historical_unbound": ledger.get("schema_version") in contract.LEGACY_LEDGER_SCHEMA_VERSIONS,
        "updated_at": ledger.get("updated_at"),
        "stats": _stats(entries, scope_id),
        "health": _ledger_health(entries, scope_id),
        "entries": [
            entry for entry in entries.values() if scope_id is None or entry.get("scope_id") == scope_id
        ],
    }


def record_evaluations(
    *,
    evaluations: Sequence[Mapping[str, Any]],
    scope_id: str,
    run_id: str,
    ledger_path: Path | str,
    run_root: Path | str,
    rules_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    known_rule_ids = set(rules_by_id) if rules_by_id is not None else None
    now = utc_now()
    ledger = load_ledger(ledger_path)
    if ledger.get("entries") and ledger.get("schema_version") in contract.LEGACY_LEDGER_SCHEMA_VERSIONS:
        raise ValueError("nonempty historical ledger must be migrated and bound before recording")
    entries = dict(ledger.get("entries", {}))
    run_dir = project.contained_child(repo_path(run_root), run_id, label="run_id")
    run_findings_path = project.contained_child(run_dir, "findings.json", label="artifact")

    raw_findings: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    updated_fingerprints: list[str] = []
    resolved_fingerprints: list[str] = []
    seen_rows: set[tuple[str, str]] = set()
    for evaluation in evaluations:
        status = validate_evaluation(
            evaluation,
            known_rule_ids=known_rule_ids,
            expected_scope_id=scope_id,
            expected_run_id=run_id,
        )
        rule_id = str(evaluation["rule_id"])
        subject_id = str(evaluation["audit_subject_id"])
        row_key = (subject_id, rule_id)
        if row_key in seen_rows:
            raise ValueError(
                f"duplicate accepted disposition row: {subject_id} {rule_id}"
            )
        seen_rows.add(row_key)
        if status in contract.RESOLUTION_STATUS_BY_EVALUATION:
            resolved_status = contract.RESOLUTION_STATUS_BY_EVALUATION[status]
            row_resolved: list[str] = []
            for fingerprint, existing in list(entries.items()):
                if (
                    existing.get("scope_id") != scope_id
                    or existing.get("rule_id") != rule_id
                    or existing.get("audit_subject_id") != subject_id
                    or existing.get("status") not in contract.OPEN_LEDGER_STATUSES
                ):
                    continue
                updated = dict(existing)
                updated.update(
                    {
                        "status": resolved_status,
                        "resolution": str(evaluation.get("reason") or evaluation.get("summary") or "").strip(),
                        "resolved_at": now,
                        "resolved_by_run": run_id,
                        "last_seen_run": run_id,
                        "last_run_artifact": display_path(run_findings_path),
                        "updated_at": now,
                    }
                )
                entries[fingerprint] = updated
                updated_fingerprints.append(fingerprint)
                resolved_fingerprints.append(fingerprint)
                row_resolved.append(fingerprint)
            dispositions.append(
                dict(
                    evaluation,
                    ledger_action="resolve",
                    ledger_status=resolved_status,
                    resolved_fingerprints=row_resolved,
                )
            )
            continue

        normalized = evaluation.get("normalized_subject") or evaluation.get("summary") or subject_id
        fingerprint = str(evaluation.get("fingerprint") or "").strip()
        if fingerprint:
            expected_prefix = f"{scope_id}:{rule_id}:{subject_id}:"
            if not fingerprint.startswith(expected_prefix):
                raise ValueError(
                    "evaluation fingerprint does not match its scope/rule/subject row"
                )
            prior_entry = entries.get(fingerprint)
            if prior_entry is not None and (
                prior_entry.get("scope_id") != scope_id
                or prior_entry.get("rule_id") != rule_id
                or prior_entry.get("audit_subject_id") != subject_id
            ):
                raise ValueError(
                    "evaluation fingerprint is owned by a different ledger row"
                )
        else:
            fingerprint = make_fingerprint(
                scope_id,
                rule_id,
                subject_id,
                str(normalized),
            )
        rule = (rules_by_id or {}).get(rule_id, {})
        existing = entries.get(fingerprint, {})
        entry = {
            "fingerprint": fingerprint,
            "scope_id": scope_id,
            "rule_id": rule_id,
            "audit_subject_id": subject_id,
            "normalized_subject": normalize_subject(str(normalized)),
            "status": status,
            "summary": str(evaluation["summary"]),
            "reason": str(evaluation.get("reason", "")),
            "evidence": evaluation["evidence"],
            "verification": list(evaluation.get("verification", [])),
            "review_prompt": str(evaluation.get("review_prompt", "")),
            "last_rule_hash": evaluation.get("rule_hash") or rule.get("rule_hash"),
            "last_run_artifact": display_path(run_findings_path),
            "first_seen_run": existing.get("first_seen_run", run_id),
            "last_seen_run": run_id,
            "seen_count": int(existing.get("seen_count", 0)) + 1,
            "recorded_at": existing.get("recorded_at", now),
            "updated_at": now,
        }
        if "migration_sources" in existing:
            entry["migration_sources"] = existing["migration_sources"]
        entries[fingerprint] = entry
        updated_fingerprints.append(fingerprint)
        raw_findings.append(dict(evaluation, fingerprint=fingerprint))
        dispositions.append(
            dict(evaluation, fingerprint=fingerprint, ledger_action="open", ledger_status=status)
        )

    run_payload = {
        "schema_version": contract.RUN_ARTIFACT_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "project": project.current().to_dict(),
        "run_id": run_id,
        "scope_id": scope_id,
        "created_at": now,
        "finding_count": len(raw_findings),
        "resolution_count": len(resolved_fingerprints),
        "disposition_count": len(dispositions),
        "findings": raw_findings,
        "dispositions": dispositions,
        "current_ledger_path": display_path(repo_path(ledger_path)),
        "updated_fingerprints": updated_fingerprints,
        "resolved_fingerprints": resolved_fingerprints,
    }
    write_json(run_findings_path, run_payload)

    ledger["entries"] = entries
    ledger["schema_version"] = contract.CURRENT_LEDGER_SCHEMA_VERSION
    ledger["project_id"] = project.current().project_id
    ledger["updated_at"] = now
    ledger["stats"] = _stats(entries)
    write_json(ledger_path, ledger)

    return {
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": run_id,
        "scope_id": scope_id,
        "finding_count": len(raw_findings),
        "resolution_count": len(resolved_fingerprints),
        "disposition_count": len(dispositions),
        "updated_fingerprints": updated_fingerprints,
        "resolved_fingerprints": resolved_fingerprints,
        "ledger_path": str(ledger_path),
        "run_findings_path": display_path(run_findings_path),
        "stats": _stats(entries, scope_id),
    }
