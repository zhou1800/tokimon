"""Compact current-state ledger for document-audit findings."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from typing import Any

from tokimon.workflow import audit_project as project
from tokimon.workflow.document_audit import contract


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project.root() / candidate


def display_path(path: Path) -> str:
    return repo_path(path).resolve().as_posix()


def normalize_subject(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"`([^`]+)`", r"\1", lowered)
    lowered = re.sub(r"[^a-z0-9./_-]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered[:120] or "finding"


def make_fingerprint(scope_id: str, rule_id: str, file_path: str, subject: str) -> str:
    return "|".join((scope_id, rule_id, file_path, normalize_subject(subject)))


def empty_ledger() -> dict[str, Any]:
    return {
        "project_id": project.current().project_id,
        "schema_version": contract.CURRENT_LEDGER_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
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
    supported_versions = (
        contract.CURRENT_LEDGER_SCHEMA_VERSION,
        *contract.LEGACY_LEDGER_SCHEMA_VERSIONS,
    )
    if schema_version not in supported_versions:
        raise ValueError(f"unsupported ledger schema version: {schema_version}")
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected workflow_id in ledger: {payload.get('workflow_id')}")
    if schema_version == contract.CURRENT_LEDGER_SCHEMA_VERSION or payload.get("project_id"):
        if payload.get("project_id") != project.current().project_id:
            raise ValueError("ledger belongs to a different project")
    payload.setdefault("entries", {})
    entries = payload["entries"]
    if not isinstance(entries, Mapping):
        raise ValueError("ledger entries must be a JSON object")
    for fingerprint, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"ledger entry {fingerprint!r} must be a JSON object")
        if "status" not in entry:
            raise ValueError(f"ledger entry {fingerprint!r} missing status")
        if entry["status"] not in contract.FINDING_STATUSES:
            raise ValueError(f"ledger entry {fingerprint!r} has invalid status")
        if "severity" in entry and entry["severity"] not in contract.FINDING_SEVERITIES:
            raise ValueError(f"ledger entry {fingerprint!r} has invalid severity")
    return payload


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=contract.JSON_ENSURE_ASCII, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_findings(path: Path | str) -> list[dict[str, Any]]:
    findings_path = repo_path(path)
    payload = json.loads(findings_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and payload.get("audit_authority") is False:
        raise ValueError("accepted findings input must not be a compile-reports support artifact")
    if isinstance(payload, list):
        findings = payload
    else:
        findings = payload.get("findings")
    if not isinstance(findings, list):
        raise ValueError("findings input must be a list or an object with a findings list")
    return findings


def _has_positive_line(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def validate_evidence(evidence: Any) -> None:
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence) or not evidence:
        raise ValueError("finding missing exact evidence")
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"evidence item {index} must be an object")
        if not any(_has_positive_line(item.get(field)) for field in contract.EVIDENCE_LINE_FIELDS):
            raise ValueError(f"evidence item {index} missing positive line or start_line")
        if not any(str(item.get(field, "")).strip() for field in contract.EVIDENCE_TEXT_FIELDS):
            raise ValueError(f"evidence item {index} missing text, excerpt, quote, or details")


def validate_finding(finding: Mapping[str, Any], known_rule_ids: set[str] | None = None) -> None:
    for field in contract.REQUIRED_FINDING_FIELDS:
        if field == "evidence":
            continue
        if not str(finding.get(field, "")).strip():
            raise ValueError(f"finding missing required field: {field}")
    validate_evidence(finding.get("evidence"))
    if finding["severity"] not in contract.FINDING_SEVERITIES:
        raise ValueError(f"invalid finding severity: {finding['severity']}")
    if finding["status"] not in contract.FINDING_STATUSES:
        raise ValueError(f"invalid finding status: {finding['status']}")
    if known_rule_ids is not None and finding["rule_id"] not in known_rule_ids:
        raise ValueError(f"unknown rule_id: {finding['rule_id']}")


def _stats(entries: Mapping[str, Mapping[str, Any]], scope_id: str | None = None) -> dict[str, Any]:
    selected = [
        entry
        for entry in entries.values()
        if scope_id is None or entry.get("scope_id") == scope_id
    ]
    by_status = {status: 0 for status in contract.FINDING_STATUSES}
    by_severity = {severity: 0 for severity in contract.FINDING_SEVERITIES}
    for entry in selected:
        if "severity" not in entry:
            raise ValueError("ledger entry missing severity for summary")
        by_status[entry["status"]] += 1
        by_severity[entry["severity"]] += 1
    return {
        "total": len(selected),
        "by_status": by_status,
        "by_severity": by_severity,
        "open_must_fix": sum(
            1 for entry in selected if entry["status"] == "open" and entry["severity"] == "must_fix"
        ),
        "open_should_fix": sum(
            1 for entry in selected if entry["status"] == "open" and entry["severity"] == "should_fix"
        ),
    }


def _ledger_health(entries: Mapping[str, Mapping[str, Any]], scope_id: str | None = None) -> dict[str, Any]:
    affected: list[dict[str, str]] = []
    for fingerprint, entry in entries.items():
        if scope_id is not None and entry.get("scope_id") != scope_id:
            continue
        file_path = str(entry.get("file_path", "")).strip()
        if file_path and not repo_path(file_path).exists():
            affected.append(
                {
                    "fingerprint": fingerprint,
                    "file_path": file_path,
                    "status": str(entry.get("status", "")),
                }
            )
    open_affected = [entry for entry in affected if entry["status"] == "open"]
    return {
        "healthy": not open_affected,
        "missing_file_count": len({entry["file_path"] for entry in affected}),
        "open_missing_file_count": len({entry["file_path"] for entry in open_affected}),
        "affected_entry_count": len(affected),
        "affected_entries": affected,
    }


def summarize_ledger(path: Path | str, scope_id: str | None = None) -> dict[str, Any]:
    current = load_ledger(path)
    entries = current.get("entries", {})
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "project_id": current.get("project_id"),
        "historical_unbound": not bool(current.get("project_id")),
        "ledger_path": str(path),
        "scope_id": scope_id,
        "updated_at": current.get("updated_at"),
        "stats": _stats(entries, scope_id),
        "health": _ledger_health(entries, scope_id),
        "entries": [
            entry for entry in entries.values() if scope_id is None or entry.get("scope_id") == scope_id
        ],
    }


def record_findings(
    *,
    findings: Sequence[Mapping[str, Any]],
    scope_id: str,
    run_id: str,
    ledger_path: Path | str,
    run_root: Path | str,
    rules_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    known_rule_ids = set(rules_by_id) if rules_by_id is not None else None
    now = utc_now()
    current = load_ledger(ledger_path)
    if current.get("entries") and not current.get("project_id"):
        raise ValueError("import the legacy unbound ledger before recording findings")
    current.update(schema_version=contract.CURRENT_LEDGER_SCHEMA_VERSION, project_id=project.current().project_id)
    entries = dict(current.get("entries", {}))
    run_dir = project.contained_child(repo_path(run_root), run_id, label="run_id")
    run_findings_path = project.contained_child(run_dir, "findings.json", label="artifact")

    raw_findings: list[dict[str, Any]] = []
    updated_fingerprints: list[str] = []
    seen_fingerprints: set[str] = set()
    for finding in findings:
        validate_finding(finding, known_rule_ids)
        normalized = finding.get("normalized_subject") or normalize_subject(finding["summary"])
        rule_id = str(finding["rule_id"])
        file_path = str(finding["file_path"])
        fingerprint = str(finding.get("fingerprint") or "").strip()
        if fingerprint:
            expected_prefix = f"{scope_id}|{rule_id}|{file_path}|"
            if not fingerprint.startswith(expected_prefix):
                raise ValueError(
                    "finding fingerprint does not match its scope/rule/file owner"
                )
            prior_entry = entries.get(fingerprint)
            if prior_entry is not None and (
                prior_entry.get("scope_id") != scope_id
                or prior_entry.get("rule_id") != rule_id
                or prior_entry.get("file_path") != file_path
            ):
                raise ValueError(
                    "finding fingerprint is owned by a different ledger entry"
                )
        else:
            fingerprint = make_fingerprint(
                scope_id,
                rule_id,
                file_path,
                str(normalized),
            )
        if fingerprint in seen_fingerprints:
            raise ValueError(f"duplicate accepted finding fingerprint: {fingerprint}")
        seen_fingerprints.add(fingerprint)
        existing = entries.get(fingerprint, {})
        rule = (rules_by_id or {}).get(rule_id, {})
        entry = {
            **({"migration_sources": existing["migration_sources"]} if "migration_sources" in existing else {}),
            "fingerprint": fingerprint,
            "scope_id": scope_id,
            "rule_id": rule_id,
            "file_path": file_path,
            "normalized_subject": normalize_subject(str(normalized)),
            "severity": str(finding["severity"]),
            "status": str(finding["status"]),
            "summary": str(finding["summary"]),
            "resolution": str(finding.get("resolution", "")),
            "first_seen_run": existing.get("first_seen_run", run_id),
            "last_seen_run": run_id,
            "seen_count": int(existing.get("seen_count", 0)) + 1,
            "last_rule_hash": finding.get("rule_hash") or rule.get("rule_hash"),
            "last_run_artifact": display_path(run_findings_path),
            "updated_at": now,
        }
        if finding.get("recommended_action"):
            entry["recommended_action"] = str(finding["recommended_action"])
        entries[fingerprint] = entry
        updated_fingerprints.append(fingerprint)
        raw_findings.append(dict(finding, fingerprint=fingerprint))

    run_payload = {
        "project": project.current().to_dict(),
        "schema_version": contract.RUN_ARTIFACT_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": run_id,
        "scope_id": scope_id,
        "created_at": now,
        "finding_count": len(raw_findings),
        "findings": raw_findings,
        "current_ledger_path": display_path(repo_path(ledger_path)),
        "updated_fingerprints": updated_fingerprints,
    }
    current["entries"] = entries
    current["updated_at"] = now
    current["stats"] = _stats(entries)
    write_json(run_findings_path, run_payload)
    write_json(ledger_path, current)

    return {
        "workflow_id": contract.WORKFLOW_ID,
        "run_id": run_id,
        "scope_id": scope_id,
        "finding_count": len(raw_findings),
        "updated_fingerprints": updated_fingerprints,
        "ledger_path": str(ledger_path),
        "run_findings_path": display_path(run_findings_path),
        "stats": _stats(entries, scope_id),
    }
