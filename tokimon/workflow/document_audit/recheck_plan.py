"""Clean-unit lock and recheck-plan support for document-audit runs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tokimon.workflow.audit_project import contained_child
from tokimon.workflow.document_audit import audit_units, contract, ledger, matrix as matrix_module


LOCK_ACTION = "use_validated_clean_unit_lock"
SETTLED_ACTION = "use_validated_terminal_disposition"
WAIT_ACTION = "waiting_on_dependency"
FRESH_ACTION = "fresh_report_required"
CLEAN_STATUSES = frozenset(("pass", "not_applicable"))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=contract.JSON_ENSURE_ASCII,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path | str) -> Any:
    return json.loads(ledger.repo_path(path).read_text(encoding="utf-8"))


def _path_identity(path_value: str) -> dict[str, str]:
    return audit_units.path_identity(path_value)


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    file_path = str(row.get("file_path", "")).strip()
    rule_id = str(row.get("rule_id", "")).strip()
    if not file_path or not rule_id:
        raise ValueError("matrix rows require file_path and rule_id")
    return file_path, rule_id


def _load_matrix(path: Path | str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("matrix input must be a JSON object")
    if payload.get("schema_version") != contract.CHECKLIST_MATRIX_SCHEMA_VERSION:
        raise ValueError(f"unsupported matrix schema_version: {payload.get('schema_version')}")
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected matrix workflow_id: {payload.get('workflow_id')}")
    run_id = str(payload.get("run_id", "")).strip()
    scope_id = str(payload.get("scope_id", "")).strip()
    if not run_id or not scope_id:
        raise ValueError("matrix input requires run_id and scope_id")
    matrix_module.validate_audit_snapshot(payload)
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("matrix input must contain rows")
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise ValueError("matrix rows must be objects")
    rows = [dict(row) for row in raw_rows]
    row_documents = list(dict.fromkeys(_row_key(row)[0] for row in rows))
    raw_units = payload.get("audit_units")
    if raw_units is None:
        units = audit_units.default_units(row_documents)
    else:
        if not isinstance(raw_units, list) or any(not isinstance(unit, Mapping) for unit in raw_units):
            raise ValueError("matrix audit_units must be a list of objects")
        units = audit_units.normalize_units(raw_units)
    audit_units.validate_document_coverage(units, row_documents)
    unit_by_document = {
        document_path: unit["audit_unit_id"]
        for unit in units
        for document_path in unit["document_paths"]
    }
    seen_rows: set[tuple[str, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen_rows:
            raise ValueError(f"duplicate matrix row: {key[0]} {key[1]}")
        seen_rows.add(key)
        expected_unit_id = unit_by_document[key[0]]
        row_unit_id = str(row.get("audit_unit_id", expected_unit_id)).strip()
        if row_unit_id != expected_unit_id:
            raise ValueError(f"matrix row audit_unit_id does not match its document: {key[0]}")
        row["audit_unit_id"] = expected_unit_id
    return dict(payload, rows=rows, audit_units=units, document_paths=row_documents)


def _rows_by_unit(matrix_payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        str(unit["audit_unit_id"]): [] for unit in matrix_payload["audit_units"]
    }
    for row in matrix_payload["rows"]:
        rows[str(row["audit_unit_id"])].append(dict(row))
    if any(not unit_rows for unit_rows in rows.values()):
        empty = sorted(unit_id for unit_id, unit_rows in rows.items() if not unit_rows)
        raise ValueError(f"audit units contain no matrix rows: {', '.join(empty)}")
    return rows


def _normalized_unit_definition(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audit_unit_id": str(unit["audit_unit_id"]),
        "document_paths": sorted(str(path) for path in unit["document_paths"]),
        "bound_input_paths": sorted(str(path) for path in unit.get("bound_input_paths", [])),
    }


def _unit_signature(
    *,
    matrix_payload: Mapping[str, Any],
    unit: Mapping[str, Any],
    unit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    definition = _normalized_unit_definition(unit)
    applicable_rows: list[dict[str, str]] = []
    for row in unit_rows:
        rule_hash = str(row.get("rule_hash", "")).strip()
        if not rule_hash:
            raise ValueError(f"matrix row is missing rule_hash: {row['file_path']} {row['rule_id']}")
        applicable_rows.append(
            {
                "file_path": str(row["file_path"]),
                "rule_id": str(row["rule_id"]),
                "rule_hash": rule_hash,
                "rule_source_path": str(row.get("rule_source_path", "")),
            }
        )
    return {
        "project": matrix_payload["project"],
        "scope_id": str(matrix_payload["scope_id"]),
        "audit_unit_id": str(unit["audit_unit_id"]),
        "unit_definition_hash": _sha256_json(definition),
        "document_path_hashes": [
            _path_identity(path) for path in definition["document_paths"]
        ],
        "bound_input_path_hashes": [
            _path_identity(path) for path in definition["bound_input_paths"]
        ],
        "applicable_rule_rows": sorted(
            applicable_rows,
            key=lambda row: (row["file_path"], row["rule_id"]),
        ),
        "rule_source_snapshots": matrix_payload.get("rule_source_snapshots"),
        "source_ref": matrix_payload.get("source_ref"),
        "resolved_source_ref": matrix_payload.get("resolved_source_ref"),
    }


def _ledger_entries(ledger_path: Path | str) -> dict[str, dict[str, Any]]:
    current = ledger.load_ledger(ledger_path)
    entries = current.get("entries", {})
    if not isinstance(entries, Mapping):
        raise ValueError("ledger entries must be an object")
    return {
        str(fingerprint): dict(entry)
        for fingerprint, entry in entries.items()
        if isinstance(entry, Mapping)
    }


def _finding_fingerprints(
    *,
    entries: Mapping[str, Mapping[str, Any]],
    scope_id: str,
    document_paths: Sequence[str],
    statuses: Sequence[str],
) -> list[str]:
    selected = set(document_paths)
    selected_statuses = set(statuses)
    return sorted(
        str(fingerprint)
        for fingerprint, entry in entries.items()
        if str(entry.get("scope_id", "")) == scope_id
        and str(entry.get("status", "")) in selected_statuses
        and str(entry.get("file_path", "")) in selected
    )


def _open_findings(
    *,
    entries: Mapping[str, Mapping[str, Any]],
    scope_id: str,
    document_paths: Sequence[str],
) -> list[str]:
    return _finding_fingerprints(
        entries=entries,
        scope_id=scope_id,
        document_paths=document_paths,
        statuses=("open",),
    )


def _report_index(reports: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        key = _row_key(report)
        if key in indexed:
            raise ValueError(f"duplicate report coverage for clean-unit lock: {key[0]} {key[1]}")
        indexed[key] = dict(report)
    return indexed


def _source_report(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = str(report.get("report_path", "")).strip()
    if not report_path:
        raise ValueError("clean report is missing report_path")
    path = ledger.repo_path(report_path)
    if not path.is_file():
        raise ValueError(f"clean report path is missing: {report_path}")
    from tokimon.workflow.document_audit import reports as report_module

    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("clean-unit source report must be a JSON object")
    validated = report_module.validate_report(payload, report_path=path)
    if validated.get("schema_version") != contract.AUDIT_REPORT_SCHEMA_VERSION:
        raise ValueError("clean-unit locks require current-schema source reports")
    if _row_key(validated) != _row_key(report):
        raise ValueError("clean-unit source report identity changed")
    if str(validated["status"]) != str(report["status"]):
        raise ValueError("clean-unit source report status changed")
    source = {
        "file_path": str(validated["file_path"]),
        "rule_id": str(validated["rule_id"]),
        "status": str(validated["status"]),
        "run_id": str(validated["run_id"]),
        "scope_id": str(validated["scope_id"]),
        "audit_snapshot_id": str(validated["audit_snapshot_id"]),
        "report_path": ledger.display_path(path),
        "report_hash": _file_hash(path),
    }
    return source, validated


def _terminal_dispositions(
    *,
    reports: Sequence[Mapping[str, Any]],
    scope_id: str,
    entries: Mapping[str, Mapping[str, Any]],
    required_last_seen_run: str | None = None,
) -> list[dict[str, str]] | None:
    fail_reports = [report for report in reports if str(report.get("status", "")) == "fail"]
    if not fail_reports:
        return None
    dispositions: dict[str, dict[str, str]] = {}
    for report in fail_reports:
        findings = report.get("findings", [])
        if not isinstance(findings, list) or not findings:
            return None
        for finding in findings:
            if not isinstance(finding, Mapping):
                return None
            fingerprint = str(finding.get("fingerprint", "")).strip()
            if not fingerprint:
                fingerprint = ledger.make_fingerprint(
                    scope_id,
                    str(finding.get("rule_id", "")),
                    str(finding.get("file_path", "")),
                    str(finding.get("normalized_subject") or finding.get("summary", "")),
                )
            entry = entries.get(fingerprint)
            accepted_status = str(entry.get("status", "")) if entry is not None else ""
            if (
                entry is None
                or str(entry.get("scope_id", "")) != scope_id
                or str(entry.get("rule_id", "")) != str(finding.get("rule_id", ""))
                or str(entry.get("file_path", "")) != str(finding.get("file_path", ""))
                or accepted_status not in contract.TERMINAL_FINDING_STATUSES
                or (
                    required_last_seen_run is not None
                    and str(entry.get("last_seen_run", "")) != required_last_seen_run
                )
            ):
                return None
            disposition_identity = {
                "fingerprint": fingerprint,
                "scope_id": str(entry.get("scope_id", "")),
                "rule_id": str(entry.get("rule_id", "")),
                "file_path": str(entry.get("file_path", "")),
                "normalized_subject": str(entry.get("normalized_subject", "")),
                "status": str(entry.get("status", "")),
            }
            dispositions[fingerprint] = {
                "fingerprint": fingerprint,
                "status": accepted_status,
                "ledger_entry_hash": _sha256_json(disposition_identity),
            }
    return sorted(dispositions.values(), key=lambda item: item["fingerprint"])


def write_clean_unit_locks(
    *,
    matrix_path: Path | str,
    reports: Sequence[Mapping[str, Any]],
    output_dir: Path | str,
    reports_dir: Path | str,
    ledger_path: Path | str,
    recheck_plan_path: Path | str | None = None,
) -> dict[str, Any]:
    matrix_payload = _load_matrix(matrix_path)
    rows_by_unit = _rows_by_unit(matrix_payload)
    reports_by_key = _report_index(reports)
    entries = _ledger_entries(ledger_path)
    locks: list[dict[str, Any]] = []
    settled_units: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    states: dict[str, dict[str, Any]] = {}
    unit_by_id = {
        str(unit["audit_unit_id"]): unit for unit in matrix_payload["audit_units"]
    }
    unit_by_document = {
        str(document_path): str(unit["audit_unit_id"])
        for unit in matrix_payload["audit_units"]
        for document_path in unit["document_paths"]
    }
    plan_units: dict[str, dict[str, Any]] = {}
    if recheck_plan_path:
        plan_payload = revalidated_plan_payload(
            recheck_plan_path=recheck_plan_path,
            matrix_path=matrix_path,
        )
        plan_units = {
            str(unit["audit_unit_id"]): dict(unit)
            for unit in plan_payload["units"]
        }

    for unit in matrix_payload["audit_units"]:
        unit_id = str(unit["audit_unit_id"])
        unit_rows = rows_by_unit[unit_id]
        plan_unit = plan_units.get(unit_id, {})
        if plan_unit.get("action") == WAIT_ACTION:
            states[unit_id] = {
                "candidate_status": None,
                "reason_codes": ["waiting_on_dependency"],
                "blocked_by_unit_ids": list(plan_unit.get("blocked_by_unit_ids", [])),
                "open_finding_fingerprints": [],
            }
            continue
        missing = [row for row in unit_rows if _row_key(row) not in reports_by_key]
        open_fingerprints = _open_findings(
            entries=entries,
            scope_id=str(matrix_payload["scope_id"]),
            document_paths=unit["document_paths"],
        )
        reasons: list[str] = []
        if missing:
            reasons.append("incomplete_unit_coverage")
        if open_fingerprints:
            reasons.append("unit_has_open_findings")
        state: dict[str, Any] = {
            "candidate_status": None,
            "reason_codes": reasons,
            "open_finding_fingerprints": open_fingerprints,
        }
        if not reasons:
            try:
                source_pairs = [
                    _source_report(reports_by_key[_row_key(row)]) for row in unit_rows
                ]
                source_reports = sorted(
                    (pair[0] for pair in source_pairs),
                    key=lambda report: (report["file_path"], report["rule_id"]),
                )
                validated_reports = [pair[1] for pair in source_pairs]
                source_scopes = {str(report["scope_id"]) for report in validated_reports}
                source_snapshots = {str(report["audit_snapshot_id"]) for report in validated_reports}
                source_runs = {str(report["run_id"]) for report in validated_reports}
                if source_scopes != {str(matrix_payload["scope_id"])}:
                    raise ValueError("unit source reports do not match the matrix scope")
                if len(source_snapshots) != 1 or len(source_runs) != 1:
                    raise ValueError("unit source reports do not share one audit snapshot")
                if plan_unit.get("action", FRESH_ACTION) == FRESH_ACTION:
                    if source_runs != {str(matrix_payload["run_id"])}:
                        raise ValueError("fresh unit source reports do not match the current matrix run")
                    if source_snapshots != {str(matrix_payload["audit_snapshot_id"])}:
                        raise ValueError("fresh unit source reports do not match the current audit snapshot")

                terminal_dispositions = _terminal_dispositions(
                    reports=validated_reports,
                    scope_id=str(matrix_payload["scope_id"]),
                    entries=entries,
                    required_last_seen_run=(
                        next(iter(source_runs))
                        if plan_unit.get("action", FRESH_ACTION) == FRESH_ACTION
                        else None
                    ),
                )
                statuses = {str(report["status"]) for report in validated_reports}
                if statuses.issubset(CLEAN_STATUSES):
                    state["candidate_status"] = "clean"
                elif terminal_dispositions is not None:
                    state["candidate_status"] = "settled"
                    state["terminal_dispositions"] = terminal_dispositions
                else:
                    state["reason_codes"] = ["unit_has_non_terminal_findings"]
                state["source_reports"] = source_reports
                state["signature"] = _unit_signature(
                    matrix_payload=matrix_payload,
                    unit=unit,
                    unit_rows=unit_rows,
                )
            except (OSError, ValueError) as exc:
                state["reason_codes"] = ["lock_input_validation_failed"]
                state["error"] = str(exc)
        states[unit_id] = state

    non_lockable = {
        unit_id for unit_id, state in states.items() if state["reason_codes"]
    }
    changed = True
    while changed:
        changed = False
        for unit_id, unit in unit_by_id.items():
            if unit_id in non_lockable:
                continue
            blocking_units = sorted(
                {
                    unit_by_document[path]
                    for path in unit.get("bound_input_paths", [])
                    if path in unit_by_document
                    and unit_by_document[path] != unit_id
                    and unit_by_document[path] in non_lockable
                }
            )
            if not blocking_units:
                continue
            states[unit_id]["candidate_status"] = None
            states[unit_id]["reason_codes"] = ["waiting_on_dependency"]
            states[unit_id]["blocked_by_unit_ids"] = blocking_units
            non_lockable.add(unit_id)
            changed = True

    for unit_id in unit_by_id:
        state = states[unit_id]
        if state["reason_codes"]:
            skipped_entry = {
                "audit_unit_id": unit_id,
                "reason_codes": state["reason_codes"],
            }
            for field in ("open_finding_fingerprints", "blocked_by_unit_ids", "error"):
                if state.get(field):
                    skipped_entry[field] = state[field]
            skipped.append(skipped_entry)
            continue
        source_reports = state["source_reports"]
        signature = state["signature"]
        state_core = {
            **signature,
            "source_reports": source_reports,
        }
        if state["candidate_status"] == "clean":
            locks.append(
                {
                    "lock_id": _sha256_json(state_core),
                    "status": "clean",
                    **state_core,
                }
            )
        else:
            settled_core = {
                **state_core,
                "terminal_dispositions": state["terminal_dispositions"],
            }
            settled_units.append(
                {
                    "lock_id": _sha256_json(settled_core),
                    "status": "settled",
                    **settled_core,
                }
            )

    target_dir = ledger.repo_path(output_dir)
    locks_path = contained_child(target_dir, contract.CLEAN_UNIT_LOCKS_NAME, label="clean unit locks output")
    payload = {
        "schema_version": contract.CLEAN_UNIT_LOCKS_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_role": "clean_unit_lock_support",
        "audit_authority": False,
        "created_at": ledger.utc_now(),
        "scope_id": matrix_payload["scope_id"],
        "run_id": matrix_payload["run_id"],
        "audit_snapshot_id": matrix_payload["audit_snapshot_id"],
        "matrix_path": ledger.display_path(ledger.repo_path(matrix_path)),
        "reports_dir": ledger.display_path(ledger.repo_path(reports_dir)),
        "ledger_path": ledger.display_path(ledger.repo_path(ledger_path)),
        "recheck_plan_path": (
            ledger.display_path(ledger.repo_path(recheck_plan_path)) if recheck_plan_path else None
        ),
        "lock_count": len(locks),
        "settled_count": len(settled_units),
        "skipped_count": len(skipped),
        "locks": locks,
        "settled_units": settled_units,
        "skipped": skipped,
    }
    ledger.write_json(locks_path, payload)
    return {
        "clean_unit_locks_path": ledger.display_path(locks_path),
        "clean_unit_lock_count": len(locks),
        "settled_unit_count": len(settled_units),
        "clean_unit_lock_skipped_count": len(skipped),
    }


def _load_clean_unit_locks(path: Path | str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = _load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, Mapping):
        return None, "clean-unit-locks input must be a JSON object"
    if payload.get("schema_version") != contract.CLEAN_UNIT_LOCKS_SCHEMA_VERSION:
        return None, f"unsupported clean-unit-locks schema_version: {payload.get('schema_version')}"
    if payload.get("workflow_id") != contract.WORKFLOW_ID:
        return None, f"unexpected clean-unit-locks workflow_id: {payload.get('workflow_id')}"
    if payload.get("audit_authority") is not False:
        return None, "clean-unit-locks artifact must be support-only"
    return dict(payload), None


def _state_index(
    payload: Mapping[str, Any],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for field, expected_status in (("locks", "clean"), ("settled_units", "settled")):
        raw_states = payload.get(field, [])
        if not isinstance(raw_states, list):
            raise ValueError(f"clean-unit-locks {field} must be a list")
        for state in raw_states:
            if not isinstance(state, Mapping):
                raise ValueError("clean-unit state entries must be objects")
            if str(state.get("status", "")) != expected_status:
                raise ValueError(f"clean-unit {field} entry has the wrong status")
            key = (
                str(state.get("scope_id", "")).strip(),
                str(state.get("audit_unit_id", "")).strip(),
            )
            if not all(key):
                raise ValueError("clean-unit state entry is missing scope_id or audit_unit_id")
            if key in indexed:
                raise ValueError(f"duplicate clean-unit state: {' '.join(key)}")
            indexed[key] = dict(state)

    skipped_index: dict[tuple[str, str], dict[str, Any]] = {}
    raw_skipped = payload.get("skipped", [])
    if not isinstance(raw_skipped, list):
        raise ValueError("clean-unit-locks skipped must be a list")
    artifact_scope = str(payload.get("scope_id", "")).strip()
    for skipped in raw_skipped:
        if not isinstance(skipped, Mapping):
            raise ValueError("clean-unit skipped entries must be objects")
        unit_id = str(skipped.get("audit_unit_id", "")).strip()
        if unit_id:
            skipped_index[(artifact_scope, unit_id)] = dict(skipped)
    return indexed, skipped_index


def _compare_signature(current: Mapping[str, Any], lock: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in (
        "project",
        "scope_id",
        "audit_unit_id",
        "unit_definition_hash",
        "document_path_hashes",
        "bound_input_path_hashes",
        "applicable_rule_rows",
        "rule_source_snapshots",
        "source_ref",
        "resolved_source_ref",
    ):
        if current.get(field) != lock.get(field):
            reasons.append(f"{field}_changed")
    return reasons


def _validate_prior_source_reports(
    lock: Mapping[str, Any],
    unit_rows: Sequence[Mapping[str, Any]],
    *,
    allowed_statuses: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    lock_core = {key: value for key, value in lock.items() if key not in {"lock_id", "status"}}
    if str(lock.get("lock_id", "")) != _sha256_json(lock_core):
        raise ValueError("clean-unit state lock_id is invalid")
    raw_reports = lock.get("source_reports")
    if not isinstance(raw_reports, list):
        raise ValueError("clean-unit lock source_reports must be a list")
    expected = {_row_key(row) for row in unit_rows}
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    from tokimon.workflow.document_audit import reports as report_module

    for source in raw_reports:
        if not isinstance(source, Mapping):
            raise ValueError("clean-unit lock source report entries must be objects")
        key = _row_key(source)
        if key in indexed:
            raise ValueError("clean-unit lock contains duplicate source reports")
        report_path = str(source.get("report_path", "")).strip()
        if not report_path:
            raise ValueError("clean-unit lock source report is missing report_path")
        path = ledger.repo_path(report_path)
        if not path.is_file():
            raise ValueError("clean-unit lock source report path is missing")
        if str(source.get("report_hash", "")) != _file_hash(path):
            raise ValueError("clean-unit lock source report hash changed")
        payload = _load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError("clean-unit lock source report must be a JSON object")
        report = report_module.validate_report(payload, report_path=path)
        if _row_key(report) != key:
            raise ValueError("clean-unit lock source report identity changed")
        if report.get("schema_version") != contract.AUDIT_REPORT_SCHEMA_VERSION:
            raise ValueError("clean-unit state source report schema is obsolete")
        if str(report.get("status", "")) not in set(allowed_statuses):
            raise ValueError("clean-unit state source report status is not reusable")
        if str(source.get("status", "")) != str(report["status"]):
            raise ValueError("clean-unit lock source report status changed")
        for field in ("run_id", "scope_id", "audit_snapshot_id"):
            if str(source.get(field, "")) != str(report.get(field, "")):
                raise ValueError(f"clean-unit lock source report {field} changed")
        indexed[key] = {
            "evaluation": report_module.evaluation_from_report(report),
            "report": report,
            "report_path": ledger.display_path(path),
            "report_hash": str(source["report_hash"]),
        }
    if set(indexed) != expected:
        raise ValueError("clean-unit lock source report coverage changed")
    return indexed


def _validate_terminal_state(
    *,
    state: Mapping[str, Any],
    source_reports: Mapping[tuple[str, str], Mapping[str, Any]],
    scope_id: str,
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    current = _terminal_dispositions(
        reports=[source["report"] for source in source_reports.values()],
        scope_id=scope_id,
        entries=entries,
    )
    if current is None or current != state.get("terminal_dispositions"):
        raise ValueError("terminal unit dispositions changed")


def _activation_reasons(
    units: Sequence[Mapping[str, Any]],
    activated_unit_ids: Sequence[str],
) -> dict[str, str]:
    units_by_id = {str(unit["audit_unit_id"]): unit for unit in units}
    requested = {str(unit_id).strip() for unit_id in activated_unit_ids if str(unit_id).strip()}
    unknown = requested - set(units_by_id)
    if unknown:
        raise ValueError(f"unknown audit_unit_id values: {', '.join(sorted(unknown))}")
    return {unit_id: "explicitly_activated" for unit_id in requested}


def _active_dependency_blockers(
    *,
    units_by_id: Mapping[str, Mapping[str, Any]],
    unit_by_document: Mapping[str, str],
    active_unit_ids: set[str],
) -> dict[str, set[str]]:
    blockers = {
        unit_id: {
            unit_by_document[path]
            for path in unit.get("bound_input_paths", [])
            if path in unit_by_document
            and unit_by_document[path] != unit_id
            and unit_by_document[path] in active_unit_ids
        }
        for unit_id, unit in units_by_id.items()
        if unit_id in active_unit_ids
    }
    reachable: dict[str, set[str]] = {}
    for start in active_unit_ids:
        seen: set[str] = set()
        pending = list(blockers.get(start, set()))
        while pending:
            unit_id = pending.pop()
            if unit_id in seen:
                continue
            seen.add(unit_id)
            pending.extend(blockers.get(unit_id, set()) - seen)
        reachable[start] = seen

    component_by_unit = {
        unit_id: frozenset(
            candidate
            for candidate in active_unit_ids
            if candidate == unit_id
            or (
                candidate in reachable[unit_id]
                and unit_id in reachable[candidate]
            )
        )
        for unit_id in active_unit_ids
    }
    return {
        unit_id: {
            blocker
            for member in component_by_unit[unit_id]
            for blocker in blockers[member]
            if blocker not in component_by_unit[unit_id]
        }
        for unit_id in blockers
    }


def build_recheck_plan_payload(
    *,
    matrix_path: Path | str,
    prior_clean_unit_locks_path: Path | str,
    ledger_path: Path | str,
    activated_unit_ids: Sequence[str] = (),
) -> dict[str, Any]:
    matrix_payload = _load_matrix(matrix_path)
    rows_by_unit = _rows_by_unit(matrix_payload)
    entries = _ledger_entries(ledger_path)
    prior_locks, lock_error = _load_clean_unit_locks(prior_clean_unit_locks_path)
    lock_validation_available = prior_locks is not None
    state_index: dict[tuple[str, str], dict[str, Any]] = {}
    if prior_locks is not None:
        try:
            state_index, _ = _state_index(prior_locks)
        except ValueError as exc:
            lock_error = str(exc)
            lock_validation_available = False
    requested_activation_ids = sorted(
        {str(unit_id).strip() for unit_id in activated_unit_ids if str(unit_id).strip()}
    )
    activation_reasons = _activation_reasons(matrix_payload["audit_units"], requested_activation_ids)

    units_by_id = {
        str(unit["audit_unit_id"]): unit for unit in matrix_payload["audit_units"]
    }
    unit_by_document = {
        str(path): unit_id
        for unit_id, unit in units_by_id.items()
        for path in unit["document_paths"]
    }
    unit_plans_by_id: dict[str, dict[str, Any]] = {}
    source_reports_by_unit: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    scope_id = str(matrix_payload["scope_id"])
    for unit in matrix_payload["audit_units"]:
        unit_id = str(unit["audit_unit_id"])
        unit_rows = rows_by_unit[unit_id]
        state = state_index.get((scope_id, unit_id))
        reasons: list[str] = []
        activation_reason = activation_reasons.get(unit_id)
        if activation_reason:
            reasons.append(activation_reason)
        source_reports: dict[tuple[str, str], dict[str, Any]] = {}
        open_fingerprints = _open_findings(
            entries=entries,
            scope_id=scope_id,
            document_paths=unit["document_paths"],
        )
        if open_fingerprints:
            reasons.append("unit_has_open_findings")
        if not lock_validation_available:
            reasons.append("clean_unit_lock_validation_unavailable")
        elif state is None:
            reasons.append("no_prior_clean_unit_lock")
        else:
            try:
                signature = _unit_signature(
                    matrix_payload=matrix_payload,
                    unit=unit,
                    unit_rows=unit_rows,
                )
                reasons.extend(_compare_signature(signature, state))
                state_status = str(state.get("status", ""))
                allowed_statuses = (
                    tuple(CLEAN_STATUSES)
                    if state_status == "clean"
                    else tuple(contract.AUDIT_REPORT_STATUSES)
                )
                source_reports = _validate_prior_source_reports(
                    state,
                    unit_rows,
                    allowed_statuses=allowed_statuses,
                )
                if state_status == "settled":
                    _validate_terminal_state(
                        state=state,
                        source_reports=source_reports,
                        scope_id=scope_id,
                        entries=entries,
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                reasons.append(str(exc).replace(" ", "_"))
        if state is not None and not reasons:
            action = LOCK_ACTION if state.get("status") == "clean" else SETTLED_ACTION
        else:
            action = FRESH_ACTION
        unit_plan: dict[str, Any] = {
            "audit_unit_id": unit_id,
            "action": action,
            "reason_codes": reasons or ["inputs_unchanged"],
            "row_count": len(unit_rows),
        }
        if open_fingerprints:
            unit_plan["open_finding_fingerprints"] = open_fingerprints
        if state is not None:
            unit_plan["lock_id"] = state.get("lock_id")
        unit_plans_by_id[unit_id] = unit_plan
        source_reports_by_unit[unit_id] = source_reports

    unstable_unit_ids = {
        unit_id
        for unit_id, unit_plan in unit_plans_by_id.items()
        if unit_plan["action"] == FRESH_ACTION
    }
    dependency_blockers = (
        _active_dependency_blockers(
            units_by_id=units_by_id,
            unit_by_document=unit_by_document,
            active_unit_ids=unstable_unit_ids,
        )
        if lock_validation_available
        else {}
    )
    for unit_id, blocking_units in dependency_blockers.items():
        unit_plan = unit_plans_by_id[unit_id]
        if not blocking_units:
            continue
        unit_plan["action"] = WAIT_ACTION
        unit_plan["reason_codes"] = [*unit_plan["reason_codes"], "waiting_on_dependency"]
        unit_plan["blocked_by_unit_ids"] = sorted(blocking_units)

    unit_plans = [
        unit_plans_by_id[str(unit["audit_unit_id"])] for unit in matrix_payload["audit_units"]
    ]
    row_plans: list[dict[str, Any]] = []
    for unit in matrix_payload["audit_units"]:
        unit_id = str(unit["audit_unit_id"])
        unit_plan = unit_plans_by_id[unit_id]
        action = str(unit_plan["action"])
        state = state_index.get((scope_id, unit_id))
        source_reports = source_reports_by_unit[unit_id]
        for row in rows_by_unit[unit_id]:
            key = _row_key(row)
            row_plan: dict[str, Any] = {
                "audit_unit_id": unit_id,
                "file_path": key[0],
                "rule_id": key[1],
                "action": action,
                "reason_codes": unit_plan["reason_codes"],
            }
            source = source_reports.get(key)
            if action in {LOCK_ACTION, SETTLED_ACTION} and source is not None and state is not None:
                evaluation_field = (
                    "locked_evaluation" if action == LOCK_ACTION else "settled_evaluation"
                )
                row_plan.update(
                    {
                        "lock_id": state.get("lock_id"),
                        "prior_report_path": source["report_path"],
                        "prior_report_hash": source["report_hash"],
                        evaluation_field: {
                            **source["evaluation"],
                            "evaluation_source": (
                                "validated_prior_clean_unit_lock"
                                if action == LOCK_ACTION
                                else "validated_terminal_disposition"
                            ),
                            "reused_in_run_id": matrix_payload["run_id"],
                            "lock_id": state.get("lock_id"),
                        },
                    }
                )
            row_plans.append(row_plan)

    unit_counts = Counter(unit["action"] for unit in unit_plans)
    row_counts = Counter(row["action"] for row in row_plans)
    return {
        "schema_version": contract.RECHECK_PLAN_SCHEMA_VERSION,
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_role": "recheck_plan_support",
        "audit_authority": False,
        "created_at": ledger.utc_now(),
        "scope_id": matrix_payload["scope_id"],
        "run_id": matrix_payload["run_id"],
        "audit_snapshot_id": matrix_payload["audit_snapshot_id"],
        "matrix_path": ledger.display_path(ledger.repo_path(matrix_path)),
        "prior_clean_unit_locks_path": ledger.display_path(
            ledger.repo_path(prior_clean_unit_locks_path)
        ),
        "ledger_path": ledger.display_path(ledger.repo_path(ledger_path)),
        "activated_unit_ids": requested_activation_ids,
        "lock_validation_available": lock_validation_available,
        "lock_validation_error": lock_error,
        "complete_fresh_reports_required": not lock_validation_available,
        "audit_unit_count": len(unit_plans),
        "locked_unit_count": unit_counts[LOCK_ACTION],
        "settled_unit_count": unit_counts[SETTLED_ACTION],
        "waiting_unit_count": unit_counts[WAIT_ACTION],
        "active_unit_count": unit_counts[FRESH_ACTION],
        "matrix_count": len(row_plans),
        "locked_row_count": row_counts[LOCK_ACTION],
        "settled_row_count": row_counts[SETTLED_ACTION],
        "waiting_row_count": row_counts[WAIT_ACTION],
        "fresh_required_row_count": row_counts[FRESH_ACTION],
        "units": unit_plans,
        "rows": row_plans,
    }


def write_recheck_plan(
    *,
    matrix_path: Path | str,
    prior_clean_unit_locks_path: Path | str,
    output_dir: Path | str,
    ledger_path: Path | str,
    activated_unit_ids: Sequence[str] = (),
) -> dict[str, Any]:
    output_path = contained_child(ledger.repo_path(output_dir), contract.RECHECK_PLAN_NAME, label="recheck plan output")
    payload = build_recheck_plan_payload(
        matrix_path=matrix_path,
        prior_clean_unit_locks_path=prior_clean_unit_locks_path,
        ledger_path=ledger_path,
        activated_unit_ids=activated_unit_ids,
    )
    artifact_paths = {"recheck_plan_json": ledger.display_path(output_path)}
    payload["artifact_paths"] = artifact_paths
    ledger.write_json(output_path, payload)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "artifact_paths": artifact_paths,
        "summary": {
            "schema_version": contract.RECHECK_PLAN_SCHEMA_VERSION,
            "workflow_id": contract.WORKFLOW_ID,
            "audit_authority": False,
            "scope_id": payload["scope_id"],
            "run_id": payload["run_id"],
            "lock_validation_available": payload["lock_validation_available"],
            "lock_validation_error": payload["lock_validation_error"],
            "audit_unit_count": payload["audit_unit_count"],
            "locked_unit_count": payload["locked_unit_count"],
            "settled_unit_count": payload["settled_unit_count"],
            "waiting_unit_count": payload["waiting_unit_count"],
            "active_unit_count": payload["active_unit_count"],
            "locked_row_count": payload["locked_row_count"],
            "settled_row_count": payload["settled_row_count"],
            "waiting_row_count": payload["waiting_row_count"],
            "fresh_required_row_count": payload["fresh_required_row_count"],
        },
    }


def revalidated_plan_payload(
    *,
    recheck_plan_path: Path | str,
    matrix_path: Path | str,
) -> dict[str, Any]:
    plan = _load_json(recheck_plan_path)
    if not isinstance(plan, Mapping):
        raise ValueError("recheck-plan input must be a JSON object")
    if plan.get("schema_version") != contract.RECHECK_PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported recheck-plan schema_version: {plan.get('schema_version')}")
    if plan.get("workflow_id") != contract.WORKFLOW_ID:
        raise ValueError(f"unexpected recheck-plan workflow_id: {plan.get('workflow_id')}")
    if plan.get("audit_authority") is not False:
        raise ValueError("recheck-plan artifact must be support-only")
    prior_path = str(plan.get("prior_clean_unit_locks_path", "")).strip()
    ledger_path = str(plan.get("ledger_path", "")).strip()
    if not prior_path or not ledger_path:
        raise ValueError("recheck-plan is missing lock or ledger provenance")
    activated = plan.get("activated_unit_ids", [])
    if not isinstance(activated, list):
        raise ValueError("recheck-plan activated_unit_ids must be a list")
    return build_recheck_plan_payload(
        matrix_path=matrix_path,
        prior_clean_unit_locks_path=prior_path,
        ledger_path=ledger_path,
        activated_unit_ids=[str(unit_id) for unit_id in activated],
    )


def validated_plan_rows(
    *,
    recheck_plan_path: Path | str,
    matrix_path: Path | str,
) -> dict[tuple[str, str], dict[str, Any]]:
    recomputed = revalidated_plan_payload(
        recheck_plan_path=recheck_plan_path,
        matrix_path=matrix_path,
    )
    return {
        (str(row["file_path"]), str(row["rule_id"])): dict(row)
        for row in recomputed["rows"]
    }


def validated_locked_rows(
    *,
    recheck_plan_path: Path | str,
    matrix_path: Path | str,
) -> dict[tuple[str, str], dict[str, Any]]:
    recomputed = revalidated_plan_payload(
        recheck_plan_path=recheck_plan_path,
        matrix_path=matrix_path,
    )
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in recomputed["rows"]:
        if row["action"] != LOCK_ACTION:
            continue
        if not isinstance(row.get("locked_evaluation"), Mapping):
            continue
        rows[(row["file_path"], row["rule_id"])] = dict(row)
    return rows
