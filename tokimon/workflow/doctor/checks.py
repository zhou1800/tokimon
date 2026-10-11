"""Read-only readiness reporting for a Tokimon workspace."""

from __future__ import annotations

import os
import importlib.util
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tokimon.memory.mem_palace import mem_palace as mem_palace
import tokimon.runtime as runtime_helpers
from tokimon.runtime import source_root
from tokimon.tools import registry
from tokimon.workflow.doctor import contract


WORKSPACE_ROOT = source_root()

REQUIRED_REPOSITORY_PATHS: tuple[tuple[str, str], ...] = (
    ("AGENTS.md", "file"),
    ("docs/guidelines/repository-guidelines.md", "file"),
    ("docs/guidelines/helix.md", "file"),
    ("docs/guidelines/code-guidelines.md", "file"),
    ("docs/guidelines/document-guidelines.md", "file"),
    ("docs/guidelines/c4-document-guidelines.md", "file"),
    ("docs/guidelines/language-and-communication-guidelines.md", "file"),
    ("docs/c4", "dir"),
    ("tokimon/memory", "dir"),
    ("tokimon/tools", "dir"),
    ("tokimon/workflow", "dir"),
    ("tests", "dir"),
)

MEMORY_REQUIRED_TABLES = frozenset(("schema_meta", "memories", "entities", "relationships"))
TOOL_STATUS_VALUES = frozenset(("available", "unavailable"))
WORKFLOW_READINESS_ORDER = (
    "doctor",
    "memory-maintenance",
    "document-audit",
    "code-audit",
    "pair",
    "learning",
)


@dataclass(frozen=True)
class DoctorCheck:
    id: str
    status: str
    summary: str
    details: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "summary": self.summary,
            "details": list(self.details),
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class ReadinessItem:
    id: str
    layer: str
    kind: str
    status: str
    summary: str
    reasons: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "kind": self.kind,
            "status": self.status,
            "summary": self.summary,
            "reasons": list(self.reasons),
            "data": dict(self.data),
        }


def _absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.absolute()
    return (Path.cwd() / expanded).absolute()


def _path_status(path: Path, expected_kind: str) -> bool:
    if expected_kind == "file":
        return path.is_file()
    if expected_kind == "dir":
        return path.is_dir()
    return path.exists()


def check_required_paths(workspace_root: Path) -> DoctorCheck:
    missing_or_wrong: list[str] = []
    for relative_path, expected_kind in REQUIRED_REPOSITORY_PATHS:
        candidate = workspace_root / relative_path
        if not _path_status(candidate, expected_kind):
            missing_or_wrong.append(f"{relative_path} ({expected_kind})")

    if missing_or_wrong:
        return DoctorCheck(
            id="repository.required_paths",
            status="fail",
            summary="Required Tokimon repository paths are missing or have the wrong type.",
            details=tuple(missing_or_wrong),
            data={"workspace_root": str(workspace_root)},
        )

    return DoctorCheck(
        id="repository.required_paths",
        status="pass",
        summary="Required Tokimon repository paths are present.",
        data={"workspace_root": str(workspace_root), "checked_count": len(REQUIRED_REPOSITORY_PATHS)},
    )


def check_runtime_policy(workspace_root: Path) -> tuple[DoctorCheck, DoctorCheck]:
    gitignore_path = workspace_root / ".gitignore"
    if not gitignore_path.is_file():
        ignore_check = DoctorCheck(
            id="runtime.gitignore",
            status="fail",
            summary=".gitignore is missing, so local runtime state may be committed.",
        )
    else:
        lines = {
            line.strip()
            for line in gitignore_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        if ".tokimon/" in lines or ".tokimon" in lines:
            ignore_check = DoctorCheck(
                id="runtime.gitignore",
                status="pass",
                summary=".tokimon/ is ignored as Tokimon's local runtime root.",
                data={"gitignore": str(gitignore_path.relative_to(workspace_root))},
            )
        else:
            ignore_check = DoctorCheck(
                id="runtime.gitignore",
                status="fail",
                summary=".gitignore does not ignore Tokimon's .tokimon/ local runtime root.",
                details=("Add .tokimon/ to .gitignore before writing local runtime artifacts.",),
            )

    runtime_root = runtime_helpers.runtime_root(workspace_root)
    if runtime_root.is_dir():
        root_check = DoctorCheck(
            id="runtime.root",
            status="pass",
            summary=".tokimon/ local runtime root exists.",
            data={"path": str(runtime_root)},
        )
    elif runtime_root.exists():
        root_check = DoctorCheck(
            id="runtime.root",
            status="fail",
            summary=".tokimon exists but is not a directory.",
            data={"path": str(runtime_root)},
        )
    else:
        root_check = DoctorCheck(
            id="runtime.root",
            status="warn",
            summary=".tokimon/ local runtime root does not exist yet.",
            details=("Run a Tokimon memory, tool, or workflow command that owns local state when runtime state is needed.",),
            data={"path": str(runtime_root)},
        )

    return ignore_check, root_check


def check_python_environment(workspace_root: Path, *, python_executable: str | None = None) -> DoctorCheck:
    actual_python = _absolute_path(Path(python_executable or sys.executable))
    if workspace_root.resolve() == runtime_helpers.distribution_workspace_root():
        environment = _absolute_path(Path(sys.prefix))
        expected = runtime_helpers.python_executable(workspace_root)
        valid = expected.is_file() and actual_python.is_relative_to(environment)
        return DoctorCheck(
            id="python.venv",
            status="pass" if valid else "fail",
            summary="Doctor is running from the wheel's owning Python environment." if valid else "Doctor is not running from the wheel's owning Python environment.",
            data={"python_executable": str(actual_python), "expected_python": str(expected)},
        )
    venv_root = workspace_root / ".venv"
    expected_python = venv_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    primary_root = runtime_helpers.primary_worktree_root(workspace_root)
    if primary_root is not None:
        primary_venv = primary_root / ".venv"
        primary_python = primary_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if primary_python.exists() and actual_python.is_relative_to(_absolute_path(primary_venv)):
            return DoctorCheck(
                id="python.venv",
                status="pass",
                summary="Doctor is running from the primary worktree .venv Python.",
                data={"python_executable": str(actual_python), "expected_python": str(primary_python)},
            )

    if not expected_python.exists():
        return DoctorCheck(
            id="python.venv",
            status="fail",
            summary="The local .venv Python executable is missing.",
            details=(f"Expected: {expected_python}",),
            data={"python_executable": str(actual_python)},
        )

    if actual_python.is_relative_to(_absolute_path(venv_root)):
        return DoctorCheck(
            id="python.venv",
            status="pass",
            summary="Doctor is running from the local .venv Python.",
            data={"python_executable": str(actual_python), "expected_python": str(expected_python)},
        )

    return DoctorCheck(
        id="python.venv",
        status="warn",
        summary="Doctor is not running from the local .venv Python.",
        details=(f"Use {contract.OPERATOR_ENTRYPOINT} for Tokimon workflow commands.",),
        data={"python_executable": str(actual_python), "expected_python": str(expected_python)},
    )


def _connect_sqlite_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def check_memory_database(workspace_root: Path) -> DoctorCheck:
    db_path = runtime_helpers.runtime_root(workspace_root) / "memory" / "memory.db"
    if not db_path.exists():
        return DoctorCheck(
            id="memory.mem_palace",
            status="warn",
            summary="Mem Palace database does not exist yet.",
            details=("Expected local database at .tokimon/memory/memory.db.",),
            data={"database": str(db_path)},
        )
    if not db_path.is_file():
        return DoctorCheck(
            id="memory.mem_palace",
            status="fail",
            summary="Mem Palace database path exists but is not a file.",
            data={"database": str(db_path)},
        )

    try:
        conn = _connect_sqlite_readonly(db_path)
    except sqlite3.Error as exc:
        return DoctorCheck(
            id="memory.mem_palace",
            status="fail",
            summary="Mem Palace database could not be opened read-only.",
            details=(str(exc),),
            data={"database": str(db_path)},
        )

    try:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        missing_tables = sorted(MEMORY_REQUIRED_TABLES - tables)
        if missing_tables:
            return DoctorCheck(
                id="memory.mem_palace",
                status="fail",
                summary="Mem Palace database is missing required schema tables.",
                details=tuple(missing_tables),
                data={"database": str(db_path)},
            )

        schema_row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        schema_version = str(schema_row["value"]) if schema_row else ""
        memory_count = int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
        pinned_count = int(conn.execute("SELECT COUNT(*) FROM memories WHERE pinned = 1").fetchone()[0])
        entity_count = int(conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
        relationship_count = int(conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0])
        current_relationship_count = int(
            conn.execute("SELECT COUNT(*) FROM relationships WHERE valid_to IS NULL").fetchone()[0]
        )
        kind_counts = [
            {"kind": row["kind"], "count": int(row["count"])}
            for row in conn.execute(
                "SELECT kind, COUNT(*) AS count FROM memories GROUP BY kind ORDER BY count DESC, kind"
            ).fetchall()
        ]
        scopes = [
            {"scope": row["scope"], "count": int(row["count"])}
            for row in conn.execute(
                "SELECT scope, COUNT(*) AS count FROM memories GROUP BY scope ORDER BY count DESC, scope LIMIT 10"
            ).fetchall()
        ]
        rooms = [
            {"room": row["room"], "count": int(row["count"])}
            for row in conn.execute(
                "SELECT room, COUNT(*) AS count FROM memories GROUP BY room ORDER BY count DESC, room LIMIT 10"
            ).fetchall()
        ]
    except sqlite3.Error as exc:
        return DoctorCheck(
            id="memory.mem_palace",
            status="fail",
            summary="Mem Palace database schema could not be inspected.",
            details=(str(exc),),
            data={"database": str(db_path)},
        )
    finally:
        conn.close()

    expected_version = str(mem_palace.SCHEMA_VERSION)
    wal_path = db_path.with_name(f"{db_path.name}-wal")
    data = {
        "database": str(db_path),
        "database_size_bytes": db_path.stat().st_size,
        "wal_size_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
        "schema_version": schema_version,
        "expected_schema_version": expected_version,
        "memories": memory_count,
        "pinned_memories": pinned_count,
        "unpinned_memories": memory_count - pinned_count,
        "entities": entity_count,
        "relationships": relationship_count,
        "current_relationships": current_relationship_count,
        "expired_relationships": relationship_count - current_relationship_count,
        "kind_counts": kind_counts,
        "scopes": scopes,
        "rooms": rooms,
    }
    if schema_version != expected_version:
        return DoctorCheck(
            id="memory.mem_palace",
            status="warn",
            summary="Mem Palace database schema version differs from the expected schema version.",
            details=(f"Found {schema_version or 'unknown'}, expected {expected_version}.",),
            data=data,
        )

    return DoctorCheck(
        id="memory.mem_palace",
        status="pass",
        summary="Mem Palace database is present and readable.",
        data=data,
    )


def check_tool_registry(
    workspace_root: Path,
    *,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    host_id: str | None = None,
) -> DoctorCheck:
    try:
        payload = registry.capability_map(
            workspace_root=workspace_root,
            env=env,
            platform_name=platform_name,
            host_id=host_id,
        )
        capabilities = payload.get("capabilities", [])
        if not isinstance(capabilities, list) or not capabilities:
            return DoctorCheck(
                id="tools.registry",
                status="fail",
                summary="Tools registry did not return any capability metadata.",
                data={"host": payload.get("host", {})},
            )

        malformed: list[str] = []
        statuses: dict[str, str] = {}
        status_counts = {status: 0 for status in TOOL_STATUS_VALUES}
        family_status_counts: dict[str, dict[str, int]] = {}
        for item in capabilities:
            if not isinstance(item, Mapping):
                malformed.append(str(item))
                continue
            capability_id = str(item.get("id") or "")
            status = str(item.get("status") or "")
            family = str(item.get("family") or "unknown")
            if not capability_id or status not in TOOL_STATUS_VALUES:
                malformed.append(capability_id or "<missing id>")
            else:
                statuses[capability_id] = status
                status_counts[status] += 1
                family_counts = family_status_counts.setdefault(family, {value: 0 for value in TOOL_STATUS_VALUES})
                family_counts[status] += 1

        if malformed:
            return DoctorCheck(
                id="tools.registry",
                status="fail",
                summary="Tools registry returned malformed capability metadata.",
                details=tuple(malformed),
                data={"host": payload.get("host", {})},
            )

        return DoctorCheck(
            id="tools.registry",
            status="pass",
            summary="Tools registry returned host and capability metadata.",
            data={
                "host": payload.get("host", {}),
                "capabilities": capabilities,
                "capability_statuses": statuses,
                "status_counts": status_counts,
                "family_status_counts": family_status_counts,
            },
        )
    except Exception as exc:  # noqa: BLE001 - doctor reports boundary failures instead of crashing.
        return DoctorCheck(
            id="tools.registry",
            status="fail",
            summary="Tools registry metadata could not be loaded.",
            details=(f"{type(exc).__name__}: {exc}",),
        )


def summarize_checks(checks: tuple[DoctorCheck, ...]) -> dict[str, int]:
    return {status: sum(1 for check in checks if check.status == status) for status in contract.STATUS_VALUES}


def summarize_readiness(items: tuple[ReadinessItem, ...]) -> dict[str, int]:
    return {status: sum(1 for item in items if item.status == status) for status in contract.READINESS_STATUS_VALUES}


def _check_report_dict(check: DoctorCheck) -> dict[str, Any]:
    result = check.to_dict()
    if check.id == "tools.registry":
        result["data"].pop("capabilities", None)
    return result


def overall_status(checks: tuple[DoctorCheck, ...], readiness_items: tuple[ReadinessItem, ...]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks) or any(item.status != "ready" for item in readiness_items):
        return "warn"
    return "pass"


def _capability_by_id(tool_check: DoctorCheck) -> dict[str, Mapping[str, Any]]:
    capabilities = tool_check.data.get("capabilities", [])
    if not isinstance(capabilities, list):
        return {}
    return {
        str(item.get("id")): item
        for item in capabilities
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    }


def _capability_status(capability_by_id: Mapping[str, Mapping[str, Any]], capability_id: str) -> str:
    capability = capability_by_id.get(capability_id, {})
    return str(capability.get("status") or "unavailable")


def _tool_reason(capability: Mapping[str, Any]) -> tuple[str, ...]:
    status = str(capability.get("status") or "unknown")
    if status == "available":
        return ()
    reasons = [f"registry status is {status}"]
    health_check = str(capability.get("health_check") or "")
    if health_check:
        reasons.append(f"health check: {health_check}")
    requirements = capability.get("host_requirements", [])
    if isinstance(requirements, list) and requirements:
        reasons.append("host requirements: " + "; ".join(str(item) for item in requirements))
    return tuple(reasons)


def build_memory_readiness(memory_check: DoctorCheck) -> ReadinessItem:
    data = dict(memory_check.data)
    if memory_check.status == "pass":
        memory_count = data.get("memories", 0)
        pinned_count = data.get("pinned_memories", 0)
        return ReadinessItem(
            id="memory.mem_palace",
            layer="memory",
            kind="capability",
            status="ready",
            summary=f"Mem Palace has {memory_count} memories ({pinned_count} pinned).",
            data=data,
        )
    if memory_check.status == "warn":
        return ReadinessItem(
            id="memory.mem_palace",
            layer="memory",
            kind="capability",
            status="partial",
            summary=memory_check.summary,
            reasons=memory_check.details,
            data=data,
        )
    return ReadinessItem(
        id="memory.mem_palace",
        layer="memory",
        kind="capability",
        status="blocked",
        summary=memory_check.summary,
        reasons=memory_check.details,
        data=data,
    )


def build_tool_readiness(tool_check: DoctorCheck) -> tuple[ReadinessItem, ...]:
    if tool_check.status != "pass":
        return (
            ReadinessItem(
                id="tools.registry",
                layer="tools",
                kind="registry",
                status="blocked",
                summary=tool_check.summary,
                reasons=tool_check.details,
                data=dict(tool_check.data),
            ),
        )

    items: list[ReadinessItem] = []
    for capability in tool_check.data.get("capabilities", []):
        if not isinstance(capability, Mapping):
            continue
        capability_id = str(capability.get("id") or "")
        status = str(capability.get("status") or "")
        if status == "available":
            readiness_status = "ready"
            summary = f"{capability_id} is available."
        else:
            readiness_status = "blocked"
            summary = f"{capability_id} is unavailable on this host."
        items.append(
            ReadinessItem(
                id=capability_id,
                layer="tools",
                kind=str(capability.get("family") or "tool"),
                status=readiness_status,
                summary=summary,
                reasons=_tool_reason(capability),
                data={
                    "name": capability.get("name"),
                    "registry_status": status,
                    "implementation": capability.get("implementation"),
                    "health_check": capability.get("health_check"),
                },
            )
        )
    return tuple(items)


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _mapping_tuple(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _workflow_readiness_sort_key(declaration: Mapping[str, Any]) -> tuple[int, str]:
    workflow_id = str(declaration.get("id") or "")
    try:
        index = WORKFLOW_READINESS_ORDER.index(workflow_id)
    except ValueError:
        index = len(WORKFLOW_READINESS_ORDER)
    return index, workflow_id


def _load_contract_module(contract_path: Path) -> Any:
    module_name = f"_tokimon_workflow_contract_{abs(hash(str(contract_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec for {contract_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_workflow_readiness_declarations(*, workspace_root: Path = WORKSPACE_ROOT) -> tuple[Mapping[str, Any], ...]:
    declarations: list[Mapping[str, Any]] = []
    workflow_root = workspace_root / "tokimon" / "workflow"
    for contract_path in sorted(workflow_root.glob("*/contract.py")):
        package_name = contract_path.parent.name
        try:
            module = _load_contract_module(contract_path)
        except Exception as exc:  # noqa: BLE001 - doctor reports declaration load failures as readiness items.
            declarations.append(
                {
                    "id": package_name,
                    "workflow_status": "unknown",
                    "entrypoint": None,
                    "modes": (),
                    "_load_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        declaration = getattr(module, "WORKFLOW_READINESS", None)
        if isinstance(declaration, Mapping):
            declarations.append(declaration)
    return tuple(sorted(declarations, key=_workflow_readiness_sort_key))


def _missing_capabilities(capability_by_id: Mapping[str, Mapping[str, Any]], capability_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(capability_id for capability_id in capability_ids if _capability_status(capability_by_id, capability_id) != "available")


def _any_available(capability_by_id: Mapping[str, Mapping[str, Any]], capability_ids: tuple[str, ...]) -> str | None:
    for capability_id in capability_ids:
        if _capability_status(capability_by_id, capability_id) == "available":
            return capability_id
    return None


def evaluate_workflow_mode(
    mode: Mapping[str, Any],
    *,
    capability_by_id: Mapping[str, Mapping[str, Any]],
    memory_item: ReadinessItem,
) -> dict[str, Any]:
    mode_id = str(mode.get("id") or "default")
    role = str(mode.get("role") or "primary")
    summary = str(mode.get("summary") or mode_id)
    status = "ready"
    reasons: list[str] = []
    selected_tool_capabilities: dict[str, str | None] = {}

    required_tool_capabilities = _string_tuple(mode.get("required_tool_capabilities"))
    missing_required = _missing_capabilities(capability_by_id, required_tool_capabilities)
    if missing_required:
        status = "blocked"
        reasons.extend(f"missing required tool capability: {capability_id}" for capability_id in missing_required)

    any_tool_capability_groups: list[dict[str, object]] = []
    for group in _mapping_tuple(mode.get("any_tool_capability_groups")):
        group_id = str(group.get("id") or "any-tool-capability")
        capability_ids = _string_tuple(group.get("capability_ids"))
        selected_capability = _any_available(capability_by_id, capability_ids)
        selected_tool_capabilities[group_id] = selected_capability
        any_tool_capability_groups.append(
            {
                "id": group_id,
                "capability_ids": list(capability_ids),
                "selected_capability": selected_capability,
            }
        )
        if selected_capability is None:
            status = "blocked"
            reasons.append(f"missing one capability from {group_id}: {', '.join(capability_ids)}")

    memory_requirement = str(mode.get("memory_requirement") or "none")
    if memory_requirement in {"readable", "existing"}:
        if memory_item.status == "blocked":
            status = "blocked"
            reasons.extend(memory_item.reasons or (memory_item.summary,))
        elif memory_item.status == "partial" and status != "blocked":
            status = "partial"
            reasons.extend(memory_item.reasons or (memory_item.summary,))

    agent_connector_capabilities = _string_tuple(mode.get("agent_connector_capabilities"))
    if agent_connector_capabilities:
        reasons.extend(
            f"requires Agent connector not inspected by local doctor: {capability_id}"
            for capability_id in agent_connector_capabilities
        )
        if status == "ready":
            status = "partial"

    return {
        "id": mode_id,
        "role": role,
        "status": status,
        "summary": summary,
        "reasons": tuple(reasons),
        "data": {
            "required_tool_capabilities": list(required_tool_capabilities),
            "any_tool_capability_groups": any_tool_capability_groups,
            "selected_tool_capabilities": selected_tool_capabilities,
            "agent_connector_capabilities": list(agent_connector_capabilities),
            "memory_requirement": memory_requirement,
        },
    }


def _select_declared_workflow_mode(modes: tuple[dict[str, Any], ...]) -> tuple[str, dict[str, Any] | None]:
    primary_modes = tuple(mode for mode in modes if mode["role"] == "primary")
    fallback_modes = tuple(mode for mode in modes if mode["role"] != "primary")

    for mode in primary_modes:
        if mode["status"] == "ready":
            return "ready", mode
    for mode in primary_modes:
        if mode["status"] == "partial":
            return "partial", mode
    for mode in fallback_modes:
        if mode["status"] == "ready":
            return "partial", mode
    for mode in fallback_modes:
        if mode["status"] == "partial":
            return "partial", mode
    return "blocked", primary_modes[0] if primary_modes else (modes[0] if modes else None)


def evaluate_workflow_readiness_declaration(
    declaration: Mapping[str, Any],
    *,
    capability_by_id: Mapping[str, Mapping[str, Any]],
    memory_item: ReadinessItem,
    foundation_failures: tuple[str, ...] = (),
) -> ReadinessItem:
    workflow_id = str(declaration.get("id") or "unknown-workflow")
    load_error = str(declaration.get("_load_error") or "")
    if load_error:
        return ReadinessItem(
            id=workflow_id,
            layer="workflow",
            kind="workflow",
            status="blocked",
            summary="Workflow readiness declaration could not be loaded.",
            reasons=(load_error,),
            data={
                "entrypoint": declaration.get("entrypoint"),
                "workflow_status": declaration.get("workflow_status"),
                "selected_mode": None,
                "modes": [],
            },
        )

    modes = tuple(
        evaluate_workflow_mode(mode, capability_by_id=capability_by_id, memory_item=memory_item)
        for mode in _mapping_tuple(declaration.get("modes"))
    )
    status, selected_mode = _select_declared_workflow_mode(modes)

    if workflow_id == contract.WORKFLOW_ID and foundation_failures:
        status = "blocked"
        summary = "Doctor found foundation failures in this checkout."
        reasons = foundation_failures
    elif selected_mode is None:
        status = "blocked"
        summary = "Workflow has no readiness modes declared."
        reasons = ()
    elif status == "blocked":
        summary = "No declared workflow mode is ready."
        reasons = tuple(
            f"{mode['id']}: {reason}"
            for mode in modes
            for reason in mode["reasons"]
        )
    elif status == "partial" and selected_mode["role"] != "primary":
        summary = str(selected_mode["summary"])
        primary_reasons = tuple(
            f"primary mode {mode['id']}: {reason}"
            for mode in modes
            if mode["role"] == "primary"
            for reason in mode["reasons"]
        )
        reasons = (*primary_reasons, *selected_mode["reasons"])
    else:
        summary = str(selected_mode["summary"])
        reasons = tuple(selected_mode["reasons"])

    return ReadinessItem(
        id=workflow_id,
        layer="workflow",
        kind="workflow",
        status=status,
        summary=summary,
        reasons=tuple(reasons),
        data={
            "entrypoint": declaration.get("entrypoint"),
            "workflow_status": declaration.get("workflow_status"),
            "selected_mode": selected_mode["id"] if selected_mode else None,
            "modes": [
                {
                    "id": mode["id"],
                    "role": mode["role"],
                    "status": mode["status"],
                    "summary": mode["summary"],
                    "reasons": list(mode["reasons"]),
                    **mode["data"],
                }
                for mode in modes
            ],
        },
    )


def build_workflow_readiness(
    *,
    checks: tuple[DoctorCheck, ...],
    memory_item: ReadinessItem,
    tool_check: DoctorCheck,
    workspace_root: Path = WORKSPACE_ROOT,
) -> tuple[ReadinessItem, ...]:
    capability_by_id = _capability_by_id(tool_check)
    foundation_failures = tuple(check.id for check in checks if check.status == "fail")
    return tuple(
        evaluate_workflow_readiness_declaration(
            declaration,
            capability_by_id=capability_by_id,
            memory_item=memory_item,
            foundation_failures=foundation_failures,
        )
        for declaration in load_workflow_readiness_declarations(workspace_root=workspace_root)
    )


def run_doctor(
    *,
    workspace_root: Path | str = WORKSPACE_ROOT,
    env: Mapping[str, str] | None = None,
    python_executable: str | None = None,
    platform_name: str | None = None,
    host_id: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    checks = (
        check_required_paths(root),
        *check_runtime_policy(root),
        check_python_environment(root, python_executable=python_executable),
        check_memory_database(root),
        check_tool_registry(root, env=env, platform_name=platform_name, host_id=host_id),
    )
    memory_check = next(check for check in checks if check.id == "memory.mem_palace")
    tool_check = next(check for check in checks if check.id == "tools.registry")
    memory_item = build_memory_readiness(memory_check)
    tool_items = build_tool_readiness(tool_check)
    workflow_items = build_workflow_readiness(
        checks=checks,
        memory_item=memory_item,
        tool_check=tool_check,
        workspace_root=root,
    )
    readiness_items = (memory_item, *tool_items, *workflow_items)
    status = overall_status(checks, readiness_items)
    return {
        "workflow_id": contract.WORKFLOW_ID,
        "status": status,
        "ok": status != "fail",
        "summary": {
            "checks": summarize_checks(checks),
            "readiness": summarize_readiness(readiness_items),
        },
        "memory": memory_item.to_dict(),
        "tools": {
            "host": tool_check.data.get("host", {}),
            "status_counts": tool_check.data.get("status_counts", {}),
            "family_status_counts": tool_check.data.get("family_status_counts", {}),
            "items": [item.to_dict() for item in tool_items],
        },
        "workflows": [item.to_dict() for item in workflow_items],
        "readiness": [item.to_dict() for item in readiness_items],
        "checks": [_check_report_dict(check) for check in checks],
    }
