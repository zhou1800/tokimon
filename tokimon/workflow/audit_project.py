"""Invocation-scoped subject identity and project-isolated audit state."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokimon.runtime import resolve_runtime_path, runtime_path_context, source_root


@dataclass(frozen=True)
class AuditProject:
    root: Path
    project_id: str

    def to_dict(self) -> dict[str, str]:
        return {"project_id": self.project_id, "project_root": str(self.root)}

    def state_root(self, workflow_id: str) -> Path:
        return resolve_runtime_path(
            Path(".tokimon/workflow") / workflow_id / "projects" / self.project_id,
            workspace_root=source_root(),
        )


_active: ContextVar[AuditProject | None] = ContextVar("audit_project", default=None)


def resolve(directory: Path | str) -> AuditProject:
    root = Path(directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"project root is not a directory: {root}")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env={**os.environ, "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Git timed out while resolving the audit project") from exc
    if result.returncode == 0:
        root = Path(result.stdout.strip()).resolve(strict=True)
    elif "not a git repository" not in result.stderr:
        raise ValueError(f"could not resolve project root: {result.stderr.strip()}")
    primary = runtime_path_context(workspace_root=root).primary_workspace_root
    project_id = hashlib.sha256(str(primary).encode("utf-8")).hexdigest()[:24]
    return AuditProject(root, project_id)


def current() -> AuditProject:
    return _active.get() or resolve(Path.cwd())


def root() -> Path:
    return current().root


@contextmanager
def selected(value: AuditProject) -> Iterator[None]:
    token = _active.set(value)
    try:
        yield
    finally:
        _active.reset(token)


def from_artifact(payload: Mapping[str, Any]) -> AuditProject | None:
    if not isinstance(payload, Mapping):
        raise ValueError("matrix input must be a JSON object")
    binding = payload.get("project")
    if binding is None:
        return None
    if not isinstance(binding, Mapping) or not binding.get("project_root"):
        raise ValueError("invalid audit project binding")
    value = resolve(str(binding["project_root"]))
    if value.to_dict() != binding:
        raise ValueError("audit project identity no longer matches its recorded checkout")
    return value


def validate_binding(payload: Mapping[str, Any]) -> None:
    if payload.get("project") != current().to_dict():
        raise ValueError("audit project does not match the selected checkout")


def runtime_path(path: Path | str | None, *, workflow_id: str, default: str) -> Path:
    if path is None:
        return current().state_root(workflow_id) / default
    candidate = Path(path).expanduser()
    return candidate.resolve() if not candidate.is_absolute() else candidate


def validate_component(component: str, *, label: str) -> None:
    """Reject path syntax where a single artifact component is required."""
    if (
        not isinstance(component, str) or not component.strip()
        or component in {".", ".."} or "/" in component or "\\" in component
        or Path(component).is_absolute()
    ):
        raise ValueError(f"{label} must be a non-empty path component")


def contained_child(directory: Path, component: str, *, label: str) -> Path:
    """Validate an artifact component and reject resolved directory escapes."""
    validate_component(component, label=label)
    child = directory / component
    try:
        contained = child.resolve().is_relative_to(directory.resolve())
    except RuntimeError as exc:
        raise ValueError(f"{label} cannot resolve its configured path") from exc
    if not contained:
        raise ValueError(f"{label} escapes its configured directory")
    return child
