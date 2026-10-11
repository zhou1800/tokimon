"""Codex compatibility authoring with optional legacy plugin-creator helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from tokimon.ai_harness_interface import contract, marketplace, plugin_manifest


SCRIPTS_DIR_ENV = "TOKIMON_PLUGIN_CREATOR_SCRIPTS_DIR"
PYTHON_ENV = "TOKIMON_PLUGIN_CREATOR_PYTHON"
REQUIRE_ENV = "TOKIMON_REQUIRE_PLUGIN_CREATOR"


def default_scripts_dir() -> Path:
    override = os.environ.get(SCRIPTS_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return Path.home() / ".codex" / "skills" / ".system" / "plugin-creator" / "scripts"


def python_executable() -> str:
    return os.environ.get(PYTHON_ENV, sys.executable)


def require_plugin_creator() -> bool:
    return os.environ.get(REQUIRE_ENV, "").strip().lower() in {"1", "true", "yes", "required"}


def bootstrap_codex_scaffold(
    *,
    output_root: Path,
    marketplace_path: Path,
    force: bool = False,
    required: bool | None = None,
) -> dict[str, Any]:
    command = bootstrap_command(output_root=output_root, marketplace_path=marketplace_path, force=force)
    if not (default_scripts_dir() / "create_basic_plugin.py").is_file() and not _helper_required(required):
        targets = {
            contract.manifest_path(contract.package_root(output_root)): plugin_manifest.apply_manifest_overlay({}),
            contract.package_root(output_root) / ".mcp.json": {"mcpServers": {}},
            marketplace_path: marketplace.apply_tokimon_marketplace_content({}),
        }
        try:
            skills_root = contract.package_root(output_root) / "skills"
            for path in (*targets, skills_root):
                _check_authoring_path(path, root=output_root)
            for path in targets:
                if path.exists() and not path.is_file():
                    raise ValueError(f"scaffold file must be a regular file: {path}")
            if skills_root.exists() and not skills_root.is_dir():
                raise ValueError(f"scaffold skills path must be a directory: {skills_root}")
            skills_root.mkdir(parents=True, exist_ok=True)
            written = []
            for path, payload in targets.items():
                if path.exists() and not force:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                written.append(str(path))
        except (OSError, ValueError) as exc:
            return {"name": "codex-scaffold", "ok": False, "method": "built-in", "message": str(exc)}
        return {
            "name": "codex-scaffold", "ok": True, "method": "built-in",
            "legacy_helper_available": False, "files_written": written,
            "stdout": "Supported Codex compatibility scaffold is ready.",
        }
    return _run_script(
        name="plugin-creator-bootstrap",
        script_name="create_basic_plugin.py",
        command=command,
        required=required,
    )


def bootstrap_command(*, output_root: Path, marketplace_path: Path, force: bool = False) -> list[str]:
    command = [
        python_executable(),
        str(default_scripts_dir() / "create_basic_plugin.py"),
        contract.PACKAGE_ID,
        "--path",
        str(output_root / "plugins"),
        "--marketplace-path",
        str(marketplace_path),
        "--marketplace-name",
        contract.MARKETPLACE_NAME,
        "--with-skills",
        "--with-mcp",
        "--with-marketplace",
    ]
    if force:
        command.append("--force")
    return command


def validate_plugin(*, package_root: Path, required: bool | None = None) -> dict[str, Any]:
    command = [
        python_executable(),
        str(default_scripts_dir() / "validate_plugin.py"),
        str(package_root),
    ]
    return _run_script(
        name="plugin-creator-validator",
        script_name="validate_plugin.py",
        command=command,
        required=required,
    )


def update_cachebuster(
    *,
    package_root: Path,
    cachebuster: str | None = None,
    required: bool | None = None,
) -> dict[str, Any]:
    command = [
        python_executable(),
        str(default_scripts_dir() / "update_plugin_cachebuster.py"),
        str(package_root),
    ]
    if cachebuster:
        command.extend(["--cachebuster", cachebuster])
    if not (default_scripts_dir() / "update_plugin_cachebuster.py").is_file() and not _helper_required(required):
        path = contract.manifest_path(package_root)
        try:
            _check_authoring_path(path, root=package_root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("plugin manifest must be a JSON object")
            version = str(payload.get("version", "")).split("+", 1)[0]
            updated = version + "+codex." + (cachebuster or uuid.uuid4().hex)
            plugin_manifest.validate_version(updated)
            payload["version"] = updated
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            return {"name": "codex-cachebuster", "ok": False, "method": "built-in", "message": str(exc)}
        return {"name": "codex-cachebuster", "ok": True, "method": "built-in", "stdout": updated}
    return _run_script(
        name="plugin-creator-cachebuster",
        script_name="update_plugin_cachebuster.py",
        command=command,
        required=required,
    )


def read_marketplace_name(*, marketplace_path: Path, required: bool | None = None) -> dict[str, Any]:
    command = [
        python_executable(),
        str(default_scripts_dir() / "read_marketplace_name.py"),
        "--marketplace-path",
        str(marketplace_path),
    ]
    return _run_script(
        name="plugin-creator-marketplace-name",
        script_name="read_marketplace_name.py",
        command=command,
        required=required,
    )


def _run_script(
    *,
    name: str,
    script_name: str,
    command: list[str],
    required: bool | None,
) -> dict[str, Any]:
    must_run = _helper_required(required)
    script_path = default_scripts_dir() / script_name
    if not script_path.is_file():
        return {
            "name": name,
            "ok": not must_run,
            "available": False,
            "skipped": True,
            "script": str(script_path),
            "command": command,
            "message": f"missing plugin-creator script: {script_path}",
        }
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "name": name,
        "ok": result.returncode == 0,
        "available": True,
        "skipped": False,
        "script": str(script_path),
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _helper_required(required: bool | None) -> bool:
    return require_plugin_creator() if required is None else required


def _check_authoring_path(path: Path, *, root: Path) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"scaffold target must remain inside {root}")
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise ValueError(f"scaffold target must not follow symbolic links: {entry}")
        if entry == root:
            break
    if path.exists() and not (path.is_file() or path.is_dir()):
        raise ValueError(f"scaffold target must be a regular file or directory: {path}")
