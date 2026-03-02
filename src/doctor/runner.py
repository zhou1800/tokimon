from __future__ import annotations

import os
import json
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DoctorCheck:
    id: str
    ok: bool
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None
    fixable: bool = False
    fix_applied: bool = False


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    checks: list[DoctorCheck]
    repairs_attempted: bool
    repairs_applied: bool


RunFn = Callable[[list[str]], subprocess.CompletedProcess[str]]
WhichFn = Callable[[str], str | None]
PathExistsFn = Callable[[Path], bool]
DirWritableFn = Callable[[Path], bool]
PortFreeFn = Callable[[str, int], bool]


@dataclass(frozen=True)
class DoctorDeps:
    repo_root: Path
    run: RunFn
    which: WhichFn
    path_exists: PathExistsFn
    dir_writable: DirWritableFn
    port_free: PortFreeFn


def default_deps(repo_root: Path) -> DoctorDeps:
    return DoctorDeps(
        repo_root=repo_root,
        run=_run_command,
        which=shutil.which,
        path_exists=lambda path: path.exists(),
        dir_writable=lambda path: os.access(path, os.W_OK),
        port_free=_port_free,
    )


def run_doctor(deps: DoctorDeps, *, repair: bool = False) -> DoctorReport:
    checks = [
        _check_required_docs(deps),
        _check_git_clean(deps),
        _check_worktree_writable(deps),
        _check_state_dirs(deps),
        _check_skills_manifest(deps),
        _check_codex_available(deps),
        _check_port_8765(deps),
    ]
    ok = all(check.ok for check in checks)
    repairs_applied = False
    # Phase 1 only supports safe, non-destructive repairs. For the initial checks we only report remediation.
    return DoctorReport(ok=ok, checks=checks, repairs_attempted=bool(repair), repairs_applied=repairs_applied)


def report_to_json_dict(report: DoctorReport) -> dict[str, Any]:
    checks = [
        {
            "id": check.id,
            "ok": check.ok,
            "summary": check.summary,
            "details": check.details,
            "remediation": check.remediation,
            "fixable": check.fixable,
            "fix_applied": check.fix_applied,
        }
        for check in report.checks
    ]
    return {
        "ok": report.ok,
        "repairs_attempted": report.repairs_attempted,
        "repairs_applied": report.repairs_applied,
        "checks": checks,
    }


def render_human(report: DoctorReport) -> str:
    lines: list[str] = []
    lines.append("tokimon doctor")
    lines.append("status: ok" if report.ok else "status: fail")
    for check in report.checks:
        status = "ok" if check.ok else "fail"
        lines.append(f"- [{status}] {check.id}: {check.summary}")
        if not check.ok and check.remediation:
            lines.append(f"  remediation: {check.remediation}")
    if report.repairs_attempted:
        if report.repairs_applied:
            lines.append("repairs: applied")
        else:
            lines.append("repairs: none (phase 1 supports only safe, non-destructive repairs)")
    return "\n".join(lines) + "\n"


def _run_command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, check=False)


def _port_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def _check_required_docs(deps: DoctorDeps) -> DoctorCheck:
    required = ["AGENTS.md", "docs/helix.md", "docs/repository-guidelines.md"]
    missing = [rel for rel in required if not deps.path_exists(deps.repo_root / rel)]
    if missing:
        return DoctorCheck(
            id="docs.required",
            ok=False,
            summary="missing required docs",
            details={"missing": missing, "required": required},
            remediation="Run from the Tokimon repo root and ensure required docs are present.",
        )
    return DoctorCheck(
        id="docs.required",
        ok=True,
        summary="required docs present",
        details={"required": required},
    )


def _check_git_clean(deps: DoctorDeps) -> DoctorCheck:
    if deps.which("git") is None:
        return DoctorCheck(
            id="git.clean",
            ok=False,
            summary="git not available",
            details={"binary": "git"},
            remediation="Install git and ensure it is available on PATH.",
        )

    toplevel = deps.run(["git", "-C", str(deps.repo_root), "rev-parse", "--show-toplevel"])
    raw = (toplevel.stdout or "").strip()
    if toplevel.returncode != 0 or not raw:
        return DoctorCheck(
            id="git.clean",
            ok=False,
            summary="not a git worktree",
            details={"stderr": (toplevel.stderr or "").strip()},
            remediation="Run inside a git worktree (or clone the Tokimon repository).",
        )
    resolved_root = deps.repo_root.resolve()
    resolved_toplevel = Path(raw).resolve()
    if resolved_root != resolved_toplevel:
        return DoctorCheck(
            id="git.clean",
            ok=False,
            summary="not at git toplevel",
            details={"repo_root": str(resolved_root), "git_toplevel": str(resolved_toplevel)},
            remediation="Run from the git toplevel so worktrees/merges are deterministic.",
        )

    status = deps.run(["git", "-C", str(deps.repo_root), "status", "--porcelain"])
    if status.returncode != 0:
        return DoctorCheck(
            id="git.clean",
            ok=False,
            summary="git status failed",
            details={"stderr": (status.stderr or "").strip()},
            remediation="Ensure git works in this repo and rerun `git status --porcelain`.",
        )
    changes = [line for line in (status.stdout or "").splitlines() if line.strip()]
    if changes:
        return DoctorCheck(
            id="git.clean",
            ok=False,
            summary="git checkout has uncommitted changes",
            details={"changes": changes[:50]},
            remediation="Commit, stash, or revert changes until `git status --porcelain` is empty.",
        )
    return DoctorCheck(id="git.clean", ok=True, summary="git checkout clean")


def _check_worktree_writable(deps: DoctorDeps) -> DoctorCheck:
    writable = deps.dir_writable(deps.repo_root)
    if writable:
        return DoctorCheck(id="worktree.writable", ok=True, summary="worktree writable")
    return DoctorCheck(
        id="worktree.writable",
        ok=False,
        summary="worktree not writable",
        details={"path": str(deps.repo_root)},
        remediation="Ensure the repo directory is writable (permissions or read-only mount).",
    )


def _check_state_dirs(deps: DoctorDeps) -> DoctorCheck:
    required = [
        ".tokimon-tmp",
        "runs",
        "memory",
        "src/skills_generated",
    ]
    missing: list[str] = []
    not_writable: list[str] = []
    for rel in required:
        path = deps.repo_root / rel
        if not deps.path_exists(path):
            missing.append(rel)
            continue
        if not deps.dir_writable(path):
            not_writable.append(rel)
    if missing or not_writable:
        details: dict[str, Any] = {"required": required}
        if missing:
            details["missing"] = missing
        if not_writable:
            details["not_writable"] = not_writable
        return DoctorCheck(
            id="state.dirs",
            ok=False,
            summary="state directories missing or not writable",
            details=details,
            remediation="Create the missing dirs under the repo root and ensure they are writable.",
        )
    return DoctorCheck(
        id="state.dirs",
        ok=True,
        summary="state directories present",
        details={"required": required},
    )


def _check_skills_manifest(deps: DoctorDeps) -> DoctorCheck:
    manifest_rel = "src/skills_generated/manifest.json"
    manifest_path = deps.repo_root / manifest_rel
    if not deps.path_exists(manifest_path):
        return DoctorCheck(
            id="skills.manifest",
            ok=True,
            summary="generated skills manifest absent (ok)",
            details={"path": manifest_rel},
        )
    try:
        raw = manifest_path.read_text(encoding="utf-8", errors="replace")
        parsed = json.loads(raw)
    except Exception as exc:
        return DoctorCheck(
            id="skills.manifest",
            ok=False,
            summary="generated skills manifest invalid JSON",
            details={"path": manifest_rel, "error": str(exc)},
            remediation="Fix JSON in src/skills_generated/manifest.json or delete it to regenerate a clean manifest.",
        )
    if not isinstance(parsed, dict):
        return DoctorCheck(
            id="skills.manifest",
            ok=False,
            summary="generated skills manifest must be a JSON object",
            details={"path": manifest_rel, "type": type(parsed).__name__},
            remediation="Rewrite src/skills_generated/manifest.json to an object like {\"skills\": []}.",
        )
    skills = parsed.get("skills")
    if skills is None:
        return DoctorCheck(
            id="skills.manifest",
            ok=False,
            summary="generated skills manifest missing 'skills' list",
            details={"path": manifest_rel},
            remediation="Add a top-level 'skills' list (e.g., {\"skills\": []}).",
        )
    if not isinstance(skills, list):
        return DoctorCheck(
            id="skills.manifest",
            ok=False,
            summary="generated skills manifest 'skills' must be a list",
            details={"path": manifest_rel, "type": type(skills).__name__},
            remediation="Rewrite src/skills_generated/manifest.json so 'skills' is a list.",
        )
    return DoctorCheck(
        id="skills.manifest",
        ok=True,
        summary="generated skills manifest valid",
        details={"path": manifest_rel, "skills_count": len(skills)},
    )


def _check_codex_available(deps: DoctorDeps) -> DoctorCheck:
    codex_path = deps.which("codex")
    if not codex_path:
        return DoctorCheck(
            id="codex.available",
            ok=False,
            summary="codex not found on PATH",
            details={"binary": "codex"},
            remediation="Install the Codex CLI and ensure `codex` is available on PATH.",
        )

    version = deps.run(["codex", "--version"])
    if version.returncode != 0:
        return DoctorCheck(
            id="codex.available",
            ok=False,
            summary="codex --version failed",
            details={"binary": codex_path, "stderr": (version.stderr or "").strip()},
            remediation="Fix the Codex CLI installation so `codex --version` succeeds.",
        )
    return DoctorCheck(
        id="codex.available",
        ok=True,
        summary="codex available",
        details={"binary": codex_path, "version": (version.stdout or "").strip()},
    )


def _check_port_8765(deps: DoctorDeps) -> DoctorCheck:
    port = 8765
    host = "127.0.0.1"
    note = "chat-ui and gateway both default to port 8765; run only one on the default port or override one port."
    free = deps.port_free(host, port)
    if free:
        return DoctorCheck(
            id="port.8765",
            ok=True,
            summary="port 8765 available (chat-ui + gateway default)",
            details={"host": host, "port": port, "note": note},
        )
    return DoctorCheck(
        id="port.8765",
        ok=False,
        summary="port 8765 is in use (chat-ui + gateway default)",
        details={"host": host, "port": port, "note": note},
        remediation="Stop the process using port 8765 or override one service port (e.g., `tokimon chat-ui --port 8766`).",
    )
