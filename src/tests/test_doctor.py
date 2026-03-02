from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import cli
from doctor.runner import DoctorCheck, DoctorDeps, report_to_json_dict, run_doctor


def _completed(*, args: list[str], returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def _deps_ok(repo_root: Path) -> DoctorDeps:
    (repo_root / "AGENTS.md").write_text("ok\n")
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "helix.md").write_text("ok\n")
    (docs_dir / "repository-guidelines.md").write_text("ok\n")
    for rel in (".tokimon-tmp", "runs", "memory", "src/skills_generated"):
        (repo_root / rel).mkdir(parents=True, exist_ok=True)

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[:3] == ["git", "-C", str(repo_root)]:
            if argv[3:6] == ["rev-parse", "--show-toplevel"]:
                return _completed(args=argv, stdout=str(repo_root) + "\n")
            if argv[3:5] == ["status", "--porcelain"]:
                return _completed(args=argv, stdout="")
        if argv == ["codex", "--version"]:
            return _completed(args=argv, stdout="codex 1.2.3\n")
        return _completed(args=argv, returncode=1, stderr="unexpected argv")

    return DoctorDeps(
        repo_root=repo_root,
        run=run,
        which=lambda name: f"/usr/bin/{name}",
        path_exists=lambda path: path.exists(),
        dir_writable=lambda _path: True,
        port_free=lambda _host, _port: True,
    )


def test_doctor_report_ok_when_all_checks_ok(tmp_path: Path) -> None:
    deps = _deps_ok(tmp_path)
    report = run_doctor(deps, repair=False)
    assert report.ok is True
    assert [check.id for check in report.checks] == [
        "docs.required",
        "git.clean",
        "worktree.writable",
        "state.dirs",
        "skills.manifest",
        "codex.available",
        "port.8765",
    ]


def test_doctor_reports_missing_required_docs(tmp_path: Path) -> None:
    deps = _deps_ok(tmp_path)
    missing = tmp_path / "docs" / "helix.md"
    deps = DoctorDeps(
        repo_root=deps.repo_root,
        run=deps.run,
        which=deps.which,
        path_exists=lambda path: path != missing,
        dir_writable=deps.dir_writable,
        port_free=deps.port_free,
    )
    report = run_doctor(deps, repair=False)
    assert report.ok is False
    required = next(check for check in report.checks if check.id == "docs.required")
    assert required.ok is False
    assert "docs/helix.md" in required.details["missing"]


def test_doctor_fails_when_git_dirty(tmp_path: Path) -> None:
    deps = _deps_ok(tmp_path)
    orig_run = deps.run

    def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if argv[:3] == ["git", "-C", str(tmp_path)] and argv[3:5] == ["status", "--porcelain"]:
            return _completed(args=argv, stdout=" M src/cli.py\n")
        return orig_run(argv)

    deps = DoctorDeps(
        repo_root=deps.repo_root,
        run=run,
        which=deps.which,
        path_exists=deps.path_exists,
        dir_writable=deps.dir_writable,
        port_free=deps.port_free,
    )
    report = run_doctor(deps, repair=False)
    git_clean = next(check for check in report.checks if check.id == "git.clean")
    assert git_clean.ok is False
    assert report.ok is False


def test_doctor_reports_missing_state_dirs(tmp_path: Path) -> None:
    deps = _deps_ok(tmp_path)
    orig_path_exists = deps.path_exists
    missing_runs = tmp_path / "runs"
    deps = DoctorDeps(
        repo_root=deps.repo_root,
        run=deps.run,
        which=deps.which,
        path_exists=lambda path: False if path == missing_runs else orig_path_exists(path),
        dir_writable=deps.dir_writable,
        port_free=deps.port_free,
    )
    report = run_doctor(deps, repair=False)
    state_dirs = next(check for check in report.checks if check.id == "state.dirs")
    assert state_dirs.ok is False
    assert "runs" in state_dirs.details["missing"]


def test_doctor_reports_invalid_skills_manifest_json(tmp_path: Path) -> None:
    deps = _deps_ok(tmp_path)
    manifest_path = tmp_path / "src" / "skills_generated" / "manifest.json"
    manifest_path.write_text("{ not json")
    report = run_doctor(deps, repair=False)
    manifest = next(check for check in report.checks if check.id == "skills.manifest")
    assert manifest.ok is False
    assert manifest.details["path"] == "src/skills_generated/manifest.json"


def test_doctor_json_dict_is_stable(tmp_path: Path) -> None:
    deps = _deps_ok(tmp_path)
    report = run_doctor(deps, repair=True)
    payload = report_to_json_dict(report)
    assert payload["ok"] is True
    assert payload["repairs_attempted"] is True
    assert payload["repairs_applied"] is False
    assert [check["id"] for check in payload["checks"]] == [check.id for check in report.checks]


def test_cli_doctor_json_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import doctor.runner as doctor_runner

    deps = _deps_ok(tmp_path)
    monkeypatch.setattr(doctor_runner, "default_deps", lambda _root: deps)
    exit_code = cli.main(["doctor", "--json"])
    assert exit_code == 0
    raw = capsys.readouterr().out
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert {check["id"] for check in payload["checks"]} == {check.id for check in run_doctor(deps).checks}


def test_cli_doctor_fix_alias_sets_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import doctor.runner as doctor_runner

    deps = _deps_ok(tmp_path)
    monkeypatch.setattr(doctor_runner, "default_deps", lambda _root: deps)
    exit_code = cli.main(["doctor", "--json", "--fix"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repairs_attempted"] is True


def test_doctor_check_serialises_details(tmp_path: Path) -> None:
    report = report_to_json_dict(
        type(
            "FakeReport",
            (),
            {
                "ok": False,
                "repairs_attempted": False,
                "repairs_applied": False,
                "checks": [DoctorCheck(id="x", ok=False, summary="no", details={"a": 1})],
            },
        )()
    )
    assert report["checks"][0]["details"] == {"a": 1}
