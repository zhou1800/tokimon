"""Check an unpacked wheel in a disposable environment without pip or plugin installation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv
import zipfile


def check_wheel(wheel: Path, scratch_root: Path) -> dict:
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as temporary:
        root = Path(temporary).resolve()
        environment = root / "environment"
        venv.EnvBuilder(with_pip=False).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        site = Path(subprocess.check_output([str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
        with zipfile.ZipFile(wheel) as archive:
            for entry in archive.infolist():
                relative = Path(entry.filename)
                if entry.is_dir():
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe wheel member")
                if relative.parts[0].endswith(".data"):
                    if relative.parts[1] != "data":
                        raise ValueError("unexpected wheel data category")
                    target = environment / Path(*relative.parts[2:])
                else:
                    target = site / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))
        caller = root / "unrelated-project"
        caller.mkdir()
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env.update(
            PYTHONDONTWRITEBYTECODE="1",
            TOKIMON_PLUGIN_CREATOR_SCRIPTS_DIR=str(root / "no-legacy-helper"),
            TOKIMON_REQUIRE_PLUGIN_CREATOR="",
            TOKIMON_CLAUDE_CLI="claude-cli-not-installed-for-wheel-test",
            TOKIMON_REQUIRE_CLAUDE_CLI="",
        )
        def command(module: str, *args: str) -> dict:
            result = subprocess.run([str(python), "-I", "-m", module, *args], cwd=caller, env=env, text=True, capture_output=True, timeout=30)
            if result.returncode:
                raise AssertionError(f"{module} {args}: {result.stdout}\n{result.stderr}")
            return json.loads(result.stdout)
        harness = "tokimon.ai_harness_interface"
        workspace = environment / "share/tokimon"
        declared = command(harness, "contract", "--json")["contract"]
        assert declared["workspace_root"] == str(workspace)
        assert "chatgpt" not in declared["package_skill_ids"]
        configured = command(harness, "configure", "--yes", "--docker-features", "disabled", "--json")
        assert configured["settings_path"] == str(workspace / ".tokimon/settings.local.json")
        command(harness, "bootstrap-codex", "--json")
        checked = []
        for platform in ("codex", "claude"):
            generated = command(harness, "generate", "--target-platform", platform, "--json")
            validated = command(harness, "validate", "--target-platform", platform, "--json")
            assert generated["ok"] and validated["ok"]
            package = Path(generated["package_root"])
            mcp = json.loads((package / ".mcp.json").read_text())
            for server in mcp["mcpServers"].values():
                assert server["command"] == str(python)
                assert server.get("cwd", server.get("env", {}).get("PYTHONPATH")) == str(workspace)
            for relative in ("LICENSE", "THIRD_PARTY_NOTICES", "licenses/Matt-Pocock-MIT.txt", "licenses/Superpowers-MIT.txt", "licenses/Playwright-Apache-2.0.txt", "licenses/Playwright-NOTICE.txt"):
                assert (package / relative).read_bytes() == (workspace / relative).read_bytes()
            checked.append(platform)
        doctor = command("tokimon.workflow.doctor", "check", "--json")
        for item in doctor["checks"]:
            if item["id"] in ("repository.required_paths", "runtime.gitignore", "python.venv"):
                assert item["status"] == "pass", item
        pair_module = "tokimon.workflow.pair"
        pair_contract = command(pair_module, "contract", "--json")["contract"]
        assert set(pair_contract["peer_capabilities"]) == {"native-subagent", "local-cli"}
        pair_run = command(pair_module, "init", "--task-summary", "Wheel workflow check", "--json")
        command(pair_module, "select-peers", "--run-id", pair_run["run_id"], "--task-behavior", "read_only", "--cwd", str(caller), "--json")
        assert command(pair_module, "validate", "--run-id", pair_run["run_id"], "--json")["ok"]
        learning_run = command("tokimon.workflow.learning", "start-active", "--target-capability", "wheel-workflow-check", "--source", str(workspace / "skills/product-development/skill.md"), "--json")
        assert Path(learning_run["run_path"]).is_relative_to(workspace / ".tokimon/workflow/learning/runs")
        assert not (caller / ".tokimon").exists()
        assert not (workspace / ".venv").exists()
        return {"ok": True, "platforms_checked": checked, "workflows_checked": ["pair-artifact-support", "learning-run-initialization"], "caller_unchanged": True, "nested_venv_required": False, "pip_or_account_installation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_wheel(args.wheel.resolve(), args.scratch_root.resolve()), indent=2))
