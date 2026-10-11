from __future__ import annotations

import copy
import json
import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

from tests.tools.mcp.container_executor.test_deployment import FakeDocker
from tokimon.tools.mcp.container_executor.workspaces import WorkspaceController
from tokimon.tools.mcp.container_executor.cli import main
from tokimon.tools.mcp.container_executor.gateway import list_tools


class WorkspaceDocker(FakeDocker):
    def __init__(self):
        super().__init__()
        self.gateway_routes = {}
        self.fail_gateway = False

    def run(self, *args, timeout=30):
        if args[:2] == ("container", "exec"):
            if args[-1] == "probe":
                container = self.resources["container"][args[4]]
                worker = self.images[container["image"]]
                identity = self.identity(worker)
                return json.dumps({"identity": identity, "workspaces": sorted(self.gateway_routes[container["image"]]["routes"]), "tunnel_ready": False,
                                   "discovery": {"result": {"tools": list_tools()}}})
            if "request" in args and args[args.index("request") + 1] == "tools/call":
                executor = self.resources["container"][args[4]]
                worker = self.images[executor["image"]]
                payload = {"identity": self.identity(worker), "deployment": worker["name"], "isolation": json.loads(Path(executor["policy"]).read_text()), "active_jobs": 0}
                return json.dumps({"result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})
        if args[:2] == ("container", "start") and self.fail_gateway:
            item = self.resources["container"][args[-1]]
            if item["Config"]["Labels"]["tokimon.role"] == "gateway":
                self.fail_gateway = False
                raise RuntimeError("injected gateway start failure")
        output = super().run(*args, timeout=timeout)
        if args[0] == "build":
            image = self.resources["image"][args[args.index("--tag") + 1]]["Id"]
            self.gateway_routes[image] = json.loads((Path(args[-1]) / "gateway.json").read_text())
        return output

    @staticmethod
    def identity(worker):
        return {"Uid": " ".join([str(worker["uid"])] * 4), "Gid": " ".join([str(worker["gid"])] * 4), "Groups": "", "NoNewPrivs": "1",
                **{key: "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")}}


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / "a", self.root / "b"
        self.a.mkdir(); self.b.mkdir()
        self.docker = WorkspaceDocker()
        self.controller = WorkspaceController(workspace_root=self.root, docker=self.docker)
        patch = mock.patch("tokimon.tools.mcp.container_executor.deployment.host_networks", return_value=[])
        patch.start(); self.addCleanup(patch.stop)

    def test_gateway_create_timeout_after_commit_recovers_without_extra_container(self):
        run = self.docker.run
        def uncertain(*args, **kwargs):
            result = run(*args, **kwargs)
            if args[:2] == ("container", "create") and "tokimon.role=gateway" in args:
                raise subprocess.TimeoutExpired("docker container create", 120)
            return result
        self.docker.run = uncertain
        self.assertTrue(self.controller.start(self.a, local=True)["ok"])
        self.assertEqual(len(self.docker.resources["container"]), 2)
        self.controller.stop(self.a)
        self.controller.remove(self.a, purge_data=True)
        self.assertFalse(any(self.docker.resources[kind] for kind in ("container", "volume", "network")))

    def test_missing_profile_fails_without_docker_and_offers_private_runtime_location(self):
        result = self.controller.start(self.a)
        self.assertFalse(result["ok"])
        self.assertIn("secrets/tunnel.key", result["setup"]["recommended_key_file"])
        key = Path(result["setup"]["recommended_key_file"])
        self.assertTrue(result["setup"]["key_file_created"])
        self.assertEqual(key.read_text(), "REPLACE_WITH_YOUR_OPENAI_TUNNEL_RUNTIME_KEY\n")
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertEqual(key.stat().st_uid, os.geteuid())
        repeated = self.controller.start(self.a)
        self.assertFalse(repeated["setup"]["key_file_created"])
        self.assertFalse(self.docker.calls)

    def test_missing_setup_preserves_existing_credentials_and_permissions(self):
        key = self.controller.root / "secrets/tunnel.key"
        key.parent.mkdir(mode=0o700)
        for content, mode in ((b"private-existing-canary\n", 0o600), (b"", 0o640)):
            with self.subTest(mode=mode):
                key.write_bytes(content); key.chmod(mode)
                before = key.stat()
                result = self.controller.start(self.a)
                self.assertFalse(result["setup"]["key_file_created"])
                self.assertEqual(key.read_bytes(), content)
                after = key.stat()
                self.assertEqual((after.st_mode, after.st_mtime_ns, after.st_ino),
                                 (before.st_mode, before.st_mtime_ns, before.st_ino))
                self.assertNotIn("private-existing-canary", json.dumps(result))
        self.assertFalse(self.docker.calls)

    def test_concurrent_missing_setup_attempts_create_one_private_template(self):
        results, errors = [], []
        def start(path):
            try:
                results.append(self.controller.start(path))
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=start, args=(path,)) for path in (self.a, self.b)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(5)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(sum(result["setup"]["key_file_created"] for result in results), 1)
        self.assertEqual(len({result["setup"]["recommended_key_file"] for result in results}), 1)
        key = Path(results[0]["setup"]["recommended_key_file"])
        self.assertEqual(key.read_text(), "REPLACE_WITH_YOUR_OPENAI_TUNNEL_RUNTIME_KEY\n")
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertFalse(self.docker.calls)

    def test_key_template_refuses_symlink_parent_and_special_targets(self):
        secrets = self.controller.root / "secrets"
        secrets.symlink_to(self.b, target_is_directory=True)
        with self.assertRaises(OSError):
            self.controller.start(self.a)
        self.assertFalse((self.b / "tunnel.key").exists())
        secrets.unlink(); secrets.mkdir()
        key = secrets / "tunnel.key"
        target = self.b / "private-key"
        target.write_text("private-existing-canary"); target.chmod(0o600)
        key.symlink_to(target)
        with self.assertRaises(OSError):
            self.controller.start(self.a)
        self.assertEqual(target.read_text(), "private-existing-canary")
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        key.unlink(); os.mkfifo(key)
        with self.assertRaisesRegex(ValueError, "regular"):
            self.controller.start(self.a)
        self.assertFalse(self.docker.calls)

    def test_unfinished_key_refuses_setup_before_download_or_profile_save(self):
        result = self.controller.start(self.a)
        key = Path(result["setup"]["recommended_key_file"])
        marker = key.read_bytes()
        with mock.patch("tokimon.tools.mcp.container_executor.workspaces.download_tunnel_client") as download:
            for content in (marker, b"", b" \n\t", b"  " + marker + b" \n"):
                with self.subTest(content=content):
                    key.write_bytes(content)
                    output = io.StringIO()
                    with mock.patch("tokimon.tools.mcp.container_executor.cli.WorkspaceController", return_value=self.controller), redirect_stdout(output):
                        code = main(["workspace", "setup", "--tunnel-id", "tunnel_canary", "--key-file", str(key), "--json"])
                    self.assertEqual(code, 1)
                    payload = json.loads(output.getvalue())
                    self.assertIn("fill the private tunnel key file", payload["error"])
                    self.assertNotIn(marker.decode().strip(), output.getvalue())
                    self.assertFalse((self.controller.store.root / "profile.json").exists())
            download.assert_not_called()
        self.assertFalse(self.docker.calls)

    def test_configured_start_refuses_key_replaced_with_template_before_docker(self):
        result = self.controller.start(self.a)
        key = Path(result["setup"]["recommended_key_file"])
        marker = key.read_bytes()
        key.write_text("private-canary\n")
        binary = self.root / "client"
        binary.write_bytes(b"canary"); binary.chmod(0o755)
        self.controller.setup("tunnel_canary", key, binary)
        key.write_bytes(marker)
        with self.assertRaisesRegex(ValueError, "unchanged template"):
            self.controller.start(self.a)
        self.assertFalse(self.docker.calls)

    def test_explicit_local_start_does_not_draft_credentials(self):
        self.assertTrue(self.controller.start(self.a, local=True)["ok"])
        self.assertFalse((self.controller.root / "secrets").exists())

    def test_two_directories_idempotent_scoped_stop_and_final_gateway_stop(self):
        first = self.controller.start(self.a, local=True)
        repeated = self.controller.start(self.a, local=True)
        self.assertEqual(first["containers"], repeated["containers"])
        second = self.controller.start(self.b, local=True)
        self.assertEqual(set(first["containers"]), {"executor"})
        self.assertEqual(set(second["containers"]), {"executor"})
        self.assertEqual(len(self.docker.resources["container"]), 3)
        self.assertNotEqual(first["workspace_id"], second["workspace_id"])
        self.assertEqual(len(self.controller.list()["workspaces"]), 2)
        stopped = self.controller.stop(self.a)
        self.assertFalse(stopped["running"])
        self.assertTrue(self.controller.status(self.b)["local_ready"])
        self.assertEqual(len(self.controller.list()["workspaces"]), 1)
        self.assertEqual(len(self.controller.list(include_stopped=True)["workspaces"]), 2)
        self.controller.stop(self.b)
        self.assertFalse(self.controller.gateway.status()["running"])
        self.controller.remove(self.a, purge_data=True)
        self.controller.remove(self.b, purge_data=True)
        self.assertFalse(self.docker.resources["container"])
        self.assertFalse(self.docker.resources["volume"])
        self.assertFalse(self.docker.resources["network"])

    def test_concurrent_starts_publish_both_and_failed_refresh_preserves_old_routes(self):
        errors = []
        def start(path):
            try:
                self.assertTrue(self.controller.start(path, local=True)["ok"])
            except BaseException as exc:
                errors.append(exc)
        threads = [threading.Thread(target=start, args=(path,)) for path in (self.a, self.b)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(5)
        self.assertFalse(errors)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(len(self.controller.gateway.status()["workspaces"]), 2)
        self.controller.stop(self.b)
        existing = self.controller.status(self.a)["containers"]
        self.docker.fail_gateway = True
        failed = self.controller.start(self.b, local=True)
        self.assertFalse(failed["ok"])
        self.assertTrue(failed["local_ready"])
        self.assertFalse(failed["gateway_ready"])
        self.assertEqual(self.controller.status(self.a)["containers"], existing)
        self.assertEqual(self.controller.gateway.status()["workspaces"], [self.controller.status(self.a)["workspace_id"]])

    def test_no_follow_registry_and_directory_replacement_refuse(self):
        first = self.controller.start(self.a, local=True)
        self.a.rename(self.root / "old-a"); self.a.mkdir()
        with self.assertRaisesRegex(ValueError, "replaced"):
            self.controller.start(self.a, local=True)
        self.assertFalse(self.controller.stop(identity=first["workspace_id"])["running"])
        registry = self.controller.store.root / "registry.json"
        registry.unlink(); os.mkfifo(registry)
        with self.assertRaisesRegex(ValueError, "regular"):
            self.controller.list()

    def test_public_cli_preserves_caller_directory_and_shapes_errors(self):
        def command(argv):
            output = io.StringIO()
            with mock.patch("tokimon.tools.mcp.container_executor.cli.WorkspaceController", return_value=self.controller), redirect_stdout(output):
                status = main(["workspace", *argv, "--json"])
            return status, json.loads(output.getvalue())
        with mock.patch("tokimon.tools.mcp.container_executor.workspaces.Path.cwd", return_value=self.a):
            status, missing = command(["start"])
            self.assertEqual(status, 1)
            self.assertTrue(missing["setup"]["key_file_created"])
            self.assertEqual(Path(missing["setup"]["recommended_key_file"]).stat().st_mode & 0o777, 0o600)
            status, started = command(["start", "--local"])
            self.assertEqual(status, 0)
            self.assertEqual(started["directory"], str(self.a))
            self.assertEqual(command(["list"])[1]["workspaces"][0]["workspace_id"], started["workspace_id"])
            self.assertFalse(command(["stop"])[1]["running"])

    def test_setup_references_only_and_protected_root_masks_fail_closed(self):
        key = self.controller.root / "secrets/key"
        key.parent.mkdir(); key.write_text("private-canary-do-not-copy"); key.chmod(0o600)
        binary = self.root / "client"
        binary.write_bytes(b"client-canary"); binary.chmod(0o755)
        self.assertTrue(self.controller.setup("tunnel_canary", key, binary)["ok"])
        self.assertFalse(self.docker.calls)
        protected = self.root / ".env"
        protected.write_text("secret-canary"); protected.chmod(0o600)
        first = self.controller.start(self.root)
        self.assertTrue(first["ok"])
        self.assertFalse(first["tunnel_ready"])
        with self.assertRaisesRegex(ValueError, "stop directory"):
            self.controller.setup("other", key, binary)
        self.assertEqual(protected.read_text(), "secret-canary")
        self.assertEqual(protected.stat().st_mode & 0o777, 0o600)
        creates = [c for c in self.docker.calls if c[:2] == ("container", "create")]
        executor = next(c for c in creates if "--cap-add" in c)
        gateway = next(c for c in creates if "tokimon.role=gateway" in c)
        self.assertIn("/mounts/project/.tokimon/tools/mcp/container-executor/secrets:ro", " ".join(executor))
        self.assertNotIn("/mounts/project/.tokimon:ro", " ".join(executor))
        self.assertNotIn("dst=/mounts/project/client,readonly", " ".join(executor))
        self.assertIn("dst=/mounts/project/.env,readonly", " ".join(executor))
        self.assertNotIn("private-canary-do-not-copy", " ".join(creates[0]))
        self.assertNotIn("dst=/mounts/project", " ".join(gateway))
        self.assertIn("dst=/run/secrets/tunnel-key,readonly", " ".join(gateway))
        self.assertEqual(len(creates), 2)
        self.assertNotIn("dst=/run/secrets/tunnel-key", " ".join(executor))
        self.assertEqual(len(self.controller.gateway.tools()), 10)
        self.assertTrue(all(b"private-canary-do-not-copy" not in p.read_bytes()
                            for p in self.controller.root.rglob("bundle/**/*") if p.is_file()))
        self.controller.stop(self.root)
        bad = self.root / ".env.bad"
        os.mkfifo(bad)
        before = len(self.docker.calls)
        with self.assertRaisesRegex(ValueError, "protected target"):
            self.controller.start(self.root)
        self.assertEqual(len(self.docker.calls), before)

    def test_parent_grant_masks_exact_key_directory_without_hiding_checkout(self):
        checkout = self.root / "tokimon"
        checkout.mkdir()
        controller = WorkspaceController(workspace_root=checkout, docker=self.docker)
        key = controller.root / "secrets/key"
        key.parent.mkdir(); key.write_text("parent-key-canary"); key.chmod(0o600)
        binary = self.root / "client"
        binary.write_text("public-client-canary"); binary.chmod(0o755)
        controller.setup("tunnel_parent_canary", key, binary)
        first = controller.start(self.root)
        self.assertTrue(first["local_ready"] and first["gateway_ready"])
        executor = next(c for c in self.docker.calls if c[:2] == ("container", "create") and "--cap-add" in c)
        self.assertIn("/mounts/project/tokimon/.tokimon/tools/mcp/container-executor/secrets:ro", " ".join(executor))
        self.assertNotIn("/mounts/project/tokimon:ro", " ".join(executor))
        self.assertNotIn("/mounts/project/tokimon/.tokimon:ro", " ".join(executor))
        self.assertEqual(key.read_text(), "parent-key-canary")
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)

    def test_custom_key_directory_is_hidden_but_its_parent_project_is_granted(self):
        key = self.a / "credentials/key"
        key.parent.mkdir(); key.write_text("custom-key-canary"); key.chmod(0o600)
        binary = self.root / "client"
        binary.write_text("public-client-canary"); binary.chmod(0o755)
        self.controller.setup("tunnel_custom_canary", key, binary)
        self.assertTrue(self.controller.start(self.a)["ok"])
        executor = next(c for c in self.docker.calls if c[:2] == ("container", "create") and "--cap-add" in c)
        self.assertIn("/mounts/project/credentials:ro", " ".join(executor))
        self.assertNotIn("/mounts/project:ro", " ".join(executor))
        before = len(self.docker.calls)
        with self.assertRaisesRegex(ValueError, "protected|secret"):
            self.controller.start(key.parent)
        self.assertEqual(len(self.docker.calls), before)

    def test_local_parent_grant_rejects_nested_symlink_secret_directory_before_docker(self):
        checkout = self.root / "tokimon"
        checkout.mkdir()
        controller = WorkspaceController(workspace_root=checkout, docker=self.docker)
        (controller.root / "secrets").symlink_to(self.a, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "protected target"):
            controller.start(self.root, local=True)
        self.assertFalse(self.docker.calls)

    def test_unsafe_roots_and_sensitive_symlink_refuse_before_resources(self):
        for path in (Path("/"), self.controller.root):
            with self.assertRaises(ValueError):
                self.controller.start(path, local=True)
        (self.a / ".env").symlink_to(self.b / "missing")
        with self.assertRaisesRegex(ValueError, "protected target"):
            self.controller.start(self.a, local=True)
        self.assertFalse(self.docker.calls)

    def test_paired_service_migration_requires_stop_and_retains_data(self):
        first = self.controller.start(self.a, local=True)
        deployment = self.controller._deployment(first["workspace_id"])
        record = deployment._record()
        record["current"].pop("worker_layout")
        sidecar = self.docker.run("container", "create", "--name", "old-sidecar", *deployment._label_args(record, "transport"), record["current"]["image"])
        self.docker.run("container", "start", sidecar)
        record["containers"]["transport"] = sidecar
        deployment._save(record, "legacy_fixture")
        volumes = copy.deepcopy(self.docker.resources["volume"])
        before = len(self.docker.calls)
        with self.assertRaisesRegex(ValueError, "explicit stop"):
            self.controller.start(self.a, local=True)
        self.assertFalse(any(c[:2] in (("container", "rm"), ("container", "stop")) for c in self.docker.calls[before:]))
        self.controller.stop(self.a)
        migrated = self.controller.start(self.a, local=True)
        self.assertTrue(migrated["local_ready"])
        self.assertEqual(set(migrated["containers"]), {"executor"})
        self.assertNotIn(sidecar, self.docker.resources["container"])
        self.assertEqual(self.docker.resources["volume"], volumes)

    def test_failed_prerequisite_entry_can_be_retired_without_resource_adoption(self):
        with mock.patch.object(self.docker, "prerequisites", side_effect=RuntimeError("Docker unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                self.controller.start(self.a, local=True)
        self.assertFalse(self.controller.status(self.a)["installed"])
        self.assertTrue(self.controller.remove(self.a, purge_data=True)["ok"])
        self.assertFalse(self.controller.list(include_stopped=True)["workspaces"])

    def test_lost_key_does_not_block_stop_last_service_or_local_health(self):
        key = self.controller.root / "key"
        key.write_text("private-canary"); key.chmod(0o600)
        binary = self.root / "client"
        binary.write_bytes(b"canary"); binary.chmod(0o755)
        self.controller.setup("tunnel_canary", key, binary)
        first = self.controller.start(self.a)
        key.unlink()
        self.assertTrue(self.controller.status(self.a)["local_ready"])
        self.assertFalse(self.controller.stop(self.a)["running"])
        self.assertFalse(self.controller.gateway.status()["running"])

    def test_sensitive_directory_sources_are_refused_before_docker(self):
        for name in (".env", ".env.production", ".ssh"):
            path = self.a / name / "child"
            path.mkdir(parents=True)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "secret"):
                self.controller.start(path, local=True)
        self.assertFalse(self.docker.calls)

    def test_gateway_diagnostic_failure_still_restores_previous_routes(self):
        first = self.controller.start(self.a, local=True)
        calls = self.docker.run
        inspect = self.docker.inspect
        for stage in ("logs", "inspect"):
            with self.subTest(stage=stage):
                armed = False
                injected = 0
                before = set(self.controller.gateway.store.root.glob("*.json"))
                def fail_once(*args, timeout=30):
                    nonlocal armed, injected
                    if (args[:2] == ("container", "start") and self.docker.fail_gateway
                            and inspect("container", args[-1])["Config"]["Labels"].get("tokimon.role") == "gateway"):
                        armed = True
                    if armed and stage == "logs" and args[:2] == ("container", "logs"):
                        armed = False
                        injected += 1
                        raise RuntimeError("diagnostic logs unavailable")
                    return calls(*args, timeout=timeout)
                def inspect_once(kind, identity):
                    nonlocal armed, injected
                    if armed and stage == "inspect":
                        armed = False
                        injected += 1
                        raise RuntimeError("diagnostic inspect unavailable")
                    return inspect(kind, identity)
                self.docker.fail_gateway = True
                with mock.patch.object(self.docker, "run", side_effect=fail_once), mock.patch.object(self.docker, "inspect", side_effect=inspect_once):
                    failed = self.controller.start(self.b, local=True)
                self.assertEqual(injected, 1, stage)
                self.assertFalse(failed["ok"])
                self.assertTrue(failed["local_ready"])
                self.assertEqual(self.controller.gateway.status()["workspaces"], [first["workspace_id"]])
                self.assertEqual(self.controller.status(self.a)["containers"], first["containers"])
                created = set(self.controller.gateway.store.root.glob("*.json")) - before
                self.assertEqual(len(created), 1)
                receipt = json.loads(created.pop().read_text())
                self.assertEqual(receipt["error"], "injected gateway start failure")
                self.assertEqual(receipt["diagnostics"]["error"], f"diagnostic {stage} unavailable")
                self.controller.stop(self.b)

    def test_nested_gateway_record_corruption_fails_status_immediately(self):
        self.controller.start(self.a, local=True)
        pristine = self.controller.gateway.store.read("gateway.json")
        for key, value in (("uid", 0), ("limits", "invalid"), ("tunnel", {"arbitrary": "invalid"}), ("mounts", {})):
            corrupt = copy.deepcopy(pristine)
            corrupt["current"]["config"][key] = value
            self.controller.gateway.store.write("gateway.json", corrupt)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.controller.status(self.a)
        self.controller.gateway.store.write("gateway.json", pristine)

    def test_trusted_import_route_from_real_foreign_cwd_ignores_shadow_modules(self):
        checkout = Path(__file__).resolve().parents[4]
        (self.a / "tokimon.py").write_text("raise RuntimeError('caller module executed')\n")
        env = {**os.environ, "PYTHONPATH": str(checkout), "PYTHONDONTWRITEBYTECODE": "1"}
        help_result = subprocess.run([sys.executable, "-P", "-m", "tokimon.tools.mcp.container_executor", "workspace", "--help"],
                                     cwd=self.a, env=env, capture_output=True, text=True, timeout=5)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        script = '''import io,json,sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
from tokimon.tools.mcp.container_executor import cli
from tokimon.tools.mcp.container_executor.workspaces import WorkspaceController
from tests.tools.mcp.container_executor.test_workspaces import WorkspaceDocker
controller=WorkspaceController(workspace_root=Path(sys.argv[1]),docker=WorkspaceDocker())
results=[]
with mock.patch.object(cli,"WorkspaceController",return_value=controller),mock.patch("tokimon.tools.mcp.container_executor.deployment.host_networks",return_value=[]):
 for args in (["start"],["start","--local"],["list"],["stop"]):
  output=io.StringIO()
  with redirect_stdout(output): cli.main(["workspace",*args,"--json"])
  results.append(json.loads(output.getvalue()))
print(json.dumps(results))
'''
        completed = subprocess.run([sys.executable, "-P", "-c", script, str(self.root / "machine")], cwd=self.a, env=env,
                                   capture_output=True, text=True, timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        missing, started, listed, stopped = json.loads(completed.stdout)
        self.assertFalse(missing["ok"])
        self.assertIn("setup required", missing["error"])
        self.assertEqual(started["directory"], str(self.a))
        self.assertEqual(listed["workspaces"][0]["workspace_id"], started["workspace_id"])
        self.assertFalse(stopped["running"])
