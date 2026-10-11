from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokimon.tools.mcp.container_executor.deployment import Deployment, Docker
from tokimon.tools.mcp.container_executor.ownership import DockerCommandError, MissingResource


class FakeDocker(Docker):
    """Docker adapter double: inspect exposes exact resources created by commands."""

    def __init__(self):
        self.resources = {"container": {}, "volume": {}, "network": {}, "image": {}}
        self.images = {}
        self.number = 0
        self.calls = []
        self.fail_start = False
        self.active_jobs = 0
        self.paused = False
        self.on_build = None

    def prerequisites(self):
        return {"rootful_linux": True}

    def inspect(self, kind, identity):
        try:
            return copy.deepcopy(self.resources[kind][identity])
        except KeyError:
            if kind == "container":
                for item in self.resources[kind].values():
                    if item.get("Name") == "/" + identity:
                        return copy.deepcopy(item)
            raise MissingResource("resource missing") from None

    def run(self, *args, timeout=30):
        self.calls.append(args)
        kind = args[0]
        if kind == "build":
            if self.on_build:
                self.on_build()
            tag = args[args.index("--tag") + 1]
            image = "sha256:" + hashlib.sha256(tag.encode()).hexdigest()
            worker = json.loads((Path(args[-1]) / "worker.json").read_text())
            self.images[image] = worker
            self.resources["image"][tag] = {"Id": image}
            return "built"
        action = args[1]
        if kind == "volume" and action == "ls":
            return "\n".join(self.resources[kind])
        if action == "create":
            labels = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "--label")
            self.number += 1
            identity = args[-1] if kind == "volume" else f"{kind}-{self.number}"
            if kind == "container":
                image = next(arg for arg in args if arg.startswith("sha256:"))
                mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]
                policy_mount = next((arg for arg in mounts if "dst=/etc/tokimon/policy.json" in arg), None)
                self.resources[kind][identity] = {"Id": identity, "Name": "/" + args[args.index("--name") + 1], "Image": image, "Config": {"Labels": labels}, "State": {"Running": False, "Status": "created"}, "image": image,
                    "policy": policy_mount.split("src=", 1)[1].split(",", 1)[0] if policy_mount else None}
            else:
                self.resources[kind][identity] = {"Id": identity, "Name": identity, "Labels": labels}
            return identity
        if action in {"start", "stop"}:
            if action == "start" and self.fail_start:
                self.fail_start = False
                raise RuntimeError("injected start failure")
            item = self.resources[kind][args[-1]]
            item["State"] = {"Running": action == "start", "Status": "running" if action == "start" else "exited"}
            if action == "start" and item["Config"]["Labels"].get("tokimon.role") == "executor":
                self.paused = False
            return args[-1]
        if action == "rm":
            del self.resources[kind][args[-1]]
            return "removed"
        if action == "logs":
            return ""
        if kind == "container" and action == "exec":
            method = args[args.index("request") + 1]
            params = json.loads(args[args.index("request") + 2])
            if method == "tokimon/maintenance":
                if params["paused"] and self.active_jobs:
                    return json.dumps({"error": {"message": "active jobs prevent update/rollback"}})
                self.paused = params["paused"]
                return json.dumps({"result": {"paused": self.paused}})
            executor = self.resources[kind][args[args.index("--user") + 2]]
            config = self.images[executor["image"]]
            identity = {"Uid": " ".join([str(config["uid"])] * 4), "Gid": " ".join([str(config["gid"])] * 4), "Groups": "", "NoNewPrivs": "1", **{key: "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")}}
            payload = {"identity": identity, "deployment": config["name"], "isolation": json.loads(Path(executor["policy"]).read_text()), "active_jobs": self.active_jobs}
            if method != "tools/call":
                raise AssertionError(method)
            return json.dumps({"result": {"content": [{"type": "text", "text": json.dumps(payload)}]}})
        raise AssertionError(args)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fake = FakeDocker()
        self.deployment = Deployment("test", workspace_root=Path(self.temp.name), docker=self.fake)
        network = mock.patch("tokimon.tools.mcp.container_executor.deployment.host_networks", return_value=["10.1.1.1/32"])
        network.start()
        self.addCleanup(network.stop)
        self.deployment.init()

    def test_create_timeout_after_commit_recovers_exact_executor(self):
        run = self.fake.run
        def uncertain(*args, **kwargs):
            result = run(*args, **kwargs)
            if args[:2] == ("container", "create"):
                raise subprocess.TimeoutExpired("docker container create", 120)
            return result
        self.fake.run = uncertain
        installed = self.deployment.install()
        self.assertTrue(installed["installed"])
        self.assertEqual(len(self.fake.resources["container"]), 1)
        self.deployment.uninstall(purge_data=True)
        self.assertFalse(self.fake.resources["container"])

    def test_late_creation_blocks_purge_until_exact_container_is_visible(self):
        run, inspect = self.fake.run, self.fake.inspect
        visible = False
        def uncertain(*args, **kwargs):
            result = run(*args, **kwargs)
            if args[:2] == ("container", "create"):
                raise subprocess.TimeoutExpired("docker container create", 120)
            return result
        def delayed(kind, identity):
            if kind == "container" and not visible:
                raise MissingResource("not visible yet")
            return inspect(kind, identity)
        self.fake.run, self.fake.inspect = uncertain, delayed
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            self.deployment.install()
        before = len(self.fake.calls)
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            self.deployment.uninstall(purge_data=True)
        self.assertFalse(any(args[1] == "rm" for args in self.fake.calls[before:]))
        visible = True
        self.deployment.uninstall(purge_data=True)
        self.assertFalse(any(self.fake.resources[kind] for kind in ("container", "volume", "network")))

    def test_create_timeout_never_adopts_matching_name_without_creation_token(self):
        run = self.fake.run
        def uncertain(*args, **kwargs):
            result = run(*args, **kwargs)
            if args[:2] == ("container", "create"):
                del self.fake.resources["container"][result]["Config"]["Labels"]["tokimon.creation"]
                raise subprocess.TimeoutExpired("docker container create", 120)
            return result
        self.fake.run = uncertain
        with self.assertRaisesRegex(RuntimeError, "identity/token mismatch"):
            self.deployment.install()
        with self.assertRaisesRegex(RuntimeError, "identity/token mismatch"):
            self.deployment.uninstall(purge_data=True)
        self.assertFalse(any(args[1] == "rm" for args in self.fake.calls if len(args) > 1))
        self.assertEqual(len(self.fake.resources["container"]), 1)

    def test_nonzero_create_preserves_pending_identity_until_late_visibility(self):
        run, inspect = self.fake.run, self.fake.inspect
        visible = False
        def uncertain(*args, **kwargs):
            result = run(*args, **kwargs)
            if args[:2] == ("container", "create"):
                raise DockerCommandError("connection reset after daemon commit")
            return result
        def delayed(kind, identity):
            if kind == "container" and not visible:
                raise MissingResource("not visible yet")
            return inspect(kind, identity)
        self.fake.run, self.fake.inspect = uncertain, delayed
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            self.deployment.install()
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            self.deployment.uninstall(purge_data=True)
        self.assertFalse(any(args[1] == "rm" for args in self.fake.calls if len(args) > 1))
        visible = True
        self.deployment.uninstall(purge_data=True)
        self.assertFalse(any(self.fake.resources[kind] for kind in ("container", "volume", "network")))

    def test_install_update_rollback_and_data_retaining_uninstall(self):
        first = self.deployment.install()
        self.assertTrue(first["installed"])
        volumes = copy.deepcopy(self.fake.resources["volume"])
        config = json.loads(self.deployment.config_path.read_text())
        config["limits"]["memory_mb"] = 1024
        self.deployment.config_path.write_text(json.dumps(config))
        second = self.deployment.update()
        self.assertNotEqual(first["current_revision"], second["current_revision"])
        self.assertEqual(second["previous_revision"], first["current_revision"])
        restored = self.deployment.rollback()
        self.assertEqual(restored["current_revision"], first["current_revision"])
        self.assertEqual(self.fake.resources["volume"], volumes)
        self.assertTrue(self.deployment.uninstall()["data_retained"])
        self.assertEqual(self.fake.resources["volume"], volumes)
        self.assertFalse(self.fake.resources["container"])
        self.assertFalse(self.fake.resources["network"])
        self.deployment.uninstall(purge_data=True)
        self.assertFalse(self.fake.resources["volume"])

    def test_failed_update_restores_old_revision_and_refuses_active_jobs(self):
        first = self.deployment.install()
        self.fake.active_jobs = 1
        with self.assertRaisesRegex(ValueError, "active jobs"):
            self.deployment.update()
        self.fake.active_jobs = 0
        self.fake.fail_start = True
        with self.assertRaisesRegex(RuntimeError, "previous deployment restored"):
            self.deployment.update()
        status = self.deployment.status()
        self.assertEqual(status["current_revision"], first["current_revision"])
        self.assertTrue(all(c["running"] for c in status["containers"].values()))
        receipts = list((self.deployment.path / "failures").glob("*.json"))
        self.assertTrue(receipts)
        self.assertIn("injected start failure", receipts[0].read_text())

    def test_job_admitted_during_build_blocks_switch_without_removing_old_executor(self):
        first = self.deployment.install()
        self.fake.on_build = lambda: setattr(self.fake, "active_jobs", 1)
        before = len(self.fake.calls)
        with self.assertRaisesRegex(ValueError, "active jobs"):
            self.deployment.update()
        self.assertFalse(any(args[:2] == ("container", "rm") for args in self.fake.calls[before:]))
        self.assertEqual(self.deployment.status()["containers"], first["containers"])
        self.assertFalse(self.fake.paused)

    def test_stopped_deployment_updates_and_preserves_volumes(self):
        first = self.deployment.install()
        volumes = copy.deepcopy(self.fake.resources["volume"])
        self.deployment.stop()
        config = json.loads(self.deployment.config_path.read_text())
        config["limits"]["memory_mb"] = 1024
        self.deployment.config_path.write_text(json.dumps(config))
        updated = self.deployment.update()
        self.assertNotEqual(updated["current_revision"], first["current_revision"])
        self.assertEqual(updated["previous_revision"], first["current_revision"])
        self.assertTrue(all(c["running"] for c in updated["containers"].values()))
        self.assertEqual(self.fake.resources["volume"], volumes)

    def test_stopped_deployment_rolls_back_and_preserves_volumes(self):
        first = self.deployment.install()
        config = json.loads(self.deployment.config_path.read_text())
        config["limits"]["memory_mb"] = 1024
        self.deployment.config_path.write_text(json.dumps(config))
        second = self.deployment.update()
        volumes = copy.deepcopy(self.fake.resources["volume"])
        self.deployment.stop()
        restored = self.deployment.rollback()
        self.assertEqual(restored["current_revision"], first["current_revision"])
        self.assertEqual(restored["previous_revision"], second["current_revision"])
        self.assertTrue(all(c["running"] for c in restored["containers"].values()))
        self.assertEqual(self.fake.resources["volume"], volumes)

    def test_connector_uses_configured_worker_identity_and_one_container(self):
        config = json.loads(self.deployment.config_path.read_text())
        config.update(uid=2001, gid=2002)
        self.deployment.config_path.write_text(json.dumps(config))
        status = self.deployment.install()
        health = json.loads(self.deployment.request("tools/call", {"name": "executor_health"})["result"]["content"][0]["text"])
        self.assertEqual(health["identity"]["Uid"].split(), ["2001"] * 4)
        self.assertEqual(health["identity"]["Gid"].split(), ["2002"] * 4)
        self.assertEqual(set(status["containers"]), {"executor"})
        execs = [args for args in self.fake.calls if args[:2] == ("container", "exec")]
        self.assertTrue(all(args[4] == status["containers"]["executor"]["id"] and "connect" in args for args in execs))

    def test_identity_and_labels_gate_every_destructive_mutation(self):
        self.deployment.install()
        record = json.loads(self.deployment.record_path.read_text())
        self.fake.resources["container"][record["containers"]["executor"]]["Config"]["Labels"]["tokimon.deployment"] = "someone-else"
        before = len(self.fake.calls)
        self.assertFalse(self.deployment.status()["ok"])
        for command in (self.deployment.stop, self.deployment.start, self.deployment.uninstall, self.deployment.update):
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                command()
        self.assertEqual(len(self.fake.calls), before)

    def test_manifest_and_runtime_never_grant_host_shell_or_docker_control(self):
        self.deployment.install()
        creates = [args for args in self.fake.calls if args[:2] == ("container", "create")]
        self.assertEqual(len(creates), 1)
        executor = creates[0]
        self.assertNotIn("--privileged", executor)
        self.assertNotIn("--publish", executor)
        self.assertNotIn("--init", executor)
        self.assertNotIn("/var/run/docker.sock", " ".join(executor))
        execs = [args for args in self.fake.calls if args[:2] == ("container", "exec")]
        self.assertTrue(execs)
        record = json.loads(self.deployment.record_path.read_text())
        self.assertTrue(all(args[4] == record["containers"]["executor"] for args in execs))
        manifest = json.loads((self.deployment.path / "revisions" / record["current"]["revision"] / "compose.json").read_text())
        self.assertEqual(set(manifest["services"]), {"executor"})
        bundle = next((self.deployment.path / "revisions").glob("*/bundle"))
        self.assertFalse(any(p.name == ".tokimon" or p.name == ".venv" for p in bundle.rglob("*")))

    def test_worker_tunnel_configuration_is_rejected_before_docker_mutation(self):
        key = self.deployment.path / "private-key"
        key.write_text("key-content-canary-must-not-be-bundled")
        key.chmod(0o600)
        binary = self.deployment.path / "client"
        binary.write_bytes(b"client-canary")
        binary.chmod(0o755)
        config = json.loads(self.deployment.config_path.read_text())
        config["tunnel"] = {"tunnel_id": "tunnel_canary", "api_key_file": str(key), "client_binary": str(binary)}
        self.deployment.config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "workspace setup"):
            self.deployment.install()
        self.assertFalse(self.fake.calls)

    def test_retired_revision_cannot_be_started_or_rolled_back_over_a_healthy_worker(self):
        first = self.deployment.install()
        record = self.deployment._record()
        old = copy.deepcopy(record["current"])
        old.pop("worker_layout")
        record["previous"] = old
        self.deployment._save(record, "legacy_fixture")
        before = len(self.fake.calls)
        with self.assertRaisesRegex(ValueError, "retired paired"):
            self.deployment.rollback()
        self.assertFalse(any(c[:2] in (("container", "rm"), ("container", "stop")) for c in self.fake.calls[before:]))
        self.assertEqual(self.deployment.status()["containers"], first["containers"])
        record["current"] = old
        self.deployment._save(record, "legacy_fixture")
        with self.assertRaisesRegex(ValueError, "retired paired"):
            self.deployment.start()
