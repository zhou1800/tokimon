from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokimon.tools.mcp.container_executor.config import default_config, validate_config
from tokimon.tools.mcp.container_executor.deployment import Docker


class ConfigTests(unittest.TestCase):
    def test_default_no_host_grants_and_strict_bounds(self):
        config = default_config("test")
        self.assertEqual(validate_config(config, runtime_root=Path("/example/.tokimon"), name="test")["mounts"], [])
        bad = []
        for key, value in (("uid", 0), ("uid", True), ("schema_version", True), ("entrypoint", "bash"), ("name", "../bad")):
            item = copy.deepcopy(config)
            item[key] = value
            bad.append(item)
        for key, value in (("cpus", float("nan")), ("pids", 0), ("concurrency", 33), ("timeout_seconds", True)):
            item = copy.deepcopy(config)
            item["limits"][key] = value
            bad.append(item)
        for item in bad:
            with self.subTest(config=item), self.assertRaises(ValueError):
                validate_config(item, runtime_root=Path("/example/.tokimon"), name="test")

    def test_narrow_mounts_and_gateway_credentials_do_not_overlap(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[4], prefix=".executor-test-") as temporary:
            root = Path(temporary)
            allowed = root / "allowed"
            allowed.mkdir()
            state = root / ".tokimon"
            state.mkdir()
            key = state / "key"
            key.write_text("secret-never-copied")
            key.chmod(0o600)
            binary = state / "client"
            binary.write_text("binary")
            binary.chmod(0o755)
            config = default_config("test")
            config["mounts"] = [{"source": str(allowed), "alias": "data", "read_only": True}]
            config["tunnel"] = {"tunnel_id": "tunnel_example", "api_key_file": str(key), "client_binary": str(binary)}
            self.assertEqual(validate_config(config, runtime_root=state, name="test", gateway=True)["mounts"][0]["alias"], "data")
            for source in (str(root), "/", "/run", "/var/lib", str(state)):
                changed = copy.deepcopy(config)
                changed["mounts"][0]["source"] = source
                with self.assertRaises(ValueError):
                    validate_config(changed, runtime_root=state, name="test", gateway=True)
            link = root / "link"
            link.symlink_to(allowed, target_is_directory=True)
            config["mounts"][0]["source"] = str(link)
            with self.assertRaises(ValueError):
                validate_config(config, runtime_root=state, name="test", gateway=True)
            config["mounts"][0]["source"] = str(allowed)
            key.chmod(0o644)
            with self.assertRaises(ValueError):
                validate_config(config, runtime_root=state, name="test", gateway=True)

    def test_remote_rootless_and_desktop_docker_rejected(self):
        with mock.patch.dict("os.environ", {"DOCKER_HOST": "tcp://remote:2375"}), mock.patch("shutil.which", return_value="/usr/bin/tool"):
            with self.assertRaisesRegex(RuntimeError, "remote"):
                Docker().prerequisites()
        for info in ({"OSType": "linux", "SecurityOptions": ["name=rootless"]}, {"OSType": "linux", "OperatingSystem": "Docker Desktop"}):
            with mock.patch.dict("os.environ", {"DOCKER_HOST": "unix:///var/run/docker.sock", "DOCKER_CONTEXT": ""}), mock.patch("shutil.which", return_value="/usr/bin/tool"), mock.patch.object(Docker, "run", return_value=json.dumps(info)):
                with self.assertRaisesRegex(RuntimeError, "rootful"):
                    Docker().prerequisites()

    def test_context_overrides_docker_host_and_remote_context_is_rejected(self):
        context = [{"Endpoints": {"docker": {"Host": "ssh://remote-host"}}}]
        with mock.patch.dict("os.environ", {"DOCKER_HOST": "unix:///var/run/docker.sock", "DOCKER_CONTEXT": "remote"}), mock.patch("shutil.which", return_value="/usr/bin/tool"), mock.patch.object(Docker, "run", return_value=json.dumps(context)):
            with self.assertRaisesRegex(RuntimeError, "remote"):
                Docker().prerequisites()
