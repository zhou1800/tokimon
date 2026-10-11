"""Reversible machine gateway publication with exact resource ownership."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid

from .config import default_config, fields, integer, validate_config
from .gateway import validate_routes
from .image import build_image
from .isolation import verify_unprivileged
from .ownership import ContainerCreation, label_args, owned
from .state import StateStore

MODULE = "tokimon.tools.mcp.container_executor.gateway"


class GatewayDeployment:
    """Caller holds the machine controller lock throughout every mutation."""

    def __init__(self, root, docker):
        self.store = StateStore(root / "gateway")
        self.docker = docker
        self.runtime_root = root.parent

    def _record(self):
        record = self.store.read("gateway.json")
        if record is None:
            return None
        fields(record, {"owner_id", "prefix", "uid", "gid", "container", "network", "current", "previous"})
        if not isinstance(record["owner_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", record["owner_id"]) or record["prefix"] != "tokimon-gateway-" + record["owner_id"][:12]:
            raise ValueError("invalid gateway ownership record")
        for name in ("uid", "gid"):
            integer(record[name], name, 1, 2147483647)
        for name in ("container", "network"):
            value = record[name]
            if value is not None and (not isinstance(value, str) or not value or len(value) > 128):
                raise ValueError("invalid gateway resource identity")
        for revision in (record["current"], record["previous"]):
            if revision is None:
                continue
            fields(revision, {"revision", "image", "config", "gateway"})
            if not isinstance(revision["revision"], str) or not re.fullmatch(r"[0-9a-f]{64}", revision["revision"]):
                raise ValueError("invalid gateway revision")
            if not isinstance(revision["image"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", revision["image"]):
                raise ValueError("invalid gateway image identity")
            fields(revision["config"], set(default_config("gateway")))
            if revision["config"]["name"] != "gateway" or revision["config"]["mounts"]:
                raise ValueError("gateway revisions cannot grant host workspace mounts")
            config = validate_config(revision["config"], runtime_root=self.runtime_root, name="gateway", require_tunnel_files=False, gateway=True)
            if (config["uid"], config["gid"]) != (record["uid"], record["gid"]):
                raise ValueError("gateway revision operator identity mismatch")
            validate_routes(revision["gateway"])
        return record

    def _verify(self, record):
        if record.get("network"):
            owned(self.docker, record["owner_id"], "network", "gateway-network", record["network"])
        if record.get("container"):
            return owned(self.docker, record["owner_id"], "container", "gateway", record["container"])
        return None

    def _creation(self, record):
        return ContainerCreation(self.store.root, self.docker, record["owner_id"], "gateway")

    def _reconcile(self, record):
        identity = self._creation(record).resolve()
        if identity:
            if record.get("container") not in (None, identity):
                raise RuntimeError("gateway creation conflicts with recorded identity; refusing mutation")
            if record.get("container") is None:
                record["container"] = identity
                self.store.write("gateway.json", record)

    def _probe(self, record):
        probe = json.loads(self.docker.run("container", "exec", "--user", f"{record['uid']}:{record['gid']}", record["container"],
                                          "/usr/local/bin/python", "-I", "-m", MODULE, "probe", timeout=10))
        identity = dict(probe["identity"])
        # Docker may repeat the primary GID; this grants no additional group access.
        if identity["Groups"].split() == [str(record["gid"])]:
            identity["Groups"] = ""
        verify_unprivileged(identity, record["uid"], record["gid"])
        return probe

    def tools(self) -> list[dict]:
        record = self._record()
        if not record:
            raise ValueError("machine Gateway is not installed")
        item = self._verify(record)
        if not item or not item["State"]["Running"]:
            raise ValueError("running machine Gateway required")
        discovery = self._probe(record).get("discovery", {})
        tools = discovery.get("result", {}).get("tools")
        if not isinstance(tools, list) or not tools:
            raise ValueError("gateway MCP discovery failed; refresh directory services")
        return tools

    def status(self) -> dict:
        record = self._record()
        result = {"configured": False, "gateway_ready": False, "tunnel_ready": False, "workspaces": [], "remote_acceptance": "unverified"}
        if not record:
            return result
        result["container_id"] = record.get("container")
        revision = record.get("current")
        result["configured"] = bool(revision and revision["config"]["tunnel"])
        result["tunnel_id"] = revision["config"]["tunnel"]["tunnel_id"] if result["configured"] else None
        try:
            item = self._verify(record)
            result["running"] = bool(item and item["State"]["Running"])
            if result["running"] and revision:
                probe = self._probe(record)
                expected = sorted(revision["gateway"]["routes"])
                if probe["workspaces"] != expected:
                    raise RuntimeError("gateway routing identity mismatch")
                result.update(gateway_ready=True, tunnel_ready=probe["tunnel_ready"], workspaces=expected)
        except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
            result["error"] = str(exc)[:4096]
        return result

    def _remove(self, record):
        self._reconcile(record)
        self._verify(record)
        if record.get("container"):
            self.docker.run("container", "rm", "--force", record["container"])
            record["container"] = None
            self.store.write("gateway.json", record)
            self._creation(record).clear()

    def _start(self, record, revision):
        self._verify(record)
        routes = validate_routes(revision["gateway"])["routes"]
        config = validate_config(revision["config"], runtime_root=self.runtime_root, name="gateway", gateway=True)
        for route in routes.values():
            owned(self.docker, route["owner_id"], "volume", "socket", route["socket_volume"])
        if not record.get("network"):
            record["network"] = self.docker.run("network", "create", *label_args(record["owner_id"], "gateway-network"), record["prefix"] + "-network")
            self.store.write("gateway.json", record)
        args = ["container", "create", "--name", record["prefix"], "--network", record["network"],
                "--user", f"{record['uid']}:{record['gid']}", "--read-only", "--cap-drop", "ALL",
                "--restart", "unless-stopped", "--cpus", "1", "--memory", "256m", "--memory-swap", "256m", "--pids-limit", "128",
                "--log-opt", "max-size=5m", "--log-opt", "max-file=3", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
                "--workdir", "/tmp", *label_args(record["owner_id"], "gateway"), "--entrypoint", "/usr/local/bin/python"]
        for identity, route in sorted(routes.items()):
            args.extend(["--mount", f"type=volume,src={route['socket_volume']},dst=/run/workspaces/{identity},readonly"])
        if config["tunnel"]:
            args.extend(["--mount", f"type=bind,src={config['tunnel']['api_key_file']},dst=/run/secrets/tunnel-key,readonly"])
        args.extend([revision["image"], "-I", "-m", MODULE, "supervise"])
        record["container"] = self._creation(record).create(args, record["prefix"], revision["image"])
        self.store.write("gateway.json", record)
        self.docker.run("container", "start", record["container"])
        failure = None
        for _ in range(20):
            try:
                probe = self._probe(record)
                if probe["workspaces"] != sorted(routes):
                    raise RuntimeError("gateway route validation failed")
                return
            except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
                failure = exc
                time.sleep(0.1)
        raise RuntimeError(f"gateway startup failed: {failure}")

    def refresh(self, routes: dict, profile: dict | None) -> dict:
        route_config = validate_routes({"schema_version": 1, "routes": routes})
        record = self._record()
        if record:
            self._reconcile(record)
        if not routes:
            if record:
                item = self._verify(record)
                if item and item["State"]["Running"]:
                    self.docker.run("container", "stop", "--time", "5", record["container"])
            return self.status()
        self.docker.prerequisites()
        config = default_config("gateway")
        config.update(uid=os.geteuid() or 1000, gid=os.getegid() or 1000, tunnel=profile["tunnel"] if profile else None)
        config = validate_config(config, runtime_root=self.runtime_root, name="gateway", gateway=True)
        if record is None:
            owner = uuid.uuid4().hex
            record = {"owner_id": owner, "prefix": "tokimon-gateway-" + owner[:12], "uid": config["uid"], "gid": config["gid"],
                      "container": None, "network": None, "current": None, "previous": None}
            self.store.write("gateway.json", record)
        item = self._verify(record)
        previous = record["current"]
        if previous and previous["gateway"] == route_config and previous["config"] == config and item and item["State"]["Running"]:
            return self.status()
        revision = build_image(self.store.root, config, self.docker, gateway=route_config)
        self._remove(record)
        try:
            self._start(record, revision)
        except Exception as failure:
            receipt = uuid.uuid4().hex + ".json"
            diagnostics = {}
            if record.get("container"):
                diagnostics["container_id"] = record["container"]
                try:
                    item = self._verify(record)
                    diagnostics["state"] = item["State"]
                    diagnostics["logs"] = self.docker.run("container", "logs", "--tail", "40", record["container"])[-4096:]
                except Exception as diagnostic_error:
                    diagnostics["error_type"] = type(diagnostic_error).__name__
                    diagnostics["error"] = str(diagnostic_error)[:4096]
            receipt_status = str(self.store.root / receipt)
            try:
                self.store.write(receipt, {"correlation_id": record["owner_id"], "stage": "gateway_refresh", "error_type": type(failure).__name__,
                                           "error": str(failure)[:4096], "diagnostics": diagnostics})
            except Exception as receipt_error:
                receipt_status = f"unavailable ({type(receipt_error).__name__})"
            self._remove(record)
            if previous:
                try:
                    self._start(record, previous)
                except Exception as restore:
                    self.store.write("gateway.json", record)
                    raise RuntimeError(f"gateway refresh and restoration failed; receipt {receipt_status}; {type(restore).__name__}") from failure
            raise RuntimeError(f"gateway refresh failed; local workers preserved; receipt {receipt_status}") from failure
        record.update(current=revision, previous=previous)
        self.store.write("gateway.json", record)
        return self.status()

    def purge(self):
        record = self._record()
        if not record:
            return
        self._remove(record)
        if record.get("network"):
            owned(self.docker, record["owner_id"], "network", "gateway-network", record["network"])
            self.docker.run("network", "rm", record["network"])
            record["network"] = None
        record.update(current=None, previous=None)
        self.store.write("gateway.json", record)
