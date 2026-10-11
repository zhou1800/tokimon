"""Locked deployment lifecycle with exact Docker ownership and reversible revisions."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import time
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

from tokimon import runtime

from . import CAPABILITY_ID, VERSION
from .config import default_config, load_config, valid_name, validate_config, protected_masks
from .image import build_image
from .isolation import verify_unprivileged
from .jobs import atomic_json
from .ownership import LABEL_CAPABILITY, LABEL_DEPLOYMENT, LABEL_ROLE, ContainerCreation, DockerCommandError, MissingResource, labels, label_args, owned

MODULE = "tokimon.tools.mcp.container_executor"
BOOTSTRAP_CAPS = ("NET_ADMIN", "SETUID", "SETGID", "SETPCAP")


class Docker:
    def run(self, *args: str, timeout: int = 30) -> str:
        result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            detail = result.stderr[-4096:].strip()
            for argument in args:
                match = re.search(r"src=([^,]+),dst=/run/secrets/tunnel-key", argument)
                if match:
                    detail = detail.replace(match[1], "<tunnel-key-file>")
            raise DockerCommandError(f"docker {args[0]} failed (exit {result.returncode}): {detail}")
        return (result.stdout + result.stderr if args[:2] == ("container", "logs") else result.stdout).strip()

    def prerequisites(self) -> dict:
        if platform.system() != "Linux" or not shutil.which("docker") or not shutil.which("ip"):
            raise RuntimeError("local Linux Docker and iproute2 are required")
        endpoint = os.environ.get("DOCKER_HOST") if not os.environ.get("DOCKER_CONTEXT") else None
        if not endpoint:
            context = json.loads(self.run("context", "inspect"))[0]
            endpoint = context["Endpoints"]["docker"]["Host"]
        if not endpoint.startswith("unix://"):
            raise RuntimeError("remote Docker endpoints are unsupported")
        info = json.loads(self.run("info", "--format", "{{json .}}"))
        if info.get("OSType") != "linux" or "desktop" in info.get("OperatingSystem", "").lower() or any("rootless" in s for s in info.get("SecurityOptions", [])):
            raise RuntimeError("local rootful Linux Docker is required")
        return {"docker_version": info.get("ServerVersion"), "endpoint_kind": "local-unix", "rootful_linux": True}

    def inspect(self, kind: str, identity: str) -> dict:
        try:
            return json.loads(self.run(kind, "inspect", identity))[0]
        except DockerCommandError as exc:
            detail = str(exc).lower()
            if f"no such {kind}" in detail or "no such object:" in detail or kind == "network" and f"network {identity} not found" in detail:
                raise MissingResource(str(exc)) from exc
            raise


def runtime_root(workspace_root: Path | None = None) -> Path:
    return runtime.resolve_runtime_path(Path(".tokimon/tools/mcp/container-executor"), workspace_root=workspace_root or runtime.source_root())


def host_networks() -> list[str]:
    import ipaddress
    addresses = json.loads(subprocess.run(["ip", "-j", "address", "show"], capture_output=True, text=True, check=True, timeout=10).stdout)
    routes = json.loads(subprocess.run(["ip", "-j", "-4", "route", "show", "table", "all"], capture_output=True, text=True, check=True, timeout=10).stdout)
    networks = set()
    for interface in addresses:
        for info in interface.get("addr_info", []):
            address = ipaddress.ip_address(info["local"])
            networks.add(str(ipaddress.ip_network(f"{address}/{address.max_prefixlen}")))
            if info.get("scope") != "host":
                networks.add(str(ipaddress.ip_network(f"{address}/{info['prefixlen']}", strict=False)))
    for route in routes:
        destination = route.get("dst", "default")
        if destination != "default" and route.get("scope") == "link":
            networks.add(str(ipaddress.ip_network(destination, strict=False)))
    return sorted(networks)


class Deployment:
    def __init__(self, name: str = "default", *, workspace_root: Path | None = None, docker: Docker | None = None):
        self.name = valid_name(name)
        self.root = runtime_root(workspace_root)
        self.path = self.root / "deployments" / self.name
        self.docker = docker or Docker()
        self.record_path = self.path / "deployment.json"
        self.config_path = self.path / "config.json"

    @contextmanager
    def locked(self):
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.path / "lifecycle.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _record(self) -> dict:
        if not self.record_path.is_file():
            raise ValueError("deployment is not installed; run init then install")
        return json.loads(self.record_path.read_text())

    def _save(self, record: dict, event: str) -> None:
        atomic_json(self.record_path, record)
        with (self.path / "lifecycle.jsonl").open("a") as journal:
            journal.write(json.dumps({"event": event, "deployment": self.name, "correlation_id": record["owner_id"], "time": time.time()}) + "\n")

    def init(self) -> dict:
        with self.locked():
            with self.config_path.open("x") as output:
                json.dump(default_config(self.name), output, indent=2)
                output.write("\n")
            self.config_path.chmod(0o600)
        return {"ok": True, "config_path": str(self.config_path)}

    def _labels(self, record: dict, role: str) -> dict:
        return labels(record["owner_id"], role)

    def _owned(self, record: dict, kind: str, role: str, identity: str) -> dict:
        return owned(self.docker, record["owner_id"], kind, role, identity)

    def _verify_all(self, record: dict) -> None:
        for role, identity in record["containers"].items():
            self._owned(record, "container", role, identity)
        for role, identity in record["volumes"].items():
            self._owned(record, "volume", role, identity)
        if record.get("network"):
            self._owned(record, "network", "network", record["network"])

    def _label_args(self, record: dict, role: str) -> list[str]:
        return label_args(record["owner_id"], role)

    def _mask_options(self, config: dict) -> tuple[list[str], list[dict], list[str]]:
        options, files, directories = [], [], []
        for grant in config["mounts"]:
            for mask in protected_masks(grant, self.root, config.get("protected_host_paths", [])):
                if mask["directory"]:
                    spec = mask["target"] + ":ro,noexec,nosuid,nodev,size=1m,mode=000"
                    options.extend(["--tmpfs", spec])
                    directories.append(spec)
                else:
                    empty = self.path / "mask-empty"
                    if not empty.exists():
                        fd = os.open(empty, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0)
                        os.close(fd)
                    info = empty.lstat()
                    if not stat.S_ISREG(info.st_mode) or info.st_size or stat.S_IMODE(info.st_mode):
                        raise ValueError("invalid owned file-mask overlay")
                    spec = {"type": "bind", "source": str(empty), "target": mask["target"], "read_only": True}
                    files.append(spec)
                    options.extend(["--mount", f"type=bind,src={empty},dst={mask['target']},readonly"])
        return options, files, directories

    def _resources(self, record: dict) -> None:
        prefix = record["prefix"]
        for role in ("workspace", "state", "socket"):
            if role not in record["volumes"]:
                name = f"{prefix}-{role}"
                # A retained or unrelated name is never adopted, even with matching labels.
                names = self.docker.run("volume", "ls", "--format", "{{.Name}}").splitlines()
                if name in names:
                    raise RuntimeError("volume name already exists outside recorded ownership")
                identity = self.docker.run("volume", "create", *self._label_args(record, role), name)
                record["volumes"][role] = identity
                self._save(record, "volume_created")
        if not record.get("network"):
            record["network"] = self.docker.run("network", "create", *self._label_args(record, "network"), f"{prefix}-network")
            self._save(record, "network_created")

    def _build(self, config: dict) -> dict:
        return build_image(self.path, config, self.docker)

    def _require_layout(self, revision: dict | None) -> None:
        if revision is None:
            raise ValueError("deployment is not installed")
        if revision.get("worker_layout") != 2:
            raise ValueError("retired paired revision: stop, uninstall without purge, then install; directory services use explicit stop/start")

    def _creation(self, record):
        return ContainerCreation(self.path, self.docker, record["owner_id"], "executor")

    def _reconcile(self, record):
        identity = self._creation(record).resolve()
        if identity:
            if record["containers"].get("executor") not in (None, identity):
                raise RuntimeError("executor creation conflicts with recorded identity; refusing mutation")
            if "executor" not in record["containers"]:
                record["containers"]["executor"] = identity
                self._save(record, "executor_creation_reconciled")

    def _remove_containers(self, record: dict) -> None:
        self._reconcile(record)
        self._verify_all(record)
        for role in ("transport", "executor"):
            identity = record["containers"].get(role)
            if identity:
                self._owned(record, "container", role, identity)
                self.docker.run("container", "rm", "--force", identity)
                del record["containers"][role]
                self._save(record, "container_removed")
                if role == "executor":
                    self._creation(record).clear()

    def _start_revision(self, record: dict, revision: dict) -> None:
        self._verify_all(record)
        config = validate_config(revision["config"], runtime_root=self.root, name=self.name)
        self._require_layout(revision)
        mask_options, _, _ = self._mask_options(config)
        policy = {"blocked_networks": sorted(set(host_networks() + config["blocked_networks"])), "ipv6": "denied", "public_ipv4": "all ordinary protocols"}
        policy_path = self.path / "network-policy.json"
        atomic_json(policy_path, policy)
        common = ["--network", record["network"], "--read-only", "--cap-drop", "ALL", "--restart", "unless-stopped",
                  "--cpus", str(config["limits"]["cpus"]), "--memory", f"{config['limits']['memory_mb']}m",
                  "--memory-swap", f"{config['limits']['memory_mb']}m", "--pids-limit", str(config["limits"]["pids"]),
                  "--log-opt", "max-size=5m", "--log-opt", "max-file=3", "--dns", "1.1.1.1", "--dns", "8.8.8.8",
                  "--sysctl", "net.ipv6.conf.all.disable_ipv6=1", "--sysctl", "net.ipv6.conf.default.disable_ipv6=1",
                  "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777"]
        executor = ["container", "create", "--name", record["prefix"] + "-executor", *common, *self._label_args(record, "executor")]
        for cap in BOOTSTRAP_CAPS:
            executor.extend(["--cap-add", cap])
        executor.extend(["--mount", f"type=bind,src={policy_path},dst=/etc/tokimon/policy.json,readonly"])
        for role, target in (("workspace", "/workspace"), ("state", "/state"), ("socket", "/run/tokimon")):
            executor.extend(["--mount", f"type=volume,src={record['volumes'][role]},dst={target}"])
        for mount in config["mounts"]:
            grant = f"type=bind,src={mount['source']},dst=/mounts/{mount['alias']}"
            executor.extend(["--mount", grant + (",readonly" if mount["read_only"] else "")])
        executor.extend(mask_options)
        executor.append(revision["image"])
        record["containers"]["executor"] = self._creation(record).create(executor, record["prefix"] + "-executor", revision["image"])
        self._save(record, "executor_created")
        self._manifest(record, revision, policy_path)
        for identity in record["containers"].values():
            self.docker.run("container", "start", identity)
        error = None
        for _ in range(30):
            try:
                health = self._request(record, "tools/call", {"name": "executor_health"})
                payload = json.loads(health["result"]["content"][0]["text"])
                verify_unprivileged(payload["identity"], config["uid"], config["gid"])
                if payload["deployment"] != self.name or payload["isolation"] != policy:
                    raise RuntimeError("MCP deployment/policy identity mismatch")
                return
            except (RuntimeError, KeyError, ValueError) as exc:
                error = exc
                time.sleep(0.2)
        raise RuntimeError(f"deployment startup failed isolation/MCP checks: {error}")

    def _manifest(self, record: dict, revision: dict, policy_path: Path) -> None:
        config = revision["config"]
        base = {
            "image": revision["image"], "read_only": True, "cap_drop": ["ALL"], "restart": "unless-stopped",
            "cpus": config["limits"]["cpus"], "mem_limit": f"{config['limits']['memory_mb']}m",
            "memswap_limit": f"{config['limits']['memory_mb']}m", "pids_limit": config["limits"]["pids"],
            "dns": ["1.1.1.1", "8.8.8.8"], "networks": ["executor-network"],
            "sysctls": {"net.ipv6.conf.all.disable_ipv6": "1", "net.ipv6.conf.default.disable_ipv6": "1"},
            "tmpfs": ["/tmp:rw,noexec,nosuid,size=64m,mode=1777"],
            "logging": {"driver": "json-file", "options": {"max-size": "5m", "max-file": "3"}},
        }
        executor = {**base, "container_name": record["prefix"] + "-executor", "cap_add": list(BOOTSTRAP_CAPS),
                    "labels": self._labels(record, "executor"), "volumes": [
                        {"type": "volume", "source": role, "target": target}
                        for role, target in (("workspace", "/workspace"), ("state", "/state"), ("socket", "/run/tokimon"))]}
        executor["volumes"].append({"type": "bind", "source": str(policy_path), "target": "/etc/tokimon/policy.json", "read_only": True})
        executor["volumes"].extend({"type": "bind", "source": mount["source"], "target": "/mounts/" + mount["alias"], "read_only": mount["read_only"]} for mount in config["mounts"])
        _, mask_files, mask_dirs = self._mask_options(config)
        executor["volumes"].extend(mask_files)
        executor["tmpfs"] = base["tmpfs"] + mask_dirs
        manifest = {"services": {"executor": executor},
                    "volumes": {role: {"external": True, "name": identity} for role, identity in record["volumes"].items()},
                    "networks": {"executor-network": {"external": True, "name": self.docker.inspect("network", record["network"])["Name"]}}}
        atomic_json(self.path / "revisions" / revision["revision"] / "compose.json", manifest)

    def _request(self, record: dict, method: str, params: dict | None = None) -> dict:
        identity = record["containers"].get("executor")
        if not identity:
            raise ValueError("executor is not installed")
        item = self._owned(record, "container", "executor", identity)
        if not item["State"]["Running"]:
            raise ValueError("executor is stopped")
        return json.loads(self.docker.run("container", "exec", "--user", "0:0", identity,
                                         "/usr/local/bin/python", "-I", "-m", MODULE + ".isolation", "connect", "request", method,
                                         json.dumps(params or {}), record["owner_id"]))

    def _maintenance(self, record: dict, paused: bool) -> None:
        result = self._request(record, "tokimon/maintenance", {"paused": paused})
        if "error" in result:
            raise ValueError(result["error"]["message"])
        if result.get("result", {}).get("paused") is not paused:
            raise RuntimeError("executor admission fence was not acknowledged")

    def _failure(self, record: dict, error: Exception) -> Path:
        causes = []
        cause = error
        while cause is not None and len(causes) < 8:
            causes.append({"type": type(cause).__name__, "message": str(cause)[:4096]})
            cause = cause.__cause__
        containers = {}
        for role, identity in record["containers"].items():
            try:
                item = self._owned(record, "container", role, identity)
                state = item["State"]
                logs = self.docker.run("container", "logs", "--tail", "50", identity)
                startup = []
                for line in logs.splitlines():
                    if len(line) > 4096:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("message") == "startup_failed":
                        startup.append({k: str(event[k])[:128] for k in ("message", "stage", "error_type", "errno", "exit_code", "correlation_id") if k in event})
                containers[role] = {"id": identity, "status": state["Status"], "exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled"), "startup_failures": startup[-10:]}
            except (RuntimeError, OSError, subprocess.SubprocessError):
                containers[role] = {"id": identity, "diagnosis": "resource inspection unavailable"}
        directory = self.path / "failures"
        directory.mkdir(exist_ok=True)
        receipt = directory / (uuid.uuid4().hex + ".json")
        atomic_json(receipt, {"deployment": self.name, "correlation_id": record["owner_id"], "time": time.time(), "causes": causes, "containers": containers})
        return receipt

    def request(self, method: str, params: dict | None = None) -> dict:
        with self.locked():
            record = self._record()
            self._verify_all(record)
            self._require_layout(record["current"])
            return self._request(record, method, params)

    def _idle(self, record: dict) -> None:
        executor = record["containers"].get("executor")
        if executor and self._owned(record, "container", "executor", executor)["State"]["Running"]:
            response = self._request(record, "tools/call", {"name": "executor_health"})
            health = json.loads(response["result"]["content"][0]["text"])
            if health["active_jobs"]:
                raise ValueError("active jobs prevent update/rollback; wait, cancel, or explicitly stop")

    def install(self, config_path: Path | None = None) -> dict:
        with self.locked():
            config = load_config(config_path or self.config_path, runtime_root=self.root, name=self.name)
            self.docker.prerequisites()
            record = self._record() if self.record_path.exists() else {"owner_id": uuid.uuid4().hex, "containers": {}, "volumes": {}, "network": None, "current": None, "previous": None}
            if record["current"]:
                raise ValueError("deployment already installed; use update")
            record.setdefault("prefix", f"tokimon-exec-{self.name}-{record['owner_id'][:12]}")
            self._verify_all(record)
            try:
                revision = self._build(config)
            except Exception as exc:
                receipt = self._failure(record, exc)
                raise RuntimeError(f"image build failed; diagnosis: {receipt}") from exc
            self._save(record, "install_started")
            try:
                self._resources(record)
                self._remove_containers(record)
                self._start_revision(record, revision)
            except Exception as exc:
                receipt = self._failure(record, exc)
                self._remove_containers(record)
                self._save(record, "install_failed")
                raise RuntimeError(f"installation failed; diagnosis: {receipt}") from exc
            record["current"] = revision
            atomic_json(self.config_path, config)
            self._save(record, "installed")
        return self.status()

    def status(self) -> dict:
        if not self.record_path.exists():
            return {"ok": True, "installed": False, "deployment": self.name}
        record = self._record()
        containers = {}
        for role, identity in record["containers"].items():
            try:
                item = self._owned(record, "container", role, identity)
                containers[role] = {"id": identity, "status": item["State"]["Status"], "running": item["State"]["Running"], "owned": True}
            except RuntimeError:
                containers[role] = {"id": identity, "status": "unverified", "running": False, "owned": False}
        revision = record.get("current")
        return {"ok": all(c["owned"] for c in containers.values()), "installed": bool(revision), "deployment": self.name,
                "current_revision": revision["revision"] if revision else None,
                "previous_revision": record["previous"]["revision"] if record.get("previous") else None,
                "containers": containers, "layout": "executor-only" if revision and revision.get("worker_layout") == 2 else "paired",
                "remote_acceptance": "unverified"}

    def doctor(self) -> dict:
        checks = {"prerequisites": self.docker.prerequisites()}
        with self.locked():
            record = self._record()
            self._verify_all(record)
            if not record["current"]:
                raise ValueError("deployment has no installed revision")
            self._require_layout(record["current"])
            validate_config(record["current"]["config"], runtime_root=self.root, name=self.name)
            health = self._request(record, "tools/call", {"name": "executor_health"})
            payload = json.loads(health["result"]["content"][0]["text"])
            verify_unprivileged(payload["identity"], record["current"]["config"]["uid"], record["current"]["config"]["gid"])
            checks["executor"] = payload
        return {"ok": True, "deployment": self.name, "checks": checks, "remote_acceptance": "unverified"}

    def stop(self) -> dict:
        with self.locked():
            record = self._record()
            self._reconcile(record)
            self._verify_all(record)
            for role in ("transport", "executor"):
                if role in record["containers"]:
                    self.docker.run("container", "stop", "--time", "5", record["containers"][role])
            self._save(record, "stopped")
        return self.status()

    def start(self) -> dict:
        with self.locked():
            self.docker.prerequisites()
            record = self._record()
            self._reconcile(record)
            self._verify_all(record)
            if not record["current"]:
                raise ValueError("deployment is not installed")
            self._require_layout(record["current"])
            if record["containers"] and all(self._owned(record, "container", role, identity)["State"]["Running"] for role, identity in record["containers"].items()):
                return self.status()
            self._remove_containers(record)
            self._resources(record)
            try:
                self._start_revision(record, record["current"])
            except Exception as exc:
                receipt = self._failure(record, exc)
                raise RuntimeError(f"start failed; diagnosis: {receipt}") from exc
            self._save(record, "started")
        return self.status()

    def _switch(self, record: dict, revision: dict, event: str) -> None:
        previous = record["current"]
        original_containers = dict(record["containers"])
        executor = original_containers.get("executor")
        fenced = bool(executor and self._owned(record, "container", "executor", executor)["State"]["Running"])
        if fenced:
            self._maintenance(record, True)
        try:
            self._remove_containers(record)
            self._start_revision(record, revision)
        except Exception as failure:
            receipt = self._failure(record, failure)
            if record["containers"] == original_containers:
                if fenced:
                    try:
                        self._maintenance(record, False)
                    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError):
                        self._save(record, "admission_resume_failed")
                raise RuntimeError(f"switch refused before replacement; diagnosis: {receipt}") from failure
            try:
                self._remove_containers(record)
                self._start_revision(record, previous)
                self._save(record, "previous_restored")
            except Exception as restore:
                self._save(record, "restoration_failed")
                restore_receipt = self._failure(record, restore)
                raise RuntimeError(f"revision startup and previous restoration failed; diagnosis: {receipt}; restoration: {restore_receipt}") from failure
            raise RuntimeError(f"revision startup failed; previous deployment restored; diagnosis: {receipt}") from failure
        record.update(current=revision, previous=previous)
        atomic_json(self.config_path, revision["config"])
        self._save(record, event)

    def update(self, config_path: Path | None = None) -> dict:
        with self.locked():
            self.docker.prerequisites()
            record = self._record()
            self._verify_all(record)
            if not record["current"]:
                raise ValueError("deployment is not installed")
            self._require_layout(record["current"])
            self._idle(record)
            config = load_config(config_path or self.config_path, runtime_root=self.root, name=self.name)
            try:
                revision = self._build(config)
            except Exception as exc:
                receipt = self._failure(record, exc)
                raise RuntimeError(f"image build failed; diagnosis: {receipt}") from exc
            self._switch(record, revision, "updated")
        return self.status()

    def rollback(self) -> dict:
        with self.locked():
            self.docker.prerequisites()
            record = self._record()
            self._verify_all(record)
            if not record.get("previous"):
                raise ValueError("no successful previous revision")
            self._require_layout(record["current"])
            self._require_layout(record["previous"])
            self._idle(record)
            self._switch(record, record["previous"], "rolled_back")
        return self.status()

    def uninstall(self, *, purge_data: bool = False) -> dict:
        with self.locked():
            record = self._record()
            self._reconcile(record)
            self._verify_all(record)
            self._remove_containers(record)
            if record.get("network"):
                self._owned(record, "network", "network", record["network"])
                self.docker.run("network", "rm", record["network"])
                record["network"] = None
            if purge_data:
                for role, identity in list(record["volumes"].items()):
                    self._owned(record, "volume", role, identity)
                    self.docker.run("volume", "rm", identity)
                    del record["volumes"][role]
            record.update(previous=record["current"] or record.get("previous"), current=None)
            self._save(record, "uninstalled_purged" if purge_data else "uninstalled_data_retained")
        return {"ok": True, "deployment": self.name, "data_retained": not purge_data}

    def serve(self) -> int:
        with self.locked():
            record = self._record()
            self._verify_all(record)
            self._require_layout(record["current"])
            identity = record["containers"].get("executor")
            if not identity or not self._owned(record, "container", "executor", identity)["State"]["Running"]:
                raise ValueError("running executor required")
        return subprocess.call(["docker", "container", "exec", "--interactive", "--user", "0:0", identity,
                                "/usr/local/bin/python", "-I", "-m", MODULE + ".isolation", "connect", "relay"])


def health() -> dict:
    try:
        return {"ok": True, "capability_id": CAPABILITY_ID, "version": VERSION, "prerequisites": Docker().prerequisites()}
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"ok": False, "capability_id": CAPABILITY_ID, "error": str(exc)}


def download_tunnel_client(*, workspace_root: Path | None = None) -> dict:
    machine = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
    if platform.system() != "Linux" or machine is None:
        raise ValueError("official download supports Linux amd64/arm64")

    def fetch(url: str, maximum: int) -> bytes:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Tokimon-Container-Executor"}), timeout=30) as response:
            data = response.read(maximum + 1)
            if len(data) > maximum:
                raise ValueError("official release response exceeds limit")
            return data

    release = json.loads(fetch("https://api.github.com/repos/openai/tunnel-client/releases/latest", 1048576))
    tag = release["tag_name"]
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", tag):
        raise ValueError("invalid official release tag")
    archive_name = f"tunnel-client-{tag}-linux-{machine}.zip"
    assets = [a for a in release["assets"] if a["name"] == archive_name]
    if len(assets) != 1:
        raise ValueError("official release has no unique supported archive")
    asset = assets[0]
    url = asset["browser_download_url"]
    if not url.startswith("https://github.com/openai/tunnel-client/releases/download/"):
        raise ValueError("unexpected official archive location")
    archive = fetch(url, 100 * 1024 * 1024)
    digest = hashlib.sha256(archive).hexdigest()
    if asset.get("digest") != "sha256:" + digest:
        raise ValueError("official release SHA256 verification failed")
    with zipfile.ZipFile(io.BytesIO(archive)) as contents:
        entries = [e for e in contents.infolist() if Path(e.filename).name == "tunnel-client" and not e.is_dir()]
        if len(entries) != 1 or entries[0].file_size > 100 * 1024 * 1024:
            raise ValueError("invalid official client archive")
        binary = contents.read(entries[0])
    directory = runtime_root(workspace_root) / "downloads" / tag
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "tunnel-client"
    target.write_bytes(binary)
    target.chmod(0o755)
    receipt = {"version": release["tag_name"], "release_url": release["html_url"], "archive_sha256": digest,
               "binary_sha256": hashlib.sha256(binary).hexdigest(), "client_binary": str(target)}
    atomic_json(directory / "receipt.json", receipt)
    return {"ok": True, **receipt}
