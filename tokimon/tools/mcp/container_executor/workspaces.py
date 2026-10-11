"""Current-directory authorization and machine-wide lifecycle transaction boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .config import TUNNEL_KEY_PLACEHOLDER, absolute_path, default_config, fields, validate_config
from .deployment import Deployment, Docker, download_tunnel_client, runtime_root
from .files import FileBoundary
from .gateway import WORKSPACE_ID
from .gateway_deployment import GatewayDeployment
from .jobs import atomic_json
from .isolation import verify_unprivileged
from .state import StateStore


def workspace_id(directory: Path) -> str:
    return "ws-" + hashlib.sha256(str(directory).encode()).hexdigest()[:20]


class WorkspaceController:
    def __init__(self, *, workspace_root: Path | None = None, docker: Docker | None = None):
        self.installation_root = workspace_root
        self.root = runtime_root(workspace_root)
        self.store = StateStore(self.root / "workspaces")
        self.docker = docker or Docker()
        self.gateway = GatewayDeployment(self.store.root, self.docker)

    def _deployment(self, identity: str) -> Deployment:
        return Deployment(identity, workspace_root=self.installation_root, docker=self.docker)

    def _registry(self) -> dict:
        data = fields(self.store.read("registry.json", {"schema_version": 1, "entries": {}}), {"schema_version", "entries"})
        if type(data["schema_version"]) is not int or data["schema_version"] != 1 or not isinstance(data["entries"], dict) or len(data["entries"]) > 256:
            raise ValueError("invalid or excessive workspace registry")
        for identity, entry in data["entries"].items():
            if not isinstance(identity, str) or not WORKSPACE_ID.fullmatch(identity):
                raise ValueError("invalid registry workspace_id")
            fields(entry, {"directory", "device", "inode"})
            if not isinstance(entry["directory"], str):
                raise ValueError("registry directory must be a string")
            path = Path(entry["directory"])
            if not path.is_absolute() or len(str(path).encode()) > 4096 or workspace_id(path) != identity:
                raise ValueError("registry directory identity mismatch")
            if any(type(entry[k]) is not int or entry[k] < 0 for k in ("device", "inode")):
                raise ValueError("invalid registry directory device/inode")
        return data

    def _profile(self, *, require_files=True):
        profile = self.store.read("profile.json")
        if profile is None:
            return None
        fields(profile, {"schema_version", "tunnel"})
        if type(profile["schema_version"]) is not int or profile["schema_version"] != 1 or profile["tunnel"] is None:
            raise ValueError("invalid machine connection profile")
        tunnel = fields(profile["tunnel"], {"tunnel_id", "api_key_file", "client_binary"})
        if not isinstance(tunnel["tunnel_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", tunnel["tunnel_id"]):
            raise ValueError("invalid machine tunnel ID")
        for name in ("api_key_file", "client_binary"):
            value = tunnel[name]
            if not isinstance(value, str) or len(value.encode()) > 4096 or not Path(value).is_absolute():
                raise ValueError("invalid machine file reference")
        if not require_files:
            return profile
        config = default_config("gateway")
        config["tunnel"] = profile["tunnel"]
        validate_config(config, runtime_root=self.root, name="gateway", gateway=True)
        return profile

    def setup_instructions(self) -> dict:
        return {"recommended_key_file": str(self.root / "secrets/tunnel.key"),
                "key_file_mode": "0600", "profile_path": str(self.store.root / "profile.json"),
                "steps": ["Create/select an OpenAI Tunnel associated with your ChatGPT workspace.",
                          "Replace the marker in the drafted private file with its matching Tunnels Read + Use runtime key as one line, or keep an existing key; never send its value in chat.",
                          "Run workspace setup with the Tunnel ID and private key-file path.",
                          "Start the directory service and shared Gateway before creating or testing the ChatGPT connection.",
                          "Create a private ChatGPT MCP connection: Connection = Tunnel, select the same ID once.",
                          "In ChatGPT call executor_workspaces and use explicit workspace_id."],
                "documentation": "https://developers.openai.com/api/docs/guides/secure-mcp-tunnels",
                "remote_acceptance": "unverified"}

    def _draft_key_file(self) -> bool:
        files = FileBoundary({str(self.root): False})
        with files.directory(str(self.root)) as fd:
            try:
                os.mkdir("secrets", mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
        path = str(self.root / "secrets/tunnel.key")
        try:
            with files.open_regular(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, write=True) as fd:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb", closefd=False) as output:
                    output.write((TUNNEL_KEY_PLACEHOLDER + "\n").encode())
                    output.flush()
                    os.fsync(fd)
        except FileExistsError:
            with files.open_regular(path, os.O_RDONLY):
                pass
            return False
        return True

    def _entry(self, registry: dict, directory: Path | None, identity: str | None, *, create=False):
        if identity is not None:
            if directory is not None or not WORKSPACE_ID.fullmatch(identity):
                raise ValueError("select directory or a valid workspace_id")
            if identity not in registry["entries"]:
                raise ValueError("workspace is not registered")
            return identity, registry["entries"][identity]
        path = absolute_path(str(directory or Path.cwd()), directory=True)
        identity = workspace_id(path)
        info = path.stat()
        entry = {"directory": str(path), "device": info.st_dev, "inode": info.st_ino}
        existing = registry["entries"].get(identity)
        if existing is not None and existing != entry:
            raise ValueError("directory path was replaced or workspace hash collided; select the old workspace_id to retire it")
        if existing is None and not create:
            raise ValueError("current directory has no registered workspace service")
        if existing is None and len(registry["entries"]) >= 256:
            raise ValueError("workspace registry reached 256 directories; retire an entry")
        return identity, entry

    def _summary(self, identity, entry):
        deployment = self._deployment(identity)
        status = deployment.status()
        containers = status.get("containers", {})
        running = containers.get("executor", {}).get("running", False)
        ready = False
        error = None
        if running and status.get("layout") != "executor-only":
            error = "older paired service requires explicit stop/start migration"
        elif running and status["ok"]:
            try:
                response = deployment.request("tools/call", {"name": "executor_health"})
                tool = response["result"]
                payload = json.loads(tool["content"][0]["text"])
                config = deployment._record()["current"]["config"]
                verify_unprivileged(payload["identity"], config["uid"], config["gid"])
                ready = not tool.get("isError", False) and payload["deployment"] == identity
            except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
                error = str(exc)[:4096]
        result = {"workspace_id": identity, "directory": entry["directory"], "project_root": "/mounts/project", "running": running,
                  "local_ready": ready, "installed": status["installed"], "containers": containers, "remote_acceptance": "unverified"}
        if error:
            result["error"] = error
        return result

    def _routes(self, registry):
        routes = {}
        for identity, entry in registry["entries"].items():
            summary = self._summary(identity, entry)
            if summary["local_ready"]:
                record = self._deployment(identity)._record()
                routes[identity] = {"directory": entry["directory"], "socket_volume": record["volumes"]["socket"], "owner_id": record["owner_id"]}
        return routes

    def _refresh(self, registry, profile):
        try:
            return self.gateway.refresh(self._routes(registry), profile), None
        except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
            return self.gateway.status(), str(exc)[:4096]

    def setup(self, tunnel_id: str, key_file: Path, client_binary: Path | None = None) -> dict:
        with self.store.locked():
            registry = self._registry()
            if any(self._summary(i, e)["running"] for i, e in registry["entries"].items()):
                raise ValueError("stop directory services before replacing machine setup")
            # Validate the credential reference before downloading anything.
            provisional = default_config("gateway")
            provisional["tunnel"] = {"tunnel_id": tunnel_id, "api_key_file": str(key_file), "client_binary": str(client_binary or Path(sys.executable).resolve())}
            validate_config(provisional, runtime_root=self.root, name="gateway", gateway=True)
            if client_binary is None:
                client_binary = Path(download_tunnel_client(workspace_root=self.installation_root)["client_binary"])
            provisional["tunnel"]["client_binary"] = str(client_binary)
            config = validate_config(provisional, runtime_root=self.root, name="gateway", gateway=True)
            self.store.write("profile.json", {"schema_version": 1, "tunnel": config["tunnel"]})
        return {"ok": True, "tunnel_id": tunnel_id, "setup": self.setup_instructions()}

    def start(self, directory: Path | None = None, *, local: bool = False) -> dict:
        with self.store.locked():
            profile = self._profile()
            if profile is None and not local:
                setup = self.setup_instructions()
                setup["key_file_created"] = self._draft_key_file()
                return {"ok": False, "error": "machine setup required before ChatGPT-accessible start", "setup": setup}
            if profile is not None and local:
                raise ValueError("machine connection is configured; omit --local to reuse it")
            registry = self._registry()
            identity, entry = self._entry(registry, directory, None, create=True)
            config = default_config(identity)
            config.update(uid=os.geteuid() or 1000, gid=os.getegid() or 1000,
                          mounts=[{"source": entry["directory"], "alias": "project", "read_only": False, "protect_sensitive": True}])
            if profile:
                config["protected_host_paths"] = [str(Path(profile["tunnel"]["api_key_file"]).parent)]
            config = validate_config(config, runtime_root=self.root, name=identity)
            registry["entries"][identity] = entry
            self.store.write("registry.json", registry)
            deployment = self._deployment(identity)
            status = deployment.status()
            if status["installed"] and status.get("layout") != "executor-only":
                if any(c["running"] for c in status["containers"].values()):
                    raise ValueError("older paired service requires explicit stop before migration; then start again")
                deployment.uninstall()
                status = deployment.status()
            if not status["installed"]:
                deployment.path.mkdir(parents=True, exist_ok=True, mode=0o700)
                atomic_json(deployment.config_path, config)
                deployment.install()
            elif not all(c["running"] for c in status["containers"].values()) or not status["containers"]:
                if deployment._record()["current"]["config"] != config:
                    atomic_json(deployment.config_path, config)
                    deployment.update()
                else:
                    deployment.start()
            gateway, error = self._refresh(registry, profile)
            summary = self._summary(identity, entry)
            summary.update(ok=summary["local_ready"] and error is None, gateway=gateway,
                           gateway_ready=gateway["gateway_ready"] and identity in gateway["workspaces"], tunnel_ready=gateway["tunnel_ready"])
            if error:
                summary["error"] = error
            return summary

    def stop(self, directory: Path | None = None, *, identity: str | None = None) -> dict:
        with self.store.locked():
            registry = self._registry()
            identity, entry = self._entry(registry, directory, identity)
            deployment = self._deployment(identity)
            if deployment.record_path.exists():
                deployment.stop()
            gateway, error = self._refresh(registry, self._profile(require_files=False))
            result = self._summary(identity, entry)
            result.update(ok=error is None, gateway=gateway)
            if error:
                result["error"] = error
            return result

    def status(self, directory: Path | None = None, *, identity: str | None = None) -> dict:
        with self.store.locked():
            registry = self._registry()
            identity, entry = self._entry(registry, directory, identity)
            gateway = self.gateway.status()
            result = self._summary(identity, entry)
            result.update(ok=True, gateway=gateway, gateway_ready=gateway["gateway_ready"] and identity in gateway["workspaces"], tunnel_ready=gateway["tunnel_ready"])
            return result

    def list(self, *, include_stopped: bool = False) -> dict:
        with self.store.locked():
            registry = self._registry()
            items = [self._summary(i, e) for i, e in sorted(registry["entries"].items())]
            return {"ok": True, "workspaces": [s for s in items if include_stopped or s["running"]],
                    "gateway": self.gateway.status(), "setup_required": self._profile(require_files=False) is None}

    def remove(self, directory: Path | None = None, *, identity: str | None = None, purge_data: bool = False) -> dict:
        with self.store.locked():
            registry = self._registry()
            identity, entry = self._entry(registry, directory, identity)
            deployment = self._deployment(identity)
            if deployment.record_path.exists():
                deployment.stop()
            gateway, error = self._refresh(registry, self._profile(require_files=False))
            if error:
                return {"ok": False, "workspace_id": identity, "error": error, "gateway": gateway}
            if not any(self._summary(i, e)["running"] for i, e in registry["entries"].items()) and purge_data:
                self.gateway.purge()
            if deployment.record_path.exists():
                deployment.uninstall(purge_data=purge_data)
            del registry["entries"][identity]
            self.store.write("registry.json", registry)
            return {"ok": True, "workspace_id": identity, "directory": entry["directory"], "data_retained": not purge_data}
