"""Strict owner configuration and mount grants, evaluated before Docker mutation."""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import stat
from pathlib import Path

from .files import FileBoundary

NAME = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")
TUNNEL_KEY_PLACEHOLDER = "REPLACE_WITH_YOUR_OPENAI_TUNNEL_RUNTIME_KEY"
DEFAULT_LIMITS = {
    "cpus": 2, "memory_mb": 512, "pids": 128, "max_jobs": 1000,
    "concurrency": 4, "timeout_seconds": 900, "output_bytes": 1048576,
}
LIMIT_BOUNDS = {
    "cpus": (0.25, 64), "memory_mb": (64, 65536), "pids": (16, 4096),
    "max_jobs": (1, 100000), "concurrency": (1, 32),
    "timeout_seconds": (1, 86400), "output_bytes": (1024, 16777216),
}
SECRET_PARTS = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".codex", ".config", ".docker", ".tokimon", ".secrets"}


def sensitive_part(name: str) -> bool:
    return name in SECRET_PARTS or name == ".env" or name.startswith(".env.")


def protected_masks(mount: dict, runtime_root: Path, protected_paths: list[str]) -> list[dict]:
    """Derive only existing, safely addressable overlays; never alter project targets."""
    if not mount.get("protect_sensitive", False):
        return []
    source = Path(mount["source"])
    candidates = set()
    runtime_path = runtime_root.resolve()
    with os.scandir(source) as children:
        for child in children:
            if child.name == ".tokimon" and runtime_path.is_relative_to(source / child.name):
                continue
            if sensitive_part(child.name):
                candidates.add(source / child.name)
    secrets = runtime_path / "secrets"
    if source in secrets.parents:
        try:
            secrets.lstat()
        except FileNotFoundError:
            pass
        else:
            candidates.add(secrets)
    for raw in protected_paths:
        path = Path(raw)
        if path == source or path in source.parents:
            raise ValueError("grant lies inside a declared protected path")
        if source in path.parents:
            candidates.add(path)
    masks = []
    for path in sorted(candidates, key=lambda p: (len(p.parts), str(p))):
        if any(Path(item["source"]) in path.parents for item in masks):
            continue
        info = path.lstat()
        directory = stat.S_ISDIR(info.st_mode)
        if not directory and not stat.S_ISREG(info.st_mode):
            raise ValueError("protected target must be a nonsymlink regular file or directory")
        absolute_path(str(path), directory=directory)
        relative = path.relative_to(source)
        target = f"/mounts/{mount['alias']}/{relative.as_posix()}"
        if any(c in target for c in (":", ",", "\n", "\r")) or len(target.encode()) > 4096:
            raise ValueError("protected target cannot be safely overlaid")
        masks.append({"source": str(path), "target": target, "directory": directory})
        if len(masks) > 128:
            raise ValueError("protected grant exceeds 128 masks")
    return masks


def valid_name(value: object) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError("name/alias must match [a-z][a-z0-9-]{0,31}")
    return value


def integer(value: object, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in {low}..{high}")
    return value


def fields(value: object, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - (optional or set()):
        raise ValueError(f"expected object fields {sorted(required)}, optional {sorted(optional or set())}")
    return value


def default_config(name: str) -> dict:
    return {"schema_version": 1, "name": valid_name(name), "uid": 1000, "gid": 1000,
            "mounts": [], "blocked_networks": [], "limits": dict(DEFAULT_LIMITS), "tunnel": None}


def absolute_path(value: object, *, directory: bool) -> Path:
    if not isinstance(value, str) or not value or "\0" in value or "," in value:
        raise ValueError("path must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be absolute without parent traversal")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("owner paths must not contain symlinks")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError("owner path must be an existing directory" if directory else "owner path must be a regular file")
    return path.resolve(strict=True)


def overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def validate_config(payload: object, *, runtime_root: Path, name: str, require_tunnel_files: bool = True, gateway: bool = False) -> dict:
    config = copy.deepcopy(fields(payload, set(default_config(name)), {"protected_host_paths"}))
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if valid_name(config["name"]) != valid_name(name):
        raise ValueError("configuration name does not match deployment")
    for key in ("uid", "gid"):
        integer(config[key], key, 1, 2147483647)
    limits = fields(config["limits"], set(DEFAULT_LIMITS))
    for key, (low, high) in LIMIT_BOUNDS.items():
        value = limits[key]
        if key == "cpus":
            if type(value) not in (int, float) or not low <= value <= high:
                raise ValueError("cpus must be a finite number in 0.25..64")
        else:
            integer(value, key, low, high)
    if not isinstance(config["mounts"], list) or len(config["mounts"]) > 32:
        raise ValueError("mounts must be an array of at most 32 grants")
    roots = []
    aliases = set()
    broad = {Path(x) for x in ("/", "/home", "/tmp", "/var", "/usr", "/opt", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/root")}
    controls = [Path(x) for x in ("/proc", "/sys", "/dev", "/run", "/etc", "/var/run", "/var/lib/docker")]
    protected = config.get("protected_host_paths", [])
    if not isinstance(protected, list) or len(protected) > 32:
        raise ValueError("protected_host_paths must contain at most 32 paths")
    for raw in protected:
        if not isinstance(raw, str):
            raise ValueError("protected path must be a string")
        absolute_path(raw, directory=Path(raw).is_dir())
    for item in config["mounts"]:
        fields(item, {"source", "alias", "read_only"}, {"protect_sensitive"})
        protect = item.get("protect_sensitive", False)
        if type(protect) is not bool:
            raise ValueError("protect_sensitive must be boolean")
        alias = valid_name(item["alias"])
        if alias in aliases or type(item["read_only"]) is not bool:
            raise ValueError("mount aliases must be unique and read_only must be boolean")
        source = absolute_path(item["source"], directory=True)
        if (source in broad or source == Path.home() or source.parent == Path("/home")
                or any(overlaps(source, p) for p in controls)
                or any(sensitive_part(part) for part in source.parts)
                or (overlaps(source, runtime_root.resolve()) and not (protect and source in runtime_root.resolve().parents))
                or any(overlaps(source, Path(p)) and not (protect and source in Path(p).parents) for p in protected)
                or any(overlaps(source, p) for p in roots) or "," in str(source)):
            raise ValueError("mount grant overlaps a broad, control, secret, runtime, or existing grant path")
        aliases.add(alias)
        roots.append(source)
        item["source"] = str(source)
        protected_masks(item, runtime_root, protected)
    if not isinstance(config["blocked_networks"], list) or len(config["blocked_networks"]) > 256:
        raise ValueError("blocked_networks must be an array of at most 256 canonical CIDRs")
    for network in config["blocked_networks"]:
        if not isinstance(network, str) or str(ipaddress.ip_network(network, strict=True)) != network:
            raise ValueError("blocked_networks requires canonical CIDRs")
    if config["tunnel"] is not None:
        if not gateway:
            raise ValueError("per-deployment tunnels are retired; keep tunnel null and use workspace setup for the machine Gateway")
        tunnel = fields(config["tunnel"], {"tunnel_id", "api_key_file", "client_binary"})
        if not isinstance(tunnel["tunnel_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", tunnel["tunnel_id"]):
            raise ValueError("tunnel_id must be an opaque identifier without whitespace")
        for field in ("api_key_file", "client_binary"):
            value = tunnel[field]
            if not isinstance(value, str) or len(value.encode()) > 4096 or not Path(value).is_absolute() or ".." in Path(value).parts or any(c in value for c in ("\0", ",")):
                raise ValueError("tunnel files must be bounded absolute references without parent traversal")
        if not require_tunnel_files:
            return config
        key = absolute_path(tunnel["api_key_file"], directory=False)
        with FileBoundary({str(key.parent): True}).open_regular(str(key), os.O_RDONLY) as fd:
            info = os.fstat(fd)
            if info.st_uid == 0 or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077 or info.st_size > 16384:
                raise ValueError("tunnel key must be a private regular file owned/readable by the non-root deployment operator")
            content = os.read(fd, 16385)
            if len(content) > 16384:
                raise ValueError("tunnel key exceeds 16 KiB")
            content = content.strip()
            if not content or content == TUNNEL_KEY_PLACEHOLDER.encode():
                raise ValueError("fill the private tunnel key file before setup or start; blank or unchanged template key")
        binary = absolute_path(tunnel["client_binary"], directory=False)
        if not os.access(binary, os.X_OK):
            raise ValueError("tunnel client must be executable")
        if any(overlaps(p, key) or overlaps(p, binary) for p in roots):
            raise ValueError("executor mounts must not overlap tunnel credentials/client")
        tunnel.update(api_key_file=str(key), client_binary=str(binary))
    return config


def load_config(path: Path, *, runtime_root: Path, name: str) -> dict:
    if path.stat().st_size > 65536:
        raise ValueError("configuration exceeds 64 KiB")
    return validate_config(json.loads(path.read_text()), runtime_root=runtime_root, name=name)


def worker_config(config: dict) -> dict:
    return {"name": config["name"], "uid": config["uid"], "gid": config["gid"],
            "limits": config["limits"], "roots": {"/workspace": False, **{
                f"/mounts/{m['alias']}": m["read_only"] for m in config["mounts"]}}}
