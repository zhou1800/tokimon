#!/usr/bin/env python3
"""Remote browser broker MCP server.

The broker exposes one browser session per MCP session.  Browser sessions are
implemented as isolated Docker containers with a headed Chromium, CDP, and
noVNC endpoint.  The MCP client does not pass owner ids or browser session ids
for normal use; the HTTP MCP session header is the isolation boundary.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import errno
import functools
import getpass
import hashlib
import http.client
import ipaddress
import json
import logging
import math
import os
import random
import re
import select
import secrets
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from . import artifacts

from tokimon.tools.mcp._protocol import response, error_response, tool_result, handle_request
from tokimon.tools._runtime import configure_logger, format_log_record
from tokimon.runtime import (
    resolve_env_runtime_path,
    runtime_root as repository_runtime_root,
    runtime_workspace_root,
    source_root,
)

WORKSPACE_ROOT = source_root()
REMOTE_BROWSER_RUNTIME_RELATIVE_PATH = Path("tools") / "mcp" / "remote-browser"
DEFAULT_RUNTIME_ROOT = repository_runtime_root(WORKSPACE_ROOT) / REMOTE_BROWSER_RUNTIME_RELATIVE_PATH
IMAGE_CONTEXT_DIR = Path(__file__).resolve().parent / "image"
SECCOMP_PROFILE_PATH = IMAGE_CONTEXT_DIR / "seccomp_profile.json"

SERVER_NAME = "tokimon-remote-browser-broker"
SERVER_VERSION = "0.2.0"
SUPPORTED_PROTOCOL_VERSION = "2025-03-26"
LOGGER_NAME = "tokimon.tools.mcp.remote_browser_broker"

__all__ = [
    "BrokerConfig",
    "BrowserBrokerError",
    "BrowserSession",
    "MCPServer",
    "RemoteBrowserBroker",
    "build_parser",
    "health_payload",
    "main",
    "make_handler",
    "mcp_tools",
    "replace_config",
    "serve",
    "serve_http",
]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT_BASE = 20000
DEFAULT_PORT_SPAN = 1000
DEFAULT_CDP_PORT_RANGE = (30000, 39999)
DEFAULT_NOVNC_PORT_RANGE = (40000, 49999)
DEFAULT_VIEWPORT = "1400,900"
DEFAULT_SCREEN = "1400x900x24"
DEFAULT_IMAGE_REPOSITORY = "tokimon-remote-browser-session"
INSTALLATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
INSTALLATION_ID_COMPONENT_RE = re.compile(r"[^a-z0-9_.-]+")
DOCKER_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")
DOCKER_FULL_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_OWNER_USER = getpass.getuser()
DEFAULT_RETENTION_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_RETENTION_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_ACTIVE_SESSIONS = 8
DEFAULT_MAX_TOTAL_SESSIONS = 32
DEFAULT_IDLE_TIMEOUT_SECONDS = 30 * 60
DEFAULT_NOVNC_TOKEN_TTL_SECONDS = 24 * 60 * 60
DEFAULT_REF_TTL_SECONDS = 120
MAX_OPERATION_PARAMETER_BYTES = 1024 * 1024
MAX_MCP_REQUEST_BYTES = 2 * 1024 * 1024
MAX_NOVNC_REQUEST_BYTES = 1024 * 1024
HTTP_BODY_READ_TIMEOUT_SECONDS = 10.0
MAX_LEASE_WAIT_SECONDS = 10.0
DOCKER_INSPECT_TIMEOUT_SECONDS = 20.0
DOCKER_START_TIMEOUT_SECONDS = 90.0
DOCKER_BUILD_TIMEOUT_SECONDS = 600.0
DOCKER_REMOVE_TIMEOUT_SECONDS = 30.0
MAX_WAIT_SECONDS = 10.0
MAX_WAIT_TIMEOUT_MS = 10_000
MAX_SNAPSHOT_TEXT_CHARS = 100_000
MAX_SNAPSHOT_ELEMENTS = 1_000
MAX_SNAPSHOT_FIELD_CHARS = 2_048
MAX_CDP_HANDSHAKE_BYTES = 64 * 1024
MAX_CDP_FRAME_BYTES = 32 * 1024 * 1024
MAX_CDP_UNMATCHED_MESSAGES = 1_000
MAX_SCREENSHOT_BYTES = 16 * 1024 * 1024
MAX_SCREENSHOT_ARTIFACTS = 100
MAX_SCREENSHOT_TOTAL_BYTES = 256 * 1024 * 1024
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_STAGED_UPLOADS = 64
MAX_STAGED_UPLOAD_TOTAL_BYTES = 512 * 1024 * 1024
DEFAULT_BACKEND_RESPONSE_MAX_CHARS = 2_000_000
MAX_BACKEND_RESPONSE_MAX_CHARS = 10_000_000
DEFAULT_BACKEND_REQUEST_TIMEOUT_MS = 60_000
MAX_BACKEND_REQUEST_TIMEOUT_MS = 60_000
DEFAULT_NETWORK_POLICY = "per-session-bridge-unrestricted-egress"
TEMPLATE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TEMPLATE_VERSION_ID_RE = re.compile(r"^v[A-Za-z0-9_.-]{1,127}$")
PROFILE_COPY_IGNORE_NAMES = {
    "BrowserMetrics",
    "Crashpad",
    "DevToolsActivePort",
    "GrShaderCache",
    "GraphiteDawnCache",
    "ShaderCache",
    "SingletonCookie",
    "SingletonLock",
    "SingletonSocket",
}


def validate_copy_source_tree(root: Path, *, label: str, ignore: Any = None) -> None:
    """Reject links and special files in the entries that will be copied."""

    if root.is_symlink() or not root.is_dir():
        raise BrowserBrokerError(
            f"{label} must be a real directory",
            code=ERROR_INVALID_ARGUMENT,
        )
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        base = Path(directory)
        if ignore is not None:
            ignored = ignore(directory, directory_names + file_names)
            directory_names[:] = [name for name in directory_names if name not in ignored]
            file_names = [name for name in file_names if name not in ignored]
        for name in directory_names:
            candidate = base / name
            if candidate.is_symlink():
                raise BrowserBrokerError(
                    f"{label} contains a directory symlink: {candidate.relative_to(root)}",
                    code=ERROR_INVALID_ARGUMENT,
                )
        for name in file_names:
            candidate = base / name
            try:
                mode = candidate.lstat().st_mode
            except OSError as exc:
                raise BrowserBrokerError(
                    f"{label} contains an unreadable entry: {candidate.relative_to(root)}",
                    code=ERROR_INVALID_ARGUMENT,
                ) from exc
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise BrowserBrokerError(
                    f"{label} contains a link or special file: {candidate.relative_to(root)}",
                    code=ERROR_INVALID_ARGUMENT,
                )

STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_PAUSED_OBSERVED = "paused_observed"
STATE_STOPPED_RETAINED = "stopped_retained"
STATE_EXPIRED = "expired"
STATE_ORPHANED = "orphaned"
STATE_PEER_OWNED = "peer_owned"
STATE_UNKNOWN = "unknown"

PACING_MODE_HUMAN = "human"
TOKEN_EXPOSURE_BASELINE = "baseline_unprotected"
TEMPLATE_ROLE_RUNTIME = "runtime"
TEMPLATE_ROLE_EDITING = "template_editing"
TEMPLATE_ROLE_STAGING = "template_staging"
TEMPLATE_STATE_EDITING = "editing"
TEMPLATE_STATE_SEALED = "sealed"
TEMPLATE_STATE_DELETED = "deleted"
ERROR_BROKER = "BROKER_ERROR"
ERROR_INVALID_ARGUMENT = "INVALID_ARGUMENT"
ERROR_SESSION_CAPACITY_EXHAUSTED = "SESSION_CAPACITY_EXHAUSTED"
ERROR_SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
ERROR_SESSION_STOPPED = "SESSION_STOPPED"
ERROR_SESSION_TEMPLATE_MISMATCH = "SESSION_TEMPLATE_MISMATCH"
ERROR_TEMPLATE_EXISTS = "TEMPLATE_EXISTS"
ERROR_TEMPLATE_NOT_FOUND = "TEMPLATE_NOT_FOUND"
ERROR_TEMPLATE_CONFLICT = "TEMPLATE_CONFLICT"
ERROR_SESSION_NOT_RETAINED = "SESSION_NOT_RETAINED"
ERROR_SESSION_RECOVERY_UNAVAILABLE = "SESSION_RECOVERY_UNAVAILABLE"
ERROR_PAUSED = "PAUSED"
ERROR_STALE_REF = "STALE_REF"
ERROR_TAB_NOT_FOUND = "TAB_NOT_FOUND"
NOVNC_TOKEN_COOKIE_NAME = "tokimon_rb_novnc_token"
def slug_component(value: str, *, fallback: str) -> str:
    slug = INSTALLATION_ID_COMPONENT_RE.sub("-", value.strip().lower()).strip("-._")
    return slug or fallback


def default_installation_id(workspace_root: Path = WORKSPACE_ROOT, *, user: str | None = None) -> str:
    owner = slug_component(user or DEFAULT_OWNER_USER, fallback="user")
    runtime_owner = runtime_workspace_root(workspace_root)
    digest = hashlib.sha256(str(runtime_owner).encode("utf-8")).hexdigest()[:12]
    candidate = f"{owner}-{digest}"
    return candidate[:64].rstrip("-._") or digest


def validate_installation_id(value: str) -> str:
    candidate = value.strip()
    if not INSTALLATION_ID_RE.fullmatch(candidate):
        raise BrowserBrokerError(
            "TOKIMON_RB_INSTALLATION_ID must start with an ASCII letter or number "
            "and contain only letters, numbers, dots, dashes, or underscores.",
            code=ERROR_INVALID_ARGUMENT,
        )
    return candidate


def default_image_name(installation_id: str) -> str:
    return f"{DEFAULT_IMAGE_REPOSITORY}:{installation_id.lower()}"


def default_broker_port(installation_id: str) -> int:
    digest = int(hashlib.sha256(installation_id.encode("utf-8")).hexdigest()[:8], 16)
    return DEFAULT_PORT_BASE + (digest % DEFAULT_PORT_SPAN)


DEFAULT_INSTALLATION_ID = default_installation_id()
DEFAULT_IMAGE_NAME = default_image_name(DEFAULT_INSTALLATION_ID)
DEFAULT_PORT = default_broker_port(DEFAULT_INSTALLATION_ID)


def network_policy_diagnostics(policy: str) -> dict[str, object]:
    return {
        "id": policy,
        "docker_network_mode": "per_session_user_defined_bridge",
        "peer_container_isolation": "enforced_for_unattached_containers",
        "network_membership": "one_broker_session_container",
        "internet_egress": "allowed",
        "host_admin_endpoints": "not_enforced_by_broker",
        "docker_socket": "not_mounted",
        "metadata_services": "not_enforced_by_broker",
        "private_local_networks": "not_enforced_by_broker",
        "local_test_pages": "disabled_without_explicit_test_setup",
        "enforcement": "per_session_network_membership_only",
        "limitations": [
            "Docker host, Desktop, WSL, and Linux network behavior can differ.",
            "Principals with Docker authority can attach another container and are trusted by this design.",
            "The per-session bridge does not enforce host, metadata-service, private-network, or internet egress denial.",
            "Use a host firewall or another external egress control when those destinations must be unreachable.",
        ],
    }


class BrowserBrokerError(RuntimeError):
    """Raised when the browser broker cannot complete an operation."""

    def __init__(
        self,
        message: str,
        *,
        code: str = ERROR_BROKER,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = dict(details or {})

    def to_payload(self) -> dict[str, object]:
        return {
            "error": self.message,
            "error_code": self.code,
            **self.details,
        }


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class BrokerConfig:
    workspace_root: Path = WORKSPACE_ROOT
    runtime_owner_root: Path = field(init=False)
    runtime_root: Path = DEFAULT_RUNTIME_ROOT
    installation_id: str = DEFAULT_INSTALLATION_ID
    owner_user: str = DEFAULT_OWNER_USER
    image_name: str = DEFAULT_IMAGE_NAME
    docker_command: str = "docker"
    bind_host: str = DEFAULT_HOST
    broker_port: int = DEFAULT_PORT
    cdp_port_range: tuple[int, int] = DEFAULT_CDP_PORT_RANGE
    novnc_port_range: tuple[int, int] = DEFAULT_NOVNC_PORT_RANGE
    ssh_host: str | None = None
    public_host: str = "127.0.0.1"
    viewport: str = DEFAULT_VIEWPORT
    screen: str = DEFAULT_SCREEN
    startup_timeout_sec: float = 45.0
    retention_ttl_seconds: int = DEFAULT_RETENTION_TTL_SECONDS
    max_retention_seconds: int = DEFAULT_MAX_RETENTION_SECONDS
    max_active_sessions: int = DEFAULT_MAX_ACTIVE_SESSIONS
    max_total_sessions: int = DEFAULT_MAX_TOTAL_SESSIONS
    idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS
    novnc_token_ttl_seconds: int = DEFAULT_NOVNC_TOKEN_TTL_SECONDS
    network_policy: str = DEFAULT_NETWORK_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_owner_root", runtime_workspace_root(self.workspace_root))
        if not is_loopback_host(self.bind_host):
            raise BrowserBrokerError(
                "Browser MCP requires a loopback bind host",
                code=ERROR_INVALID_ARGUMENT,
            )
        installation_id = validate_installation_id(self.installation_id)
        object.__setattr__(self, "installation_id", installation_id)
        owner_user = self.owner_user.strip() or DEFAULT_OWNER_USER
        object.__setattr__(self, "owner_user", owner_user)
        if self.image_name == DEFAULT_IMAGE_NAME and installation_id != DEFAULT_INSTALLATION_ID:
            object.__setattr__(self, "image_name", default_image_name(installation_id))
        if self.broker_port == DEFAULT_PORT and installation_id != DEFAULT_INSTALLATION_ID:
            object.__setattr__(self, "broker_port", default_broker_port(installation_id))
        if self.max_active_sessions <= 0:
            raise BrowserBrokerError(
                "max_active_sessions must be positive",
                code=ERROR_INVALID_ARGUMENT,
            )
        if self.max_total_sessions < self.max_active_sessions:
            raise BrowserBrokerError(
                "max_total_sessions must be at least max_active_sessions",
                code=ERROR_INVALID_ARGUMENT,
            )
        if self.idle_timeout_seconds < 0:
            raise BrowserBrokerError(
                "idle_timeout_seconds must be nonnegative",
                code=ERROR_INVALID_ARGUMENT,
            )

    @classmethod
    def from_env(cls, *, env: Mapping[str, str] | None = None) -> "BrokerConfig":
        effective_env = dict(os.environ if env is None else env)
        installation_id = validate_installation_id(
            effective_env.get("TOKIMON_RB_INSTALLATION_ID")
            or default_installation_id(WORKSPACE_ROOT)
        )
        runtime_root = resolve_env_runtime_path(
            default_relative_path=REMOTE_BROWSER_RUNTIME_RELATIVE_PATH,
            env_var="TOKIMON_RB_RUNTIME_ROOT",
            env=effective_env,
            workspace_root=WORKSPACE_ROOT,
        )
        return cls(
            workspace_root=WORKSPACE_ROOT,
            runtime_root=runtime_root,
            installation_id=installation_id,
            owner_user=effective_env.get("TOKIMON_RB_OWNER_USER", DEFAULT_OWNER_USER),
            image_name=effective_env.get(
                "TOKIMON_RB_IMAGE",
                default_image_name(installation_id),
            ),
            docker_command=effective_env.get("TOKIMON_RB_DOCKER", "docker"),
            bind_host=effective_env.get("TOKIMON_RB_HOST", DEFAULT_HOST),
            broker_port=int(
                effective_env.get("TOKIMON_RB_PORT")
                or str(default_broker_port(installation_id))
            ),
            cdp_port_range=parse_port_range(
                effective_env.get("TOKIMON_RB_CDP_PORT_RANGE"),
                DEFAULT_CDP_PORT_RANGE,
            ),
            novnc_port_range=parse_port_range(
                effective_env.get("TOKIMON_RB_NOVNC_PORT_RANGE"),
                DEFAULT_NOVNC_PORT_RANGE,
            ),
            ssh_host=empty_to_none(effective_env.get("TOKIMON_RB_SSH_HOST")),
            public_host=effective_env.get("TOKIMON_RB_PUBLIC_HOST", "127.0.0.1"),
            viewport=effective_env.get("TOKIMON_RB_VIEWPORT", DEFAULT_VIEWPORT),
            screen=effective_env.get("TOKIMON_RB_SCREEN", DEFAULT_SCREEN),
            startup_timeout_sec=float(
                effective_env.get("TOKIMON_RB_STARTUP_TIMEOUT_SEC", "45")
            ),
            retention_ttl_seconds=int(
                effective_env.get(
                    "TOKIMON_RB_RETENTION_TTL_SECONDS",
                    str(DEFAULT_RETENTION_TTL_SECONDS),
                )
            ),
            max_retention_seconds=int(
                effective_env.get(
                    "TOKIMON_RB_MAX_RETENTION_SECONDS",
                    str(DEFAULT_MAX_RETENTION_SECONDS),
                )
            ),
            max_active_sessions=int(
                effective_env.get(
                    "TOKIMON_RB_MAX_ACTIVE_SESSIONS",
                    str(DEFAULT_MAX_ACTIVE_SESSIONS),
                )
            ),
            max_total_sessions=int(
                effective_env.get(
                    "TOKIMON_RB_MAX_TOTAL_SESSIONS",
                    str(DEFAULT_MAX_TOTAL_SESSIONS),
                )
            ),
            idle_timeout_seconds=int(
                effective_env.get(
                    "TOKIMON_RB_IDLE_TIMEOUT_SECONDS",
                    str(DEFAULT_IDLE_TIMEOUT_SECONDS),
                )
            ),
            novnc_token_ttl_seconds=int(
                effective_env.get(
                    "TOKIMON_RB_NOVNC_TOKEN_TTL_SECONDS",
                    str(DEFAULT_NOVNC_TOKEN_TTL_SECONDS),
                )
            ),
            network_policy=effective_env.get(
                "TOKIMON_RB_NETWORK_POLICY",
                DEFAULT_NETWORK_POLICY,
            ),
        )


@dataclass
class BrowserSession:
    id: str
    mcp_session_key: str
    profile_dir: Path
    cdp_port: int
    novnc_port: int
    novnc_backend_port: int
    container_name: str
    created_at: str
    last_used_at: str
    container_id: str | None = None
    template_name: str | None = None
    state: str = STATE_RUNNING
    retention_expires_at: str | None = None
    recovery_token_hash: str | None = field(default=None, repr=False)
    pacing_mode: str = PACING_MODE_HUMAN
    token_exposure_mode: str = TOKEN_EXPOSURE_BASELINE
    template_version_id: str | None = None
    template_role: str = TEMPLATE_ROLE_RUNTIME
    novnc_token: str | None = None
    novnc_token_issued_at: str | None = None
    novnc_token_expires_at: str | None = None
    novnc_token_active: bool = False
    active_tab_id: str | None = None
    ref_epoch: int = 0
    element_refs: dict[str, dict[str, object]] = field(default_factory=dict)
    operation_lock: Any = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )
    ownership_lease: Any = field(
        default=None,
        repr=False,
        compare=False,
    )
    mcp_ownership_lease: Any = field(
        default=None,
        repr=False,
        compare=False,
    )
    template_editor_lease: Any = field(
        default=None,
        repr=False,
        compare=False,
    )
    port_ownership_leases: list[Any] = field(
        default_factory=list,
        repr=False,
        compare=False,
    )

    def touch(self) -> None:
        self.last_used_at = utc_now()

    def to_status_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "state": self.state,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "template_name": self.template_name,
            "template_version_id": self.template_version_id,
            "template_role": self.template_role,
            "retention_expires_at": self.retention_expires_at,
            "pacing_mode": self.pacing_mode,
            "active_tab_id": self.active_tab_id,
        }

    def to_internal_diagnostics(self) -> dict[str, object]:
        return {
            **self.to_status_dict(),
            "container_name": self.container_name,
            "profile_dir": str(self.profile_dir),
            "cdp_port": self.cdp_port,
            "novnc_port": self.novnc_port,
            "novnc_backend_port": self.novnc_backend_port,
            "token_exposure_mode": self.token_exposure_mode,
        }

    def invalidate_refs(self) -> None:
        self.ref_epoch += 1
        self.element_refs.clear()

    def record_snapshot_refs(self, page: dict[str, object]) -> None:
        self.invalidate_refs()
        elements = page.get("elements")
        if not isinstance(elements, list):
            return
        now = time.monotonic()
        for index, item in enumerate(elements):
            if not isinstance(item, dict):
                continue
            ref = f"e{self.ref_epoch}_{index}"
            item["ref"] = ref
            self.element_refs[ref] = {
                "ref": ref,
                "index": index,
                "tab_id": self.active_tab_id,
                "snapshot_id": f"{self.id}:{self.ref_epoch}",
                "created_monotonic": now,
            }

    def resolve_ref(self, ref: str) -> dict[str, object]:
        item = self.element_refs.get(ref)
        if item is None or item.get("tab_id") != self.active_tab_id:
            raise BrowserBrokerError(
                f"Element ref is stale or unknown: {ref}",
                code=ERROR_STALE_REF,
            )
        created = item.get("created_monotonic")
        if isinstance(created, (int, float)) and time.monotonic() - created > DEFAULT_REF_TTL_SECONDS:
            self.element_refs.pop(ref, None)
            raise BrowserBrokerError(
                f"Element ref expired: {ref}",
                code=ERROR_STALE_REF,
            )
        return item


@dataclass(frozen=True)
class BrowserTemplate:
    name: str
    version_id: str
    state: str
    template_dir: Path
    profile_dir: Path
    metadata_path: Path
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        payload = dict(self.metadata)
        payload.update(
            {
                "name": self.name,
                "version_id": self.version_id,
                "state": self.state,
                "template_dir": str(self.template_dir),
                "profile_dir": str(self.profile_dir),
            }
        )
        return payload


@dataclass(frozen=True)
class StagedUploadFile:
    source_name: str
    upload_name: str
    container_path: str
    size_bytes: int


def empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_after(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_expired(value: str | None) -> bool:
    parsed = parse_datetime(value)
    if parsed is None:
        return False
    return parsed <= datetime.now(timezone.utc)


def seconds_until(value: str | None, *, default: float) -> float:
    parsed = parse_datetime(value)
    if parsed is None:
        return default
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def validate_operation_parameters(parameters: Mapping[str, object]) -> None:
    encoded = json.dumps(parameters, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_OPERATION_PARAMETER_BYTES:
        raise BrowserBrokerError(
            f"Operation parameters exceed {MAX_OPERATION_PARAMETER_BYTES} bytes",
            code=ERROR_INVALID_ARGUMENT,
        )


def parse_port_range(raw: str | None, default: tuple[int, int]) -> tuple[int, int]:
    if not raw:
        return default
    if "-" not in raw:
        raise BrowserBrokerError(
            f"Port range must use low-high form: {raw}",
            code=ERROR_INVALID_ARGUMENT,
        )
    low_text, high_text = raw.split("-", 1)
    low = int(low_text)
    high = int(high_text)
    if low <= 0 or high <= 0 or high < low:
        raise BrowserBrokerError(
            f"Invalid port range: {raw}",
            code=ERROR_INVALID_ARGUMENT,
        )
    return (low, high)


def validate_template_name(name: str | None) -> str:
    if name is None:
        raise BrowserBrokerError("template_name is required", code=ERROR_INVALID_ARGUMENT)
    normalized = name.strip()
    if not TEMPLATE_NAME_RE.fullmatch(normalized):
        raise BrowserBrokerError(
            "template_name must start with a letter or digit and contain only "
            "letters, digits, dot, underscore, or dash; maximum length is 64",
            code=ERROR_INVALID_ARGUMENT,
        )
    return normalized


def validate_template_version_id(version_id: str | None) -> str:
    if version_id is None:
        raise BrowserBrokerError("version_id is required", code=ERROR_INVALID_ARGUMENT)
    normalized = version_id.strip()
    if not TEMPLATE_VERSION_ID_RE.fullmatch(normalized):
        raise BrowserBrokerError(
            "version_id must be one sealed v* basename containing only letters, "
            "digits, dot, underscore, or dash",
            code=ERROR_INVALID_ARGUMENT,
        )
    return normalized


def profile_copy_ignore(_: str, names: list[str]) -> set[str]:
    ignored = set(PROFILE_COPY_IGNORE_NAMES).intersection(names)
    ignored.update(name for name in names if name.endswith(".tmp"))
    return ignored


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(6)}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def find_available_port(
    port_range: tuple[int, int],
    *,
    excluded_ports: set[int] | None = None,
    host: str = "127.0.0.1",
    attempts: int = 300,
) -> int:
    low, high = port_range
    excluded = excluded_ports or set()
    for _ in range(attempts):
        port = random.randint(low, high)
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise BrowserBrokerError(f"No available port in range {low}-{high}")


def docker_user_args() -> list[str]:
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return ["--user", f"{os.getuid()}:{os.getgid()}"]
    return []


def validate_backend_request_path(path: str) -> str:
    stripped = path.strip()
    parsed = urllib.parse.urlparse(stripped)
    if parsed.scheme or parsed.netloc or not stripped.startswith("/") or stripped.startswith("//"):
        raise BrowserBrokerError(
            "backend request path must be a same-origin absolute path such as /api/drafts",
            code=ERROR_INVALID_ARGUMENT,
        )
    return stripped


def validate_backend_request_method(method: str) -> str:
    normalized = method.strip().upper()
    if normalized not in {"GET", "POST", "PATCH", "PUT", "DELETE"}:
        raise BrowserBrokerError(
            "backend request method must be GET, POST, PATCH, PUT, or DELETE",
            code=ERROR_INVALID_ARGUMENT,
        )
    return normalized


def strip_token_query(path: str) -> str:
    parsed = urllib.parse.urlparse(path)
    query_items = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"rb_token", "token"}
    ]
    clean_query = urllib.parse.urlencode(query_items, doseq=True)
    return urllib.parse.urlunparse(
        (
            "",
            "",
            parsed.path or "/",
            parsed.params,
            clean_query,
            parsed.fragment,
        )
    )


def request_token_from_path(path: str) -> str | None:
    parsed = urllib.parse.urlparse(path)
    for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        if key in {"rb_token", "token"} and value:
            return value
    return None


def request_token_from_cookie(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    for item in cookie_header.split(";"):
        if "=" not in item:
            continue
        name, value = item.split("=", 1)
        if name.strip() == NOVNC_TOKEN_COOKIE_NAME:
            return value.strip()
    return None


class StructuredLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return format_log_record(
            record, timestamp=datetime.now(timezone.utc).isoformat(),
            service=SERVER_NAME, runtime=getattr(record, "runtime", "http-mcp"),
            fields=('request_id', 'method', 'path', 'status', 'client', 'mcp_session_key'),
            format_exception=self.formatException,
        )


def configure_runtime_logger(*, stream: TextIO | None = None) -> logging.Logger:
    return configure_logger(LOGGER_NAME, StructuredLogFormatter, stream=stream)


def log_runtime_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    correlation_id: str | None = None,
    request_id: Any = None,
    method: Any = None,
    path: Any = None,
    status: Any = None,
    client: Any = None,
    mcp_session_key: Any = None,
    runtime: str | None = None,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "correlation_id": correlation_id or "server",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "client": client,
            "mcp_session_key": mcp_session_key,
            "runtime": runtime or "http-mcp",
        },
    )


class NoVNCProxy:
    """Token-gated HTTP/WebSocket proxy for a session's noVNC backend."""

    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        backend_host: str,
        backend_port: int,
        token: str,
        expires_at: str | None = None,
    ) -> None:
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.token = token
        self.expires_at = expires_at
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._thread_started = False
        self._revoked = threading.Event()
        self._connections_lock = threading.Lock()
        self._active_connections: set[socket.socket] = set()

    def start(self) -> None:
        if self._httpd is not None:
            return
        self._revoked.clear()
        proxy = self

        class NoVNCProxyHandler(BaseHTTPRequestHandler):
            server_version = "TokimonNoVNCProxy/0.1"

            def setup(self) -> None:
                super().setup()
                self._proxy_connection_registered = proxy._register_connection(
                    self.connection
                )

            def finish(self) -> None:
                try:
                    super().finish()
                finally:
                    if getattr(self, "_proxy_connection_registered", False):
                        proxy._unregister_connection(self.connection)

            def do_GET(self) -> None:  # noqa: N802 - http.server API.
                self._handle_proxy()

            def do_POST(self) -> None:  # noqa: N802 - http.server API.
                self._handle_proxy()

            def do_HEAD(self) -> None:  # noqa: N802 - http.server API.
                self._handle_proxy()

            def _handle_proxy(self) -> None:
                if proxy._revoked.is_set():
                    self.send_response(403)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(b"noVNC access is revoked")
                    return
                request_token = request_token_from_path(self.path)
                cookie_token = request_token_from_cookie(self.headers.get("Cookie"))
                if is_expired(proxy.expires_at):
                    self.send_response(403)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(b"noVNC token is expired")
                    return
                if request_token != proxy.token and cookie_token != proxy.token:
                    self.send_response(403)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if self.command != "HEAD":
                        self.wfile.write(b"noVNC token is required")
                    return
                if self.headers.get("Upgrade", "").lower() == "websocket":
                    self._proxy_websocket()
                    return
                self._proxy_http(set_cookie=request_token == proxy.token)

            def _proxy_http(self, *, set_cookie: bool) -> None:
                try:
                    content_length = bounded_content_length(
                        self.headers.get("Content-Length"),
                        maximum=MAX_NOVNC_REQUEST_BYTES,
                    )
                    body = read_http_body_with_deadline(self, content_length)
                except TimeoutError:
                    self.send_error(408, "Request body deadline expired")
                    return
                except BrowserBrokerError as exc:
                    self.send_error(
                        413 if exc.details.get("request_too_large") else 400,
                        str(exc),
                    )
                    return
                headers = self._forward_headers(upgrade=False)
                path = strip_token_query(self.path)
                connection = http.client.HTTPConnection(
                    proxy.backend_host,
                    proxy.backend_port,
                    timeout=20,
                )
                try:
                    connection.request(self.command, path, body=body, headers=headers)
                    backend_response = connection.getresponse()
                    response_body = backend_response.read()
                except OSError as exc:
                    self.send_error(502, f"noVNC backend unavailable: {exc}")
                    return
                finally:
                    connection.close()
                self.send_response(backend_response.status, backend_response.reason)
                for key, value in backend_response.getheaders():
                    if key.lower() in {
                        "connection",
                        "keep-alive",
                        "proxy-authenticate",
                        "proxy-authorization",
                        "te",
                        "trailer",
                        "transfer-encoding",
                        "upgrade",
                    }:
                        continue
                    self.send_header(key, value)
                if set_cookie:
                    self.send_header(
                        "Set-Cookie",
                        f"{NOVNC_TOKEN_COOKIE_NAME}={proxy.token}; Path=/; HttpOnly; SameSite=Lax",
                    )
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(response_body)

            def _proxy_websocket(self) -> None:
                if not getattr(self, "_proxy_connection_registered", False):
                    return
                backend: socket.socket | None = None
                try:
                    backend = socket.create_connection(
                        (proxy.backend_host, proxy.backend_port),
                        timeout=20,
                    )
                except OSError as exc:
                    self.send_error(502, f"noVNC backend unavailable: {exc}")
                    return
                if not proxy._register_connection(backend):
                    backend.close()
                    return
                try:
                    with backend:
                        request = self._raw_backend_request(upgrade=True)
                        backend.sendall(request)
                        self._relay_websocket(backend)
                finally:
                    proxy._unregister_connection(backend)

            def _raw_backend_request(self, *, upgrade: bool) -> bytes:
                path = strip_token_query(self.path)
                headers = self._forward_headers(upgrade=upgrade)
                lines = [f"{self.command} {path} {self.request_version}"]
                for key, value in headers.items():
                    lines.append(f"{key}: {value}")
                lines.append("")
                lines.append("")
                return "\r\n".join(lines).encode("latin1")

            def _forward_headers(self, *, upgrade: bool) -> dict[str, str]:
                headers: dict[str, str] = {}
                hop_by_hop = {
                    "host",
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailer",
                    "transfer-encoding",
                }
                if not upgrade:
                    hop_by_hop.add("upgrade")
                for key, value in self.headers.items():
                    if key.lower() in hop_by_hop:
                        continue
                    headers[key] = value
                headers["Host"] = f"{proxy.backend_host}:{proxy.backend_port}"
                if upgrade:
                    headers["Connection"] = "Upgrade"
                    headers["Upgrade"] = self.headers.get("Upgrade", "websocket")
                else:
                    headers["Connection"] = "close"
                return headers

            def _relay_websocket(self, backend: socket.socket) -> None:
                sockets = [self.connection, backend]
                while True:
                    if proxy._revoked.is_set() or is_expired(proxy.expires_at):
                        return
                    timeout = min(60.0, seconds_until(proxy.expires_at, default=60.0))
                    if timeout <= 0:
                        return
                    try:
                        readable, _, _ = select.select(sockets, [], [], timeout)
                    except OSError:
                        return
                    if not readable:
                        continue
                    for source in readable:
                        if proxy._revoked.is_set() or is_expired(proxy.expires_at):
                            return
                        try:
                            data = source.recv(65536)
                        except OSError:
                            return
                        if not data:
                            return
                        if proxy._revoked.is_set() or is_expired(proxy.expires_at):
                            return
                        destination = backend if source is self.connection else self.connection
                        try:
                            destination.sendall(data)
                        except OSError:
                            return

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.listen_host, self.listen_port), NoVNCProxyHandler)
        self.listen_port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        self._thread_started = True

    def _register_connection(self, connection: socket.socket) -> bool:
        with self._connections_lock:
            if self._revoked.is_set():
                return False
            self._active_connections.add(connection)
            return True

    def _unregister_connection(self, connection: socket.socket) -> None:
        with self._connections_lock:
            self._active_connections.discard(connection)

    def _close_active_connections(self) -> None:
        with self._connections_lock:
            connections = list(self._active_connections)
            self._active_connections.clear()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def stop(self) -> None:
        self._revoked.set()
        self._close_active_connections()
        httpd = self._httpd
        if httpd is None:
            return
        if self._thread_started:
            httpd.shutdown()
        httpd.server_close()
        thread = self._thread
        if thread is not None and self._thread_started:
            thread.join(timeout=2)
            if thread.is_alive():
                raise BrowserBrokerError("noVNC proxy thread did not stop")
        self._close_active_connections()
        self._httpd = None
        self._thread = None
        self._thread_started = False


class DockerSessionLauncher:
    def __init__(self, config: BrokerConfig) -> None:
        self.config = config

    def _network_name(self, session_id: str) -> str:
        scope = json.dumps(
            {
                "checkout": str(self.config.runtime_owner_root),
                "installation": self.config.installation_id,
                "owner": self.config.owner_user,
                "runtime_root": str(self.config.runtime_root),
                "session": session_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:20]
        return f"tokimon-rbn-{digest}"

    def _stable_ownership_labels(self, session_id: str) -> dict[str, str]:
        return {
            "tokimon.remote-browser-broker": "true",
            "tokimon.remote-browser-installation": self.config.installation_id,
            "tokimon.remote-browser-owner-user": self.config.owner_user,
            "tokimon.remote-browser-checkout": str(self.config.runtime_owner_root),
            "tokimon.remote-browser-runtime-root": str(self.config.runtime_root),
            "tokimon.remote-browser-session": session_id,
        }

    @staticmethod
    def _network_not_found(detail: str) -> bool:
        normalized = detail.lower()
        return (
            "no such network" in normalized
            or "no such object" in normalized
            or re.search(
                r"\bnetwork [a-z0-9][a-z0-9_.-]* not found\b",
                normalized,
            )
            is not None
        )

    def _inspect_owned_network(
        self,
        session_id: str,
    ) -> tuple[str, Mapping[str, object]] | None:
        network_name = self._network_name(session_id)
        inspected = subprocess.run(
            [self.config.docker_command, "network", "inspect", network_name],
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
        if inspected.returncode != 0:
            detail = f"{inspected.stderr}\n{inspected.stdout}".lower()
            if self._network_not_found(detail):
                return None
            raise BrowserBrokerError(
                "Docker network identity could not be inspected safely: "
                f"{(inspected.stderr or inspected.stdout or '').strip()}"
            )
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise BrowserBrokerError(
                "Docker network identity inspection returned invalid JSON"
            ) from exc
        if not isinstance(payload, list) or len(payload) != 1:
            raise BrowserBrokerError("Docker network identity inspection was ambiguous")
        record = payload[0]
        if not isinstance(record, Mapping):
            raise BrowserBrokerError("Docker network identity inspection was malformed")
        network_id = str(record.get("Id") or "")
        labels = record.get("Labels")
        containers = record.get("Containers")
        expected_labels = self._stable_ownership_labels(session_id)
        if (
            not DOCKER_CONTAINER_ID_RE.fullmatch(network_id)
            or str(record.get("Name") or "") != network_name
            or str(record.get("Driver") or "") != "bridge"
            or record.get("Internal") is not False
            or not isinstance(labels, Mapping)
            or any(labels.get(key) != value for key, value in expected_labels.items())
            or not isinstance(containers, Mapping)
        ):
            raise BrowserBrokerError(
                "Docker network name now resolves to a foreign or ambiguous identity"
            )
        return network_id, containers

    def _network_is_absent(self, network_id: str) -> bool:
        inspected = subprocess.run(
            [self.config.docker_command, "network", "inspect", network_id],
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
        if inspected.returncode == 0:
            return False
        detail = f"{inspected.stderr}\n{inspected.stdout}".lower()
        if self._network_not_found(detail):
            return True
        raise BrowserBrokerError(
            "Docker network absence could not be verified: "
            f"{(inspected.stderr or inspected.stdout or '').strip()}"
        )

    def _remove_session_network(self, session_id: str) -> None:
        inspected = self._inspect_owned_network(session_id)
        if inspected is None:
            return
        network_id, containers = inspected
        if containers:
            raise BrowserBrokerError(
                "Docker session network still has attached containers; cleanup is unsafe"
            )
        removed = subprocess.run(
            [self.config.docker_command, "network", "rm", network_id],
            capture_output=True,
            text=True,
            timeout=DOCKER_REMOVE_TIMEOUT_SECONDS,
            check=False,
        )
        absent = self._network_is_absent(network_id)
        if removed.returncode != 0 and not absent:
            raise BrowserBrokerError(
                "Docker network removal failed: "
                f"{(removed.stderr or removed.stdout or '').strip()}"
            )
        if not absent:
            raise BrowserBrokerError("Docker network removal could not be verified")

    def _cleanup_failed_network_create(self, session_id: str) -> bool:
        try:
            discovered = self._inspect_owned_network(session_id)
            if discovered is None:
                return True
            _, containers = discovered
            if containers:
                return False
            self._remove_session_network(session_id)
        except Exception:
            return False
        return True

    def _ensure_session_network(self, session_id: str) -> str:
        existing = self._inspect_owned_network(session_id)
        if existing is not None:
            _, containers = existing
            if containers:
                raise BrowserBrokerError(
                    "Docker session network already has attached containers; startup is unsafe"
                )
            self._remove_session_network(session_id)

        network_name = self._network_name(session_id)
        labels = {
            **self._stable_ownership_labels(session_id),
            "tokimon.remote-browser-network-policy": self.config.network_policy,
        }
        command = [
            self.config.docker_command,
            "network",
            "create",
            "--driver",
            "bridge",
        ]
        for key, value in labels.items():
            command.extend(["--label", f"{key}={value}"])
        command.append(network_name)
        try:
            created = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=DOCKER_START_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception as exc:
            if not self._cleanup_failed_network_create(session_id):
                raise BrowserBrokerError(
                    "Docker session network creation failed and cleanup was ambiguous",
                    details={"cleanup_verified": False},
                ) from exc
            raise
        if created.returncode != 0:
            detail = (created.stderr or created.stdout or "").strip()
            if not self._cleanup_failed_network_create(session_id):
                raise BrowserBrokerError(
                    "Docker session network creation failed and cleanup was ambiguous: "
                    f"{detail}",
                    details={"cleanup_verified": False},
                )
            raise BrowserBrokerError(
                f"Docker session network creation failed: {detail}",
                details={"cleanup_verified": True},
            )
        try:
            inspected = self._inspect_owned_network(session_id)
        except Exception as exc:
            cleanup_verified = self._cleanup_failed_network_create(session_id)
            raise BrowserBrokerError(
                "Docker session network creation could not be verified"
                + (
                    " and cleanup was ambiguous"
                    if not cleanup_verified
                    else "; the verified empty network was removed"
                ),
                details={"cleanup_verified": cleanup_verified},
            ) from exc
        if inspected is None:
            raise BrowserBrokerError(
                "Docker session network disappeared before verification",
                details={"cleanup_verified": True},
            )
        _, containers = inspected
        if containers:
            raise BrowserBrokerError(
                "Docker session network acquired an unexpected attached container",
                details={"cleanup_verified": False},
            )
        return network_name

    def ensure_image(self) -> None:
        if not shutil.which(self.config.docker_command):
            raise BrowserBrokerError(f"Docker command not found: {self.config.docker_command}")
        inspect = subprocess.run(
            [self.config.docker_command, "image", "inspect", self.config.image_name],
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
        if inspect.returncode == 0:
            return
        self.build_image()

    def build_image(self) -> None:
        if not shutil.which(self.config.docker_command):
            raise BrowserBrokerError(f"Docker command not found: {self.config.docker_command}")
        if not IMAGE_CONTEXT_DIR.exists():
            raise BrowserBrokerError(f"Image context directory is missing: {IMAGE_CONTEXT_DIR}")
        docker_temp_root = runtime_workspace_root(self.config.workspace_root)
        with tempfile.TemporaryDirectory(
            prefix="tokimon-rb-image-context-",
            dir=docker_temp_root,
        ) as tmpdir:
            build_context = Path(tmpdir) / "image"
            validate_copy_source_tree(
                IMAGE_CONTEXT_DIR,
                label="Remote browser image context",
            )
            shutil.copytree(IMAGE_CONTEXT_DIR, build_context)
            build = subprocess.run(
                [
                    self.config.docker_command,
                    "build",
                    "-t",
                    self.config.image_name,
                    str(build_context),
                ],
                capture_output=True,
                text=True,
                timeout=DOCKER_BUILD_TIMEOUT_SECONDS,
                check=False,
            )
        if build.returncode != 0:
            detail = (build.stderr or build.stdout or "").strip()
            raise BrowserBrokerError(f"Docker image build failed: {detail}")

    def start(
        self,
        *,
        session_id: str,
        profile_dir: Path,
        cdp_port: int,
        novnc_backend_port: int,
        container_started: Callable[[str], None] | None = None,
    ) -> str:
        self.ensure_image()
        profile_dir.mkdir(parents=True, exist_ok=True)
        container_name = f"tokimon-rb-{self.config.installation_id}-{session_id}"
        with tempfile.NamedTemporaryFile(
            prefix="tokimon-rb-seccomp-",
            suffix=".json",
            dir=runtime_workspace_root(self.config.workspace_root),
            delete=False,
        ) as seccomp_file:
            seccomp_tmp_path = Path(seccomp_file.name)
            seccomp_file.write(SECCOMP_PROFILE_PATH.read_bytes())
        cid_fd, cid_name = tempfile.mkstemp(
            prefix="tokimon-rb-container-",
            suffix=".cid",
            dir=runtime_workspace_root(self.config.workspace_root),
        )
        os.close(cid_fd)
        cidfile_path = Path(cid_name)
        cidfile_path.unlink()
        try:
            network_name = self._ensure_session_network(session_id)
            cmd = [
                self.config.docker_command,
                "run",
                "--detach",
                "--rm",
                "--cidfile",
                str(cidfile_path),
                "--security-opt",
                f"seccomp={seccomp_tmp_path}",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "SYS_CHROOT",
                "--memory",
                "2g",
                "--pids-limit",
                "512",
                "--cpus",
                "2",
                "--network",
                network_name,
                "--tmpfs",
                "/tmp:size=512m,exec,mode=1777",
                *docker_user_args(),
                "--name",
                container_name,
                *[
                    argument
                    for key, value in self._stable_ownership_labels(session_id).items()
                    for argument in ("--label", f"{key}={value}")
                ],
                "--label",
                f"tokimon.remote-browser-network-policy={self.config.network_policy}",
                "-p",
                f"127.0.0.1:{cdp_port}:9223",
                "-p",
                f"127.0.0.1:{novnc_backend_port}:6080",
                "-v",
                f"{profile_dir}:/profile",
                "-e",
                "HOME=/tmp",
                "-e",
                f"SCREEN_RESOLUTION={self.config.screen}",
                "-e",
                f"VIEWPORT_SIZE={self.config.viewport}",
                "-e",
                "CLEAR_STALE_PROFILE_LOCKS=1",
                "-e",
                f"TOKIMON_RB_NETWORK_POLICY={self.config.network_policy}",
                self.config.image_name,
            ]
            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=DOCKER_START_TIMEOUT_SECONDS,
                    check=False,
                )
            except Exception:
                if not self._cleanup_failed_start(cidfile_path, session_id):
                    raise BrowserBrokerError(
                        "Docker session start failed and cleanup could not be verified",
                        details={"cleanup_verified": False},
                    )
                raise
            if process.returncode != 0:
                detail = (process.stderr or process.stdout or "").strip()
                if not self._cleanup_failed_start(cidfile_path, session_id):
                    raise BrowserBrokerError(
                        f"Docker session start failed and cleanup was ambiguous: {detail}",
                        details={"cleanup_verified": False},
                    )
                raise BrowserBrokerError(
                    f"Docker session start failed: {detail}",
                    details={"cleanup_verified": True},
                )
            try:
                container_id = cidfile_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                container_id = ""
            if not DOCKER_FULL_CONTAINER_ID_RE.fullmatch(container_id):
                self._cleanup_failed_start(cidfile_path, session_id)
                raise BrowserBrokerError(
                    "Docker session started without a verifiable container id",
                    details={"cleanup_verified": False},
                )
            if container_started is not None:
                try:
                    container_started(container_id)
                except Exception as exc:
                    cleanup_verified = self._cleanup_failed_start(cidfile_path, session_id)
                    raise BrowserBrokerError(
                        "Docker container identity checkpoint could not be persisted",
                        details={"cleanup_verified": cleanup_verified},
                    ) from exc
            return container_name
        finally:
            try:
                seccomp_tmp_path.unlink()
            except OSError:
                pass
            try:
                cidfile_path.unlink()
            except OSError:
                pass

    def _cleanup_failed_start(self, cidfile_path: Path, session_id: str) -> bool:
        cleanup_verified = True
        try:
            self._remove_start_container_from_cidfile(cidfile_path)
        except Exception:
            cleanup_verified = False
        try:
            self._remove_session_network(session_id)
        except Exception:
            cleanup_verified = False
        return cleanup_verified

    def _remove_start_container_from_cidfile(self, cidfile_path: Path) -> None:
        try:
            container_id = cidfile_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise BrowserBrokerError(
                "Docker start cleanup is ambiguous because no cidfile was produced"
            ) from exc
        if not container_id:
            raise BrowserBrokerError(
                "Docker start cleanup is ambiguous because the cidfile was empty"
            )
        if not DOCKER_FULL_CONTAINER_ID_RE.fullmatch(container_id):
            raise BrowserBrokerError("Docker cidfile contained an invalid container id")
        self._force_remove_container(container_id)

    def stop(self, session: BrowserSession) -> None:
        if not shutil.which(self.config.docker_command):
            raise BrowserBrokerError(f"Docker command not found: {self.config.docker_command}")
        container_id = self._owned_container_id(session)
        if container_id is not None:
            try:
                self._request_chrome_shutdown(container_id, session)
            except (OSError, subprocess.TimeoutExpired):
                pass
            try:
                wait = subprocess.run(
                    [self.config.docker_command, "wait", container_id],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if wait.returncode == 0 and self._container_is_absent(container_id):
                    container_id = None
            except subprocess.TimeoutExpired:
                pass
            if container_id is not None:
                try:
                    subprocess.run(
                        [self.config.docker_command, "stop", "--time", "5", container_id],
                        capture_output=True,
                        text=True,
                        timeout=8,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    pass
                self._force_remove_container(container_id)
        self._remove_session_network(session.id)

    def _owned_container_id(self, session: BrowserSession) -> str | None:
        if session.container_id is not None and not DOCKER_FULL_CONTAINER_ID_RE.fullmatch(session.container_id):
            raise BrowserBrokerError("Recorded browser container identity is malformed")
        inspected = subprocess.run(
            [
                self.config.docker_command,
                "container",
                "inspect",
                session.container_name,
            ],
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
        if inspected.returncode != 0:
            detail = f"{inspected.stderr}\n{inspected.stdout}".lower()
            if "no such container" in detail or "no such object" in detail:
                if session.container_id is not None and not self._container_is_absent(session.container_id):
                    raise BrowserBrokerError(
                        "Recorded browser container still exists under a different name"
                    )
                return None
            raise BrowserBrokerError(
                "Docker container identity could not be inspected safely: "
                f"{(inspected.stderr or inspected.stdout or '').strip()}"
            )
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as exc:
            raise BrowserBrokerError(
                "Docker container identity inspection returned invalid JSON"
            ) from exc
        if not isinstance(payload, list) or len(payload) != 1:
            raise BrowserBrokerError(
                "Docker container identity inspection was ambiguous"
            )
        record = payload[0]
        if not isinstance(record, Mapping):
            raise BrowserBrokerError(
                "Docker container identity inspection was malformed"
            )
        container_id = str(record.get("Id") or "")
        container_name = str(record.get("Name") or "").lstrip("/")
        config_payload = record.get("Config")
        labels = (
            config_payload.get("Labels")
            if isinstance(config_payload, Mapping)
            else None
        )
        expected_labels = self._stable_ownership_labels(session.id)
        if (
            session.container_id is None
            or not DOCKER_FULL_CONTAINER_ID_RE.fullmatch(session.container_id)
            or container_id != session.container_id
            or container_name != session.container_name
            or not isinstance(labels, Mapping)
            or any(labels.get(key) != value for key, value in expected_labels.items())
        ):
            raise BrowserBrokerError(
                "Docker container name now resolves to a foreign or ambiguous identity"
            )
        mounts = record.get("Mounts")
        profile_mounts = [
            mount
            for mount in mounts
            if isinstance(mount, Mapping)
            and str(mount.get("Destination") or "") == "/profile"
        ] if isinstance(mounts, list) else []
        if len(profile_mounts) != 1 or not isinstance(mounts, list) or len(mounts) != 1:
            raise BrowserBrokerError(
                "Docker container profile ownership could not be verified"
            )
        source = Path(str(profile_mounts[0].get("Source") or "")).resolve()
        if source != session.profile_dir.resolve():
            raise BrowserBrokerError(
                "Docker container profile mount does not match the owned browser session"
            )
        return container_id

    def _container_is_absent(self, container_name: str) -> bool:
        inspect = subprocess.run(
            [self.config.docker_command, "container", "inspect", container_name],
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
        if inspect.returncode == 0:
            return False
        detail = f"{inspect.stderr}\n{inspect.stdout}".lower()
        if "no such container" in detail or "no such object" in detail:
            return True
        raise BrowserBrokerError(
            "Docker container absence could not be verified: "
            f"{(inspect.stderr or inspect.stdout or '').strip()}"
        )

    def _force_remove_container(self, container_name: str) -> None:
        removed = subprocess.run(
            [self.config.docker_command, "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=DOCKER_REMOVE_TIMEOUT_SECONDS,
            check=False,
        )
        if removed.returncode == 0:
            return
        detail = f"{removed.stderr}\n{removed.stdout}".lower()
        if "no such container" in detail or "no such object" in detail:
            return
        if self._container_is_absent(container_name):
            return
        raise BrowserBrokerError(
            "Docker container removal failed: "
            f"{(removed.stderr or removed.stdout or '').strip()}"
        )

    def _request_chrome_shutdown(self, container_name: str, session: BrowserSession) -> None:
        if session.cdp_port:
            try:
                with CDPConnection.for_session(session) as cdp:
                    cdp.call("Browser.close")
                return
            except (BrowserBrokerError, OSError, TimeoutError):
                pass
        subprocess.run(
            [
                self.config.docker_command,
                "exec",
                container_name,
                "bash",
                "-lc",
                (
                    "pkill -TERM -f -- '[c]hrome.*--user-data-dir=/profile' || true; "
                    "for _ in $(seq 1 50); do "
                    "pgrep -f -- '[c]hrome.*--user-data-dir=/profile' >/dev/null || exit 0; "
                    "sleep 0.1; "
                    "done; "
                    "exit 1"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )


class BrowserController:
    def _reference_context(self, cdp: "CDPConnection") -> int:
        frame_tree = cdp.call("Page.getFrameTree")
        frame_id = frame_tree.get("frameTree", {}).get("frame", {}).get("id")
        if not isinstance(frame_id, str) or not frame_id:
            raise BrowserBrokerError("CDP frame tree is missing the main frame")
        world = cdp.call(
            "Page.createIsolatedWorld",
            {"frameId": frame_id, "worldName": "tokimon-browser-refs"},
        )
        context_id = world.get("executionContextId")
        if not isinstance(context_id, int):
            raise BrowserBrokerError("CDP reference world is missing its execution context")
        return context_id

    def wait_until_ready(self, session: BrowserSession, *, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        last_error = ""
        created_blank_target = False
        while time.monotonic() < deadline:
            try:
                fetch_json(f"http://127.0.0.1:{session.cdp_port}/json/version", timeout=2)
                targets = fetch_json(f"http://127.0.0.1:{session.cdp_port}/json", timeout=2)
                if isinstance(targets, list) and any(
                    isinstance(target, Mapping)
                    and target.get("type") == "page"
                    and target.get("webSocketDebuggerUrl")
                    for target in targets
                ):
                    return
                if isinstance(targets, list) and not created_blank_target:
                    self._create_blank_target(session)
                    created_blank_target = True
            except Exception as exc:  # noqa: BLE001 - readiness polling keeps the final error.
                last_error = str(exc)
            time.sleep(0.5)
        raise BrowserBrokerError(
            f"Browser CDP did not become ready on port {session.cdp_port}: {last_error}"
        )

    def _create_blank_target(self, session: BrowserSession) -> None:
        endpoint = f"http://127.0.0.1:{session.cdp_port}/json/new?about:blank"
        request = urllib.request.Request(endpoint, method="PUT")
        with urllib.request.urlopen(request, timeout=2) as response_obj:
            json.loads(response_obj.read().decode("utf-8"))

    def navigate(self, session: BrowserSession, url: str) -> dict[str, object]:
        last_error: BrowserBrokerError | None = None
        for _ in range(3):
            try:
                return self._navigate_once(session, url)
            except BrowserBrokerError as exc:
                last_error = exc
                time.sleep(0.5)
        assert last_error is not None
        raise last_error

    def _navigate_once(self, session: BrowserSession, url: str) -> dict[str, object]:
        normalized = normalize_url(url)
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Page.enable")
            cdp.call("Runtime.enable")
            cdp.call("Page.navigate", {"url": normalized})
            self._wait_for_document_ready(cdp)
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": "({url: location.href, title: document.title})",
                    "returnByValue": True,
                },
            )
        value = runtime_value(result)
        return {"url": value.get("url", normalized), "title": value.get("title", "")}

    def click(self, session: BrowserSession, ref_data: Mapping[str, object]) -> dict[str, object]:
        expression = element_action_expression(ref_data, "click")
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Runtime.enable")
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "contextId": self._reference_context(cdp),
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
        value = runtime_value(result)
        if not isinstance(value, dict):
            raise BrowserBrokerError("Click operation returned a non-object payload")
        return value

    def type_text(
        self,
        session: BrowserSession,
        ref_data: Mapping[str, object],
        *,
        text: str,
        clear: bool,
    ) -> dict[str, object]:
        expression = element_action_expression(
            ref_data,
            "type",
            text=text,
            clear=clear,
        )
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Runtime.enable")
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "contextId": self._reference_context(cdp),
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
        value = runtime_value(result)
        if not isinstance(value, dict):
            raise BrowserBrokerError("Type operation returned a non-object payload")
        return value

    def upload_file(
        self,
        session: BrowserSession,
        ref_data: Mapping[str, object],
        *,
        file_path: str,
    ) -> dict[str, object]:
        expression = file_input_object_expression(ref_data)
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Runtime.enable")
            cdp.call("DOM.enable")
            evaluation = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "contextId": self._reference_context(cdp),
                    "objectGroup": "tokimon-file-upload",
                    "returnByValue": False,
                    "awaitPromise": True,
                },
            )
            object_id = runtime_object_id(evaluation)
            try:
                cdp.call(
                    "DOM.setFileInputFiles",
                    {
                        "objectId": object_id,
                        "files": [file_path],
                    },
                )
                result = cdp.call(
                    "Runtime.callFunctionOn",
                    {
                        "objectId": object_id,
                        "functionDeclaration": """
function() {
  const file = this.files && this.files.length ? this.files[0] : null;
  this.dispatchEvent(new Event('input', {bubbles: true}));
  this.dispatchEvent(new Event('change', {bubbles: true}));
  return {
    status: 'uploaded',
    file_count: this.files ? this.files.length : 0,
    file_name: file ? file.name : ''
  };
}
""".strip(),
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                )
            finally:
                try:
                    cdp.call("Runtime.releaseObjectGroup", {"objectGroup": "tokimon-file-upload"})
                except BrowserBrokerError:
                    pass
        value = runtime_value(result)
        if not isinstance(value, dict):
            raise BrowserBrokerError("Upload operation returned a non-object payload")
        return value

    def backend_request(
        self,
        session: BrowserSession,
        *,
        method: str,
        path: str,
        body: Mapping[str, Any] | None,
        max_response_chars: int,
        timeout_ms: int = DEFAULT_BACKEND_REQUEST_TIMEOUT_MS,
    ) -> dict[str, object]:
        if timeout_ms <= 0 or timeout_ms > MAX_BACKEND_REQUEST_TIMEOUT_MS:
            raise BrowserBrokerError(
                f"timeout_ms must be between 1 and {MAX_BACKEND_REQUEST_TIMEOUT_MS}",
                code=ERROR_INVALID_ARGUMENT,
            )
        expression = backend_request_expression(
            method=method,
            path=path,
            body=body,
            max_response_chars=max_response_chars,
            timeout_ms=timeout_ms,
        )
        with CDPConnection.for_session(session, timeout=max(1.0, timeout_ms / 1000)) as cdp:
            cdp.call("Runtime.enable")
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
        value = runtime_value(result)
        if not isinstance(value, dict):
            raise BrowserBrokerError("Backend request operation returned a non-object payload")
        return value

    def press_key(self, session: BrowserSession, key: str) -> dict[str, object]:
        expression = dispatch_key_expression(key)
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Runtime.enable")
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
        value = runtime_value(result)
        return value if isinstance(value, dict) else {"key": key}

    def wait(
        self,
        session: BrowserSession,
        *,
        seconds: float | None = None,
        text: str | None = None,
        timeout_ms: int = 5000,
    ) -> dict[str, object]:
        if seconds is not None:
            if not math.isfinite(seconds) or seconds < 0 or seconds > MAX_WAIT_SECONDS:
                raise BrowserBrokerError(
                    f"seconds must be finite and between 0 and {MAX_WAIT_SECONDS:g}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            time.sleep(seconds)
            return {"status": "waited", "seconds": seconds}
        if text:
            if timeout_ms <= 0 or timeout_ms > MAX_WAIT_TIMEOUT_MS:
                raise BrowserBrokerError(
                    f"timeout_ms must be between 1 and {MAX_WAIT_TIMEOUT_MS}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            deadline = time.monotonic() + (timeout_ms / 1000)
            while time.monotonic() < deadline:
                with CDPConnection.for_session(session) as cdp:
                    cdp.call("Runtime.enable")
                    result = cdp.call(
                        "Runtime.evaluate",
                        {
                            "expression": (
                                "document.body && "
                                f"document.body.innerText.includes({json.dumps(text)})"
                            ),
                            "returnByValue": True,
                        },
                    )
                if runtime_value(result) is True:
                    return {"status": "text_found", "text": text}
                time.sleep(0.25)
            raise BrowserBrokerError(f"Timed out waiting for text: {text}")
        return {"status": "noop"}

    def new_tab(self, session: BrowserSession, url: str | None = None) -> dict[str, object]:
        target_url = normalize_url(url) if url else "about:blank"
        with CDPConnection.for_session(session) as cdp:
            result = cdp.call("Target.createTarget", {"url": target_url})
        tab_id = str(result.get("targetId") or "")
        if not tab_id:
            raise BrowserBrokerError("CDP did not return a targetId for the new tab")
        session.active_tab_id = tab_id
        return {"tab_id": tab_id, "active_tab_id": tab_id, "url": target_url}

    def switch_tab(self, session: BrowserSession, tab_id: str) -> dict[str, object]:
        target = self._find_target(session, tab_id)
        if target is None:
            raise BrowserBrokerError(f"Tab not found: {tab_id}", code=ERROR_TAB_NOT_FOUND)
        session.active_tab_id = tab_id
        return {
            "tab_id": tab_id,
            "active_tab_id": tab_id,
            "url": target.get("url", ""),
            "title": target.get("title", ""),
        }

    def close_tab(self, session: BrowserSession, tab_id: str | None = None) -> dict[str, object]:
        target_id = tab_id or session.active_tab_id
        if not target_id:
            target = self._first_page_target(session)
            if target is None:
                raise BrowserBrokerError("No active tab is available", code=ERROR_TAB_NOT_FOUND)
            target_id = str(target.get("id") or "")
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Target.closeTarget", {"targetId": target_id})
        remaining = [target for target in self._page_targets(session) if target.get("id") != target_id]
        next_target = remaining[0] if remaining else None
        session.active_tab_id = str(next_target.get("id")) if next_target else None
        return {
            "closed_tab_id": target_id,
            "active_tab_id": session.active_tab_id,
        }

    def fingerprint_baseline(self, session: BrowserSession) -> dict[str, object]:
        script = """
(() => {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
  return {
    webdriver: navigator.webdriver === true,
    user_agent: navigator.userAgent || '',
    languages: Array.from(navigator.languages || []),
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || '',
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      device_pixel_ratio: window.devicePixelRatio
    },
    webgl_present: !!gl
  };
})()
""".strip()
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Runtime.enable")
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": script,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
        value = runtime_value(result)
        if not isinstance(value, dict):
            raise BrowserBrokerError("Fingerprint baseline returned a non-object payload")
        return {
            "signals": value,
            "checks": {
                "webdriver_not_true": value.get("webdriver") is not True,
                "user_agent_present": bool(value.get("user_agent")),
                "timezone_present": bool(value.get("timezone")),
                "viewport_present": bool(value.get("viewport")),
                "webgl_present": bool(value.get("webgl_present")),
            },
            "claim": "baseline self-check only; does not claim third-party undetectability",
        }

    def _page_targets(self, session: BrowserSession) -> list[Mapping[str, Any]]:
        targets = fetch_json(f"http://127.0.0.1:{session.cdp_port}/json", timeout=5)
        if not isinstance(targets, list):
            return []
        return [
            target
            for target in targets
            if isinstance(target, Mapping) and target.get("type") == "page"
        ]

    def _first_page_target(self, session: BrowserSession) -> Mapping[str, Any] | None:
        targets = self._page_targets(session)
        return targets[0] if targets else None

    def _find_target(
        self,
        session: BrowserSession,
        tab_id: str,
    ) -> Mapping[str, Any] | None:
        for target in self._page_targets(session):
            if target.get("id") == tab_id:
                return target
        return None

    def snapshot(
        self,
        session: BrowserSession,
        *,
        max_text_chars: int = 20000,
        max_elements: int = 200,
    ) -> dict[str, object]:
        if max_text_chars <= 0 or max_text_chars > MAX_SNAPSHOT_TEXT_CHARS:
            raise BrowserBrokerError(
                f"max_text_chars must be between 1 and {MAX_SNAPSHOT_TEXT_CHARS}",
                code=ERROR_INVALID_ARGUMENT,
            )
        if max_elements <= 0 or max_elements > MAX_SNAPSHOT_ELEMENTS:
            raise BrowserBrokerError(
                f"max_elements must be between 1 and {MAX_SNAPSHOT_ELEMENTS}",
                code=ERROR_INVALID_ARGUMENT,
            )
        last_error: BrowserBrokerError | None = None
        for _ in range(3):
            try:
                return self._snapshot_once(
                    session,
                    max_text_chars=max_text_chars,
                    max_elements=max_elements,
                )
            except BrowserBrokerError as exc:
                last_error = exc
                time.sleep(0.5)
        assert last_error is not None
        raise last_error

    def _snapshot_once(
        self,
        session: BrowserSession,
        *,
        max_text_chars: int,
        max_elements: int,
    ) -> dict[str, object]:
        script = snapshot_expression(
            max_text_chars=max_text_chars,
            max_elements=max_elements,
            snapshot_id=f"{session.id}:{session.ref_epoch + 1}",
        )
        with CDPConnection.for_session(session) as cdp:
            cdp.call("Runtime.enable")
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": script,
                    "contextId": self._reference_context(cdp),
                    "returnByValue": True,
                    "awaitPromise": True,
                },
            )
        value = runtime_value(result)
        if not isinstance(value, dict):
            raise BrowserBrokerError("Snapshot script returned a non-object payload")
        return value

    def evaluate(
        self,
        session: BrowserSession,
        *,
        expression: str,
        await_promise: bool = True,
    ) -> Any:
        script = str(expression or "").strip()
        if not script:
            raise BrowserBrokerError("expression is required")
        last_error: BrowserBrokerError | None = None
        for _ in range(3):
            try:
                with CDPConnection.for_session(session) as cdp:
                    cdp.call("Runtime.enable")
                    result = cdp.call(
                        "Runtime.evaluate",
                        {
                            "expression": script,
                            "returnByValue": True,
                            "awaitPromise": bool(await_promise),
                        },
                    )
                return runtime_value(result)
            except BrowserBrokerError as exc:
                last_error = exc
                time.sleep(0.5)
        assert last_error is not None
        raise last_error

    def isolated_evaluate(
        self, session: BrowserSession, expression: str, *, timeout: float = 10,
    ) -> Any:
        """One bounded read in the broker world; no retries or page-owned globals."""
        deadline = time.monotonic() + timeout
        connection = CDPConnection.for_session(session, timeout=timeout)
        connection.timeout = _remaining_deadline_seconds(deadline, operation="page observation")
        with connection as cdp:
            def call(method: str, params: Mapping[str, Any] | None = None) -> dict:
                cdp.timeout = _remaining_deadline_seconds(deadline, operation="page observation")
                return cdp.call(method, params)
            frame = call("Page.getFrameTree").get("frameTree", {}).get("frame", {}).get("id")
            if not isinstance(frame, str) or not frame:
                raise BrowserBrokerError("Page observation has no main frame")
            world = call("Page.createIsolatedWorld", {"frameId": frame, "worldName": "tokimon-browser-refs"})
            context = world.get("executionContextId")
            if not isinstance(context, int):
                raise BrowserBrokerError("Page observation has no isolated context")
            result = call("Runtime.evaluate", {"expression": expression, "contextId": context,
                                               "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            detail = str(result["exceptionDetails"])
            for code in ("STALE_REF", "SCOPE_NOT_FOUND_OR_AMBIGUOUS", "SCOPE_NOT_VISIBLE",
                         "ARTIFACT_TOO_LARGE", "ARTIFACT_UNSUPPORTED"):
                if code in detail:
                    hint = ""
                    if code == "ARTIFACT_UNSUPPORTED":
                        hint = " Use a current-page same-origin download link without redirects, or use the site's normal download flow."
                    raise BrowserBrokerError(code + hint, code=code)
            raise BrowserBrokerError("Page observation failed; verify the selector and current document")
        return runtime_value(result)

    def screenshot(
        self,
        session: BrowserSession,
        *,
        image_format: str,
        quality: int | None = None,
    ) -> str:
        last_error: BrowserBrokerError | None = None
        for _ in range(3):
            try:
                with CDPConnection.for_session(session) as cdp:
                    cdp.call("Page.enable")
                    params: dict[str, object] = {"format": image_format}
                    if image_format == "jpeg" and quality is not None:
                        params["quality"] = quality
                    result = cdp.call("Page.captureScreenshot", params)
                data = result.get("data")
                if not isinstance(data, str) or not data:
                    raise BrowserBrokerError("CDP screenshot response is missing image data")
                return data
            except BrowserBrokerError as exc:
                last_error = exc
                time.sleep(0.5)
        assert last_error is not None
        raise last_error

    def _wait_for_document_ready(
        self,
        cdp: "CDPConnection",
        *,
        timeout_sec: float = 20.0,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            result = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": "document.readyState",
                    "returnByValue": True,
                },
            )
            if runtime_value(result) in {"interactive", "complete"}:
                return
            time.sleep(0.25)


def normalize_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        raise BrowserBrokerError("url is required", code=ERROR_INVALID_ARGUMENT)
    try:
        parsed = urllib.parse.urlsplit(stripped)
        if not parsed.scheme and "://" not in stripped:
            stripped = f"https://{stripped}"
            parsed = urllib.parse.urlsplit(stripped)
        if parsed.scheme.lower() in {"http", "https"}:
            if not parsed.hostname or any(char.isspace() for char in parsed.netloc):
                raise ValueError("missing or invalid host")
            parsed.port
    except ValueError as exc:
        raise BrowserBrokerError(
            "URL contains an invalid host or port", code=ERROR_INVALID_ARGUMENT,
        ) from exc
    return stripped


def fetch_json(url: str, *, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response_obj:
        return json.loads(response_obj.read().decode("utf-8"))


def runtime_value(result: Mapping[str, Any]) -> Any:
    if "exceptionDetails" in result:
        raise_runtime_exception(result)
    result_obj = result.get("result")
    if not isinstance(result_obj, Mapping):
        return None
    return result_obj.get("value")


def runtime_object_id(result: Mapping[str, Any]) -> str:
    if "exceptionDetails" in result:
        raise_runtime_exception(result)
    result_obj = result.get("result")
    if not isinstance(result_obj, Mapping):
        raise BrowserBrokerError("CDP evaluation did not return an object")
    object_id = result_obj.get("objectId")
    if not isinstance(object_id, str) or not object_id:
        raise BrowserBrokerError("CDP evaluation did not return an object id")
    return object_id


def raise_runtime_exception(result: Mapping[str, Any]) -> None:
    detail = str(result["exceptionDetails"])
    code = ERROR_BROKER
    if "STALE_REF" in detail:
        code = ERROR_STALE_REF
    elif "UPLOAD_REF_NOT_FILE_INPUT" in detail:
        code = ERROR_INVALID_ARGUMENT
    raise BrowserBrokerError(f"CDP evaluation failed: {detail}", code=code)


SNAPSHOT_ELEMENT_SELECTOR = "a,button,input,textarea,select,[role],[aria-label],[placeholder]"
SNAPSHOT_ELEMENT_DESCRIPTION = f"""
  const fieldLimit = {MAX_SNAPSHOT_FIELD_CHARS};
  const clean = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
  const bounded = (value) => clean(value).slice(0, fieldLimit);
  const describeElement = (el) => ({{
    tag: bounded(el.tagName.toLowerCase()),
    role: bounded(el.getAttribute('role')),
    text: clean(el.innerText || el.textContent).slice(0, 240),
    aria_label: bounded(el.getAttribute('aria-label')),
    placeholder: bounded(el.getAttribute('placeholder')),
    type: bounded(el.getAttribute('type')),
    name: bounded(el.getAttribute('name')),
    id: bounded(el.id),
    href: bounded(el instanceof HTMLAnchorElement ? el.href : '')
  }});
""".strip()


def snapshot_expression(*, max_text_chars: int, max_elements: int, snapshot_id: str) -> str:
    return f"""
(() => {{
  const textLimit = {max_text_chars};
  const elementLimit = {max_elements};
  {SNAPSHOT_ELEMENT_DESCRIPTION}
  const selector = '{SNAPSHOT_ELEMENT_SELECTOR}';
  const nodes = Array.from(document.querySelectorAll(selector)).slice(0, elementLimit);
  const elements = nodes.map((el, index) => ({{index, ...describeElement(el)}}));
  globalThis.__tokimonSnapshot = {{
    id: {json.dumps(snapshot_id)}, document, url: location.href, nodes,
    descriptions: nodes.map(el => JSON.stringify(describeElement(el)))
  }};
  return {{
    url: bounded(location.href),
    title: bounded(document.title),
    text: clean(document.body ? document.body.innerText : '').slice(0, textLimit),
    elements
  }};
}})()
""".strip()


def normalize_snapshot_payload(
    payload: Mapping[str, object],
    *,
    max_text_chars: int,
    max_elements: int,
) -> dict[str, object]:
    """Bound and whitelist page-derived snapshot data at the broker boundary."""

    def bounded_string(value: object, limit: int) -> str:
        return value[:limit] if isinstance(value, str) else ""

    normalized: dict[str, object] = {
        "url": bounded_string(payload.get("url"), MAX_SNAPSHOT_FIELD_CHARS),
        "title": bounded_string(payload.get("title"), MAX_SNAPSHOT_FIELD_CHARS),
        "text": bounded_string(payload.get("text"), max_text_chars),
        "elements": [],
    }
    raw_elements = payload.get("elements")
    if not isinstance(raw_elements, list):
        return normalized
    fields = (
        "tag",
        "role",
        "text",
        "aria_label",
        "placeholder",
        "type",
        "name",
        "id",
        "href",
    )
    elements: list[dict[str, object]] = []
    for item in raw_elements[:max_elements]:
        if not isinstance(item, Mapping):
            continue
        normalized_item: dict[str, object] = {"index": len(elements)}
        for field in fields:
            normalized_item[field] = bounded_string(
                item.get(field),
                240 if field == "text" else MAX_SNAPSHOT_FIELD_CHARS,
            )
        elements.append(normalized_item)
    normalized["elements"] = elements
    return normalized


def element_ref_expression(ref_data: Mapping[str, object]) -> str:
    return f"""
  {SNAPSHOT_ELEMENT_DESCRIPTION}
  const snapshot = globalThis.__tokimonSnapshot;
  const index = {int(ref_data.get("index", -1))};
  const el = snapshot && snapshot.nodes[index];
  if (!snapshot || snapshot.id !== {json.dumps(ref_data.get("snapshot_id"))}
      || snapshot.document !== document || snapshot.url !== location.href
      || !el || !el.isConnected || el.ownerDocument !== document
      || document.querySelectorAll('{SNAPSHOT_ELEMENT_SELECTOR}')[index] !== el
      || JSON.stringify(describeElement(el)) !== snapshot.descriptions[index]) {{
    throw new Error('STALE_REF');
  }}
""".strip()


def element_action_expression(
    ref_data: Mapping[str, object],
    action: str,
    *,
    text: str = "",
    clear: bool = False,
) -> str:
    text_json = json.dumps(text)
    clear_json = "true" if clear else "false"
    return f"""
(() => {{
  {element_ref_expression(ref_data)}
  el.scrollIntoView({{block: 'center', inline: 'center'}});
  if ({json.dumps(action)} === 'click') {{
    el.click();
    return {{status: 'clicked'}};
  }}
  if ({json.dumps(action)} === 'type') {{
    el.focus();
    const value = {text_json};
    if ('value' in el) {{
      if ({clear_json}) {{
        el.value = '';
      }}
      el.value = String(el.value || '') + value;
      el.dispatchEvent(new Event('input', {{bubbles: true}}));
      el.dispatchEvent(new Event('change', {{bubbles: true}}));
    }} else {{
      el.textContent = ({clear_json} ? '' : el.textContent || '') + value;
      el.dispatchEvent(new InputEvent('input', {{bubbles: true, data: value}}));
    }}
    return {{status: 'typed', length: value.length}};
  }}
  throw new Error('Unsupported action');
}})()
""".strip()


def file_input_object_expression(ref_data: Mapping[str, object]) -> str:
    return f"""
(() => {{
  {element_ref_expression(ref_data)}
  if (!(el instanceof HTMLInputElement) || String(el.type || '').toLowerCase() !== 'file') {{
    throw new Error('UPLOAD_REF_NOT_FILE_INPUT');
  }}
  return el;
}})()
""".strip()


def backend_request_expression(
    *,
    method: str,
    path: str,
    body: Mapping[str, Any] | None,
    max_response_chars: int,
    timeout_ms: int,
) -> str:
    body_json = json.dumps(body, ensure_ascii=False) if body is not None else None
    return f"""
(async () => {{
  try {{
    const method = {json.dumps(method)};
    const path = {json.dumps(path)};
    const maxChars = {int(max_response_chars)};
    const timeoutMs = {int(timeout_ms)};
    const readBoundedText = async (response, limit) => {{
      if (!response.body) {{
        return {{text: '', truncated: false}};
      }}
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let text = '';
      let truncated = false;
      while (true) {{
        const item = await reader.read();
        if (item.done) {{
          text += decoder.decode();
          break;
        }}
        const piece = decoder.decode(item.value, {{stream: true}});
        const remaining = Math.max(0, limit - text.length);
        if (piece.length > remaining) {{
          text += piece.slice(0, remaining);
          truncated = true;
          await reader.cancel();
          break;
        }}
        text += piece;
      }}
      return {{text: text.slice(0, limit), truncated: truncated || text.length > limit}};
    }};
    const timedFetchAndRead = async (url, options, limit) => {{
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {{
        const response = await fetch(url, {{...options, signal: controller.signal}});
        const body = await readBoundedText(response, limit);
        return {{response, ...body}};
      }} finally {{
        clearTimeout(timer);
      }}
    }};
    const requestUrl = new URL(path, location.origin);
    if (requestUrl.origin !== location.origin) {{
      throw new Error('BACKEND_REQUEST_CROSS_ORIGIN');
    }}
    const headers = {{Accept: 'application/json'}};
    let requestBody = null;
    const encodedBody = {json.dumps(body_json)};
    if (encodedBody !== null) {{
      headers['Content-Type'] = 'application/json';
      requestBody = encodedBody;
    }}
    const requestResult = await timedFetchAndRead(requestUrl.href, {{
      method,
      credentials: 'include',
      headers,
      body: requestBody
    }}, maxChars);
    const response = requestResult.response;
    return {{
      status: response.status,
      status_text: response.statusText,
      text: requestResult.text,
      truncated: requestResult.truncated,
      url_path: requestUrl.pathname + requestUrl.search
    }};
  }} catch (error) {{
    return {{status: 0, error: String(error)}};
  }}
}})()
""".strip()


def dispatch_key_expression(key: str) -> str:
    key_json = json.dumps(key)
    return f"""
(() => {{
  const key = {key_json};
  const target = document.activeElement || document.body;
  for (const type of ['keydown', 'keyup']) {{
    target.dispatchEvent(new KeyboardEvent(type, {{key, bubbles: true}}));
  }}
  return {{status: 'pressed', key}};
}})()
""".strip()


class CDPConnection:
    def __init__(self, ws_url: str, *, timeout: float = 10.0) -> None:
        self.ws_url = ws_url
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._recv_buffer = bytearray()
        self.next_id = 0

    @classmethod
    def for_session(cls, session: BrowserSession, *, timeout: float = 10.0) -> "CDPConnection":
        try:
            targets = fetch_json(f"http://127.0.0.1:{session.cdp_port}/json", timeout=5)
        except Exception as exc:  # noqa: BLE001 - convert transport failures to tool errors.
            raise BrowserBrokerError(
                f"Failed to read CDP targets on port {session.cdp_port}: {exc}"
            ) from exc
        if not isinstance(targets, list):
            raise BrowserBrokerError("CDP /json endpoint returned a non-list payload")
        page_target = None
        if session.active_tab_id:
            for target in targets:
                if (
                    isinstance(target, Mapping)
                    and target.get("type") == "page"
                    and target.get("id") == session.active_tab_id
                ):
                    page_target = target
                    break
        if page_target is None:
            for target in targets:
                if isinstance(target, Mapping) and target.get("type") == "page":
                    page_target = target
                    session.active_tab_id = str(target.get("id") or "") or None
                    break
        if page_target is None and targets and isinstance(targets[0], Mapping):
            page_target = targets[0]
        if page_target is None:
            raise BrowserBrokerError("No CDP page target is available")
        ws_url = str(page_target.get("webSocketDebuggerUrl") or "")
        if not ws_url:
            raise BrowserBrokerError("CDP target is missing webSocketDebuggerUrl")
        parsed = urllib.parse.urlparse(ws_url)
        host_ws_url = urllib.parse.urlunparse(
            (
                "ws",
                f"127.0.0.1:{session.cdp_port}",
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
        return cls(host_ws_url, timeout=timeout)

    def __enter__(self) -> "CDPConnection":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def connect(self) -> None:
        parsed = urllib.parse.urlparse(self.ws_url)
        if parsed.scheme != "ws":
            raise BrowserBrokerError(f"Only ws:// CDP endpoints are supported: {self.ws_url}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        deadline = time.monotonic() + self.timeout
        try:
            sock = socket.create_connection(
                (host, port),
                timeout=_remaining_deadline_seconds(
                    deadline,
                    operation="CDP WebSocket handshake",
                ),
            )
        except OSError as exc:
            raise BrowserBrokerError(f"Failed to connect to CDP WebSocket: {exc}") from exc
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        response_bytes = bytearray()
        try:
            sock.settimeout(
                _remaining_deadline_seconds(
                    deadline,
                    operation="CDP WebSocket handshake",
                )
            )
            sock.sendall(request.encode("ascii"))
            while b"\r\n\r\n" not in response_bytes:
                if len(response_bytes) >= MAX_CDP_HANDSHAKE_BYTES:
                    raise BrowserBrokerError(
                        "CDP WebSocket handshake exceeds "
                        f"{MAX_CDP_HANDSHAKE_BYTES} bytes"
                    )
                sock.settimeout(
                    _remaining_deadline_seconds(
                        deadline,
                        operation="CDP WebSocket handshake",
                    )
                )
                chunk = sock.recv(
                    min(
                        4096,
                        MAX_CDP_HANDSHAKE_BYTES + 1 - len(response_bytes),
                    )
                )
                if not chunk:
                    break
                response_bytes.extend(chunk)
            separator_index = response_bytes.find(b"\r\n\r\n")
            if separator_index < 0:
                if len(response_bytes) > MAX_CDP_HANDSHAKE_BYTES:
                    raise BrowserBrokerError(
                        "CDP WebSocket handshake exceeds "
                        f"{MAX_CDP_HANDSHAKE_BYTES} bytes"
                    )
                raise BrowserBrokerError(
                    "CDP WebSocket handshake closed before the upgrade completed"
                )
            header_end = separator_index + 4
            if header_end > MAX_CDP_HANDSHAKE_BYTES:
                raise BrowserBrokerError(
                    f"CDP WebSocket handshake exceeds {MAX_CDP_HANDSHAKE_BYTES} bytes"
                )
        except BrowserBrokerError:
            sock.close()
            raise
        except TimeoutError as exc:
            sock.close()
            raise BrowserBrokerError("CDP WebSocket handshake timed out") from exc
        except OSError as exc:
            sock.close()
            raise BrowserBrokerError(f"CDP WebSocket handshake failed: {exc}") from exc
        header_bytes = bytes(response_bytes[:header_end])
        header = header_bytes.decode("latin1", errors="replace")
        status_line = header.splitlines()[0] if header.splitlines() else "<empty response>"
        if " 101 " not in status_line:
            sock.close()
            raise BrowserBrokerError(f"CDP WebSocket handshake failed: {status_line}")
        self._recv_buffer.clear()
        self._recv_buffer.extend(response_bytes[header_end:])
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
                self._recv_buffer.clear()

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.sock is None:
            raise BrowserBrokerError("CDP connection is not open")
        self.next_id += 1
        command_id = self.next_id
        deadline = time.monotonic() + self.timeout
        try:
            self.sock.settimeout(
                _remaining_deadline_seconds(
                    deadline,
                    operation="CDP command response",
                )
            )
            self._send_json({"id": command_id, "method": method, "params": dict(params or {})})
        except BrowserBrokerError:
            raise
        except OSError as exc:
            raise BrowserBrokerError(f"Failed to send CDP command: {exc}") from exc
        payload = self._recv_json(expected_id=command_id, deadline=deadline)
        if "error" in payload:
            raise BrowserBrokerError(f"CDP command failed: {payload['error']}")
        result = payload.get("result")
        return result if isinstance(result, dict) else {}

    def _send_json(self, payload: Mapping[str, Any]) -> None:
        assert self.sock is not None
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
        self.sock.sendall(bytes(header) + mask + masked)

    def _recv_json(self, *, expected_id: int, deadline: float) -> dict[str, Any]:
        unmatched_messages = 0
        while True:
            _remaining_deadline_seconds(deadline, operation="CDP command response")
            payload, opcode = self._recv_frame(deadline=deadline)
            _remaining_deadline_seconds(deadline, operation="CDP command response")
            if opcode == 0x8:
                raise BrowserBrokerError("CDP WebSocket closed")
            if opcode == 0x9:
                self._send_control_frame(0xA, payload, deadline=deadline)
                continue
            if opcode in {0x1, 0x2}:
                data = json.loads(payload.decode("utf-8"))
                if isinstance(data, dict) and data.get("id") == expected_id:
                    return data
                unmatched_messages += 1
                if unmatched_messages >= MAX_CDP_UNMATCHED_MESSAGES:
                    raise BrowserBrokerError(
                        "CDP command response exceeded the unmatched-message limit "
                        f"of {MAX_CDP_UNMATCHED_MESSAGES}"
                    )

    def _recv_frame(self, *, deadline: float | None = None) -> tuple[bytes, int]:
        assert self.sock is not None
        first_two = self._recv_exact(2, deadline=deadline)
        first, second = first_two[0], first_two[1]
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2, deadline=deadline))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8, deadline=deadline))[0]
        if length > MAX_CDP_FRAME_BYTES:
            raise BrowserBrokerError(
                f"CDP WebSocket frame exceeds {MAX_CDP_FRAME_BYTES} bytes"
            )
        mask = self._recv_exact(4, deadline=deadline) if masked else b""
        payload = self._recv_exact(length, deadline=deadline) if length else b""
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return payload, opcode

    def _recv_exact(self, length: int, *, deadline: float | None = None) -> bytes:
        assert self.sock is not None
        chunks: list[bytes] = []
        remaining = length
        if self._recv_buffer and remaining:
            buffered_length = min(remaining, len(self._recv_buffer))
            chunks.append(bytes(self._recv_buffer[:buffered_length]))
            del self._recv_buffer[:buffered_length]
            remaining -= buffered_length
        while remaining > 0:
            if deadline is not None:
                self.sock.settimeout(
                    _remaining_deadline_seconds(
                        deadline,
                        operation="CDP command response",
                    )
                )
            try:
                chunk = self.sock.recv(remaining)
            except TimeoutError as exc:
                raise BrowserBrokerError("CDP command response timed out") from exc
            except OSError as exc:
                raise BrowserBrokerError(f"Socket read failed: {exc}") from exc
            if not chunk:
                raise BrowserBrokerError("Socket closed before expected bytes arrived")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _send_control_frame(
        self,
        opcode: int,
        payload: bytes,
        *,
        deadline: float | None = None,
    ) -> None:
        assert self.sock is not None
        if len(payload) > 125:
            payload = payload[:125]
        header = bytearray([0x80 | opcode, 0x80 | len(payload)])
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        if deadline is not None:
            self.sock.settimeout(
                _remaining_deadline_seconds(
                    deadline,
                    operation="CDP command response",
                )
            )
        self.sock.sendall(bytes(header) + mask + masked)


def _remaining_deadline_seconds(deadline: float, *, operation: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BrowserBrokerError(f"{operation} timed out")
    return remaining


def bounded_content_length(raw_value: str | None, *, maximum: int) -> int:
    try:
        length = int(raw_value or "0")
    except (TypeError, ValueError) as exc:
        raise BrowserBrokerError(
            "Content-Length must be a non-negative integer",
            code=ERROR_INVALID_ARGUMENT,
        ) from exc
    if length < 0:
        raise BrowserBrokerError(
            "Content-Length must be a non-negative integer",
            code=ERROR_INVALID_ARGUMENT,
        )
    if length > maximum:
        raise BrowserBrokerError(
            f"Request body exceeds {maximum} bytes",
            code=ERROR_INVALID_ARGUMENT,
            details={"request_too_large": True, "maximum_bytes": maximum},
        )
    return length


def read_http_body_with_deadline(
    handler: BaseHTTPRequestHandler,
    length: int,
    *,
    timeout_seconds: float = HTTP_BODY_READ_TIMEOUT_SECONDS,
) -> bytes:
    if length == 0:
        return b""
    previous_timeout = handler.connection.gettimeout()
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    remaining = length
    try:
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError("Request body read deadline expired")
            handler.connection.settimeout(timeout)
            try:
                chunk = handler.rfile.read(min(remaining, 64 * 1024))
            except socket.timeout as exc:
                raise TimeoutError("Request body read deadline expired") from exc
            if not chunk:
                raise BrowserBrokerError(
                    "Request body ended before Content-Length bytes were received",
                    code=ERROR_INVALID_ARGUMENT,
                )
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        handler.connection.settimeout(previous_timeout)
    return b"".join(chunks)


def lifecycle_serialized(method: Any) -> Any:
    """Serialize browser admission and lifecycle changes for one MCP session key."""

    @functools.wraps(method)
    def wrapped(
        self: "RemoteBrowserBroker",
        mcp_session_key: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._admission():
            with self._lifecycle_lock_for_key(mcp_session_key):
                return method(self, mcp_session_key, *args, **kwargs)

    return wrapped


def admission_serialized(method: Any) -> Any:
    """Prevent new browser admission from crossing broker shutdown."""

    @functools.wraps(method)
    def wrapped(self: "RemoteBrowserBroker", *args: Any, **kwargs: Any) -> Any:
        with self._admission():
            return method(self, *args, **kwargs)

    return wrapped


def template_operation_serialized(method: Any) -> Any:
    """Serialize one template's filesystem mutation across broker processes."""

    @functools.wraps(method)
    def wrapped(self: "RemoteBrowserBroker", *args: Any, **kwargs: Any) -> Any:
        with self._admission():
            template_name = kwargs.get("template_name")
            if not isinstance(template_name, str):
                raise BrowserBrokerError(
                    "template_name is required",
                    code=ERROR_INVALID_ARGUMENT,
                )
            with self._template_operation(template_name):
                return method(self, *args, **kwargs)

    return wrapped


def optional_template_operation_serialized(method: Any) -> Any:
    """Hold template filesystem ownership while an optional clone is admitted."""

    @functools.wraps(method)
    def wrapped(self: "RemoteBrowserBroker", *args: Any, **kwargs: Any) -> Any:
        template_name = kwargs.get("template_name")
        if template_name is None:
            return method(self, *args, **kwargs)
        if not isinstance(template_name, str):
            raise BrowserBrokerError(
                "template_name must be a string",
                code=ERROR_INVALID_ARGUMENT,
            )
        with self._template_operation(template_name):
            return method(self, *args, **kwargs)

    return wrapped


class RemoteBrowserBroker:
    def __init__(
        self,
        *,
        config: BrokerConfig | None = None,
        launcher: DockerSessionLauncher | None = None,
        controller: BrowserController | None = None,
        novnc_proxy_factory: Any | None = None,
    ) -> None:
        self.config = config or BrokerConfig.from_env()
        self.launcher = launcher or DockerSessionLauncher(self.config)
        self.controller = controller or BrowserController()
        self.novnc_proxy_factory = novnc_proxy_factory or NoVNCProxy
        self._lock = threading.RLock()
        self._admission_condition = threading.Condition(self._lock)
        self._admission_local = threading.local()
        self._template_operation_local = threading.local()
        self._shutdown_event = threading.Event()
        self._active_admissions = 0
        self._closing = False
        self._sessions_by_mcp: dict[str, BrowserSession] = {}
        self._retained_by_mcp: dict[str, BrowserSession] = {}
        self._retained_by_id: dict[str, BrowserSession] = {}
        self._orphaned_by_root: dict[str, BrowserSession] = {}
        self._novnc_proxies: dict[str, Any] = {}
        self._lifecycle_locks: weakref.WeakValueDictionary[str, Any] = (
            weakref.WeakValueDictionary()
        )
        self._automatic_cleanup_counts: Counter[str] = Counter()
        try:
            self._reconcile_retained_sessions()
        except Exception:
            with self._lock:
                sessions = list(
                    {
                        id(session): session
                        for session in (
                            list(self._sessions_by_mcp.values())
                            + list(self._retained_by_id.values())
                            + list(self._orphaned_by_root.values())
                        )
                    }.values()
                )
            for session in sessions:
                self._release_session_lease(session)
                self._release_mcp_lease(session)
                self._release_template_editor_lease(session)
                self._release_port_leases(session)
            raise

    @contextlib.contextmanager
    def _admission(self) -> Any:
        depth = int(getattr(self._admission_local, "depth", 0))
        if depth == 0:
            with self._admission_condition:
                shutdown_internal = bool(
                    getattr(self._admission_local, "shutdown_internal", False)
                )
                if self._closing and not shutdown_internal:
                    raise BrowserBrokerError(
                        "Remote Browser Broker is shutting down; ownership operations are closed",
                        code=ERROR_SESSION_STOPPED,
                    )
                self._active_admissions += 1
        self._admission_local.depth = depth + 1
        try:
            yield
        finally:
            if depth:
                self._admission_local.depth = depth
            else:
                try:
                    del self._admission_local.depth
                except AttributeError:
                    pass
                with self._admission_condition:
                    self._active_admissions -= 1
                    self._admission_condition.notify_all()

    def begin_shutdown(self) -> None:
        with self._admission_condition:
            self._closing = True
            self._shutdown_event.set()

    def wait_for_shutdown(self, timeout: float) -> bool:
        return self._shutdown_event.wait(timeout)

    def _wait_for_admission_drain(self) -> None:
        with self._admission_condition:
            while self._active_admissions:
                self._admission_condition.wait()

    @contextlib.contextmanager
    def _shutdown_internal_operations(self) -> Any:
        previous = bool(
            getattr(self._admission_local, "shutdown_internal", False)
        )
        self._admission_local.shutdown_internal = True
        try:
            yield
        finally:
            if previous:
                self._admission_local.shutdown_internal = True
            else:
                try:
                    del self._admission_local.shutdown_internal
                except AttributeError:
                    pass

    def _lifecycle_lock_for_key(self, mcp_session_key: str) -> Any:
        with self._lock:
            lock = self._lifecycle_locks.get(mcp_session_key)
            if lock is None:
                lock = threading.RLock()
                self._lifecycle_locks[mcp_session_key] = lock
            return lock

    def _status_snapshot_for_key(
        self,
        mcp_session_key: str,
        *,
        touch: bool,
        refresh_expiry: bool,
    ) -> dict[str, object] | None:
        with self._lock:
            session = self._current_session_locked(mcp_session_key)
        if session is None:
            return None
        acquired = session.operation_lock.acquire(blocking=False)
        if not acquired:
            with self._lock:
                if (
                    self._current_session_locked(mcp_session_key) is session
                    and session.state == STATE_STARTING
                ):
                    return session.to_status_dict()
            session.operation_lock.acquire()
        try:
            with self._lock:
                if self._current_session_locked(mcp_session_key) is not session:
                    return None
                if touch:
                    session.touch()
                if (
                    refresh_expiry
                    and session.state == STATE_STOPPED_RETAINED
                    and is_expired(session.retention_expires_at)
                ):
                    session.state = STATE_EXPIRED
                    self._write_session_metadata(session)
                return session.to_status_dict()
        finally:
            session.operation_lock.release()

    @admission_serialized
    def whoami(self, mcp_session_key: str) -> dict[str, object]:
        status = self._status_snapshot_for_key(
            mcp_session_key,
            touch=False,
            refresh_expiry=False,
        )
        return {
            "mcp_session_key": mcp_session_key,
            "has_browser_session": status is not None,
            "browser_session": status,
        }

    @admission_serialized
    def current_session_status(self, mcp_session_key: str) -> dict[str, object]:
        status = self._status_snapshot_for_key(
            mcp_session_key,
            touch=True,
            refresh_expiry=True,
        )
        return {
            "mcp_session_key": mcp_session_key,
            "has_browser_session": status is not None,
            "browser_session": status,
        }

    @admission_serialized
    def diagnostics(self, mcp_session_key: str) -> dict[str, object]:
        def response(
            session: BrowserSession | None,
            diagnostics: Mapping[str, object],
        ) -> dict[str, object]:
            return {
                "mcp_session_key": mcp_session_key,
                "has_browser_session": session is not None,
                "browser_session": session.to_status_dict() if session else None,
                "diagnostics": dict(diagnostics),
                "automatic_cleanup": {
                    **automatic_cleanup_policy(self.config),
                    "counts": dict(self._automatic_cleanup_counts),
                },
                "redaction": {
                    "page_content": "omitted",
                    "typed_text": "omitted",
                    "dom": "omitted",
                    "screenshots": "omitted",
                    "cookies": "omitted",
                    "tokens": "omitted",
                    "connect_urls": "omitted",
                    "cdp_endpoints": "omitted",
                },
            }

        with self._lock:
            session = self._current_session_locked(mcp_session_key)
        if session is not None:
            with session.operation_lock:
                with self._lock:
                    if self._current_session_locked(mcp_session_key) is not session:
                        session = None
                if session is not None:
                    session.touch()
                    diagnostics = self._diagnostics_for_session(session)
                    if session.state == STATE_RUNNING:
                        try:
                            diagnostics["fingerprint_baseline"] = (
                                self.controller.fingerprint_baseline(session)
                            )
                        except Exception as exc:  # noqa: BLE001 - diagnostics stays best-effort.
                            diagnostics["fingerprint_baseline"] = {
                                "status": "unavailable",
                                "error_code": ERROR_BROKER,
                                "error": str(exc),
                            }
                    return response(session, diagnostics)
        return response(None, self._diagnostics_for_session(None))

    @admission_serialized
    def session_list(self) -> dict[str, object]:
        with self._lock:
            candidates = self._all_sessions_locked()
        sessions: list[dict[str, object]] = []
        for session in candidates:
            with session.operation_lock:
                with self._lock:
                    if not self._session_is_indexed_locked(session):
                        continue
                    if (
                        self._retained_by_id.get(session.id) is session
                        and session.state == STATE_STOPPED_RETAINED
                        and is_expired(session.retention_expires_at)
                    ):
                        session.state = STATE_EXPIRED
                        self._write_session_metadata(session)
                    sessions.append(session.to_status_dict())
        return {
            "sessions": sessions,
            "session_count": len(sessions),
        }

    @admission_serialized
    def cleanup_expired_sessions(self) -> dict[str, object]:
        cleaned: list[dict[str, object]] = []
        with self._lock:
            candidates = list(self._retained_by_id.values())
        for session in candidates:
            with self._lifecycle_lock_for_key(session.mcp_session_key):
                with session.operation_lock:
                    with self._lock:
                        if (
                            self._retained_by_mcp.get(session.mcp_session_key) is not session
                            or self._retained_by_id.get(session.id) is not session
                        ):
                            continue
                        if (
                            session.state == STATE_STOPPED_RETAINED
                            and is_expired(session.retention_expires_at)
                        ):
                            session.state = STATE_EXPIRED
                            self._write_session_metadata(session)
                        if session.state != STATE_EXPIRED:
                            continue
                    self._delete_session_runtime(session)
                    with self._lock:
                        removed = self._remove_retained_session_locked(session)
                    if not removed:
                        continue
                    cleaned.append(session.to_status_dict())
        return {
            "status": "cleaned",
            "cleaned_sessions": cleaned,
            "cleaned_count": len(cleaned),
        }

    def _refresh_orphaned_container_inventory(self) -> None:
        if not isinstance(self.launcher, DockerSessionLauncher):
            return
        containers, inventory_error = docker_containers_with_label(
            self.config,
            None,
        )
        if inventory_error:
            raise BrowserBrokerError(
                f"Docker inventory is unavailable; orphan cleanup failed closed: {inventory_error}"
            )
        sessions_root = self._sessions_root().resolve()
        templates_root = (self.config.runtime_root / "templates").resolve()
        runtime_root = self.config.runtime_root.resolve()
        with self._lock:
            indexed_sessions = (
                list(self._sessions_by_mcp.values())
                + list(self._retained_by_id.values())
            )
            active_container_names = {
                session.container_name
                for session in self._sessions_by_mcp.values()
                if session.container_name
            }
            indexed_roots = {
                str(session.profile_dir.parent.resolve())
                for session in indexed_sessions
            }
            indexed_ids = {session.id for session in indexed_sessions}
            for root_key in indexed_roots:
                self._orphaned_by_root.pop(root_key, None)
            for container in containers:
                name = str(container.get("name") or "")
                if name in active_container_names:
                    continue
                labels = container.get("labels")
                if not isinstance(labels, Mapping):
                    labels = {}
                mount_sources = container.get("mount_sources")
                sources = [
                    Path(str(source)).resolve()
                    for source in mount_sources
                    if source
                ] if isinstance(mount_sources, list) else []
                owned_sources = [
                    source
                    for source in sources
                    if source == runtime_root
                    or source in runtime_root.parents
                    or runtime_root in source.parents
                ]
                strictly_owned = (
                    labels.get("tokimon.remote-browser-broker") == "true"
                    and labels.get("tokimon.remote-browser-installation")
                    == self.config.installation_id
                    and labels.get("tokimon.remote-browser-owner-user")
                    == self.config.owner_user
                    and labels.get("tokimon.remote-browser-checkout")
                    == str(self.config.runtime_owner_root)
                    and labels.get("tokimon.remote-browser-runtime-root")
                    == str(self.config.runtime_root)
                )
                if not strictly_owned:
                    if owned_sources:
                        raise BrowserBrokerError(
                            "A foreign or ambiguously labeled container mounts an owned "
                            "browser runtime; orphan cleanup failed closed"
                        )
                    continue
                session_id = str(
                    labels.get("tokimon.remote-browser-session") or ""
                )
                container_id = str(container.get("id") or "")
                if not DOCKER_FULL_CONTAINER_ID_RE.fullmatch(container_id):
                    raise BrowserBrokerError("Owned orphan container has no full immutable identity")
                expected_name = (
                    f"tokimon-rb-{self.config.installation_id}-{session_id}"
                )
                if not session_id or name != expected_name:
                    raise BrowserBrokerError(
                        "Owned browser container identity labels are inconsistent"
                    )
                profile_sources = [
                    source for source in owned_sources if source.name == "profile"
                ]
                if len(profile_sources) != 1 or len(owned_sources) != 1:
                    raise BrowserBrokerError(
                        "Owned orphan container does not have exactly one safe runtime profile mount"
                    )
                profile_dir = profile_sources[0]
                root_key = str(profile_dir.parent)
                if root_key in indexed_roots or session_id in indexed_ids:
                    raise BrowserBrokerError(
                        "An unexpected container overlaps an indexed browser session; "
                        "orphan cleanup failed closed"
                    )
                template_name: str | None = None
                template_version_id: str | None = None
                template_role = TEMPLATE_ROLE_RUNTIME
                if profile_dir.parent.parent == sessions_root:
                    if profile_dir.parent.name != session_id:
                        raise BrowserBrokerError(
                            "Owned container session label does not match its profile root"
                        )
                else:
                    try:
                        relative = profile_dir.relative_to(templates_root)
                    except ValueError as exc:
                        raise BrowserBrokerError(
                            "Owned container profile mount is outside recognized runtime roots"
                        ) from exc
                    if (
                        len(relative.parts) != 4
                        or relative.parts[1] not in {"drafts", "staging"}
                        or relative.parts[3] != "profile"
                        or not TEMPLATE_NAME_RE.fullmatch(relative.parts[0])
                        or (
                            relative.parts[1] == "drafts"
                            and not relative.parts[2].startswith("draft-")
                        )
                    ):
                        raise BrowserBrokerError(
                            "Owned container template-draft mount is not safely structured"
                        )
                    template_name = relative.parts[0]
                    template_version_id = relative.parts[2]
                    template_role = (
                        TEMPLATE_ROLE_EDITING
                        if relative.parts[1] == "drafts"
                        else TEMPLATE_ROLE_STAGING
                    )
                orphan = self._orphaned_by_root.get(root_key)
                if orphan is None:
                    orphan = BrowserSession(
                        id=session_id,
                        mcp_session_key=f"orphan:{session_id}",
                        profile_dir=profile_dir,
                        cdp_port=0,
                        novnc_port=0,
                        novnc_backend_port=0,
                        container_name=name,
                        container_id=container_id,
                        created_at="",
                        last_used_at="",
                        state=STATE_ORPHANED,
                        template_name=template_name,
                        template_version_id=template_version_id,
                        template_role=template_role,
                        token_exposure_mode="none",
                    )
                    self._orphaned_by_root[root_key] = orphan
                else:
                    if (
                        (orphan.container_id is not None and orphan.container_id != container_id)
                        or (orphan.container_id is None and self._session_metadata_path(orphan).exists())
                    ):
                        raise BrowserBrokerError(
                            "Owned orphan container does not match its recorded immutable identity"
                        )
                    if orphan.container_name and orphan.container_name != name:
                        raise BrowserBrokerError(
                            "Multiple owned containers mount the same orphan profile; "
                            "cleanup failed closed"
                        )
                    orphan.id = session_id
                    orphan.container_name = name
                    orphan.container_id = container_id

    @admission_serialized
    def cleanup_orphaned_sessions(self) -> dict[str, object]:
        self._refresh_orphaned_container_inventory()
        with self._lock:
            candidates = list(self._orphaned_by_root.items())
            initial_on_disk = self._valid_session_records_on_disk()
        initial_key_counts = Counter(
            candidate.mcp_session_key for candidate in initial_on_disk
        )
        initial_id_counts = Counter(candidate.id for candidate in initial_on_disk)
        cleaned: list[dict[str, object]] = []
        peer_owned: list[dict[str, object]] = []
        reclassified: list[dict[str, object]] = []
        for root_key, session in candidates:
            with self._lifecycle_lock_for_key(session.mcp_session_key):
                with session.operation_lock:
                    with self._lock:
                        indexed_roots = {
                            str(item.profile_dir.parent.resolve())
                            for item in (
                                list(self._sessions_by_mcp.values())
                                + list(self._retained_by_id.values())
                            )
                        }
                        if root_key in indexed_roots:
                            self._orphaned_by_root.pop(root_key, None)
                            continue
                        if self._orphaned_by_root.get(root_key) is not session:
                            continue
                    session_root = session.profile_dir.parent
                    if not session_root.exists() and not session.container_name:
                        with self._lock:
                            if self._orphaned_by_root.get(root_key) is session:
                                self._orphaned_by_root.pop(root_key, None)
                        self._release_session_lease(session)
                        self._release_mcp_lease(session)
                        self._release_template_editor_lease(session)
                        self._release_port_leases(session)
                        session.state = STATE_EXPIRED
                        cleaned.append(session.to_status_dict())
                        continue
                    had_session_lease = session.ownership_lease is not None
                    if not self._try_acquire_session_lease(
                        session,
                        allow_missing_root=True,
                    ):
                        session.state = STATE_PEER_OWNED
                        peer_owned.append(session.to_status_dict())
                        continue
                    with self._lock:
                        on_disk = self._valid_session_records_on_disk()
                        retained_candidate = next(
                            (
                                candidate
                                for candidate in on_disk
                                if candidate.state
                                in {STATE_STOPPED_RETAINED, STATE_EXPIRED}
                                and str(candidate.profile_dir.parent.resolve())
                                == root_key
                            ),
                            None,
                        )
                        if retained_candidate is not None:
                            duplicate_key = (
                                initial_key_counts[
                                    retained_candidate.mcp_session_key
                                ]
                                > 1
                            ) or sum(
                                candidate.mcp_session_key
                                == retained_candidate.mcp_session_key
                                for candidate in on_disk
                            ) > 1
                            duplicate_id = (
                                initial_id_counts[retained_candidate.id] > 1
                            ) or sum(
                                candidate.id == retained_candidate.id
                                for candidate in on_disk
                            ) > 1
                            if not duplicate_key and not duplicate_id:
                                promoted = self._promote_retained_candidate_locked(
                                    retained_candidate
                                )
                            else:
                                promoted = None
                        else:
                            duplicate_key = False
                            duplicate_id = False
                            promoted = None
                    if retained_candidate is not None and not (
                        duplicate_key or duplicate_id
                    ):
                        if promoted is None:
                            session.state = STATE_PEER_OWNED
                            peer_owned.append(session.to_status_dict())
                        else:
                            reclassified.append(promoted.to_status_dict())
                        continue
                    if (
                        session.template_role
                        in {TEMPLATE_ROLE_EDITING, TEMPLATE_ROLE_STAGING}
                        and session.template_editor_lease is None
                    ):
                        editor_lease = self._try_acquire_template_editor_lease(
                            session.template_name or ""
                        )
                        if editor_lease is None:
                            if not had_session_lease:
                                self._release_session_lease(session)
                            session.state = STATE_PEER_OWNED
                            peer_owned.append(session.to_status_dict())
                            continue
                        session.template_editor_lease = editor_lease
                    session.state = STATE_ORPHANED
                    if session.container_name:
                        self.launcher.stop(session)
                    self._delete_session_runtime(session)
                    with self._lock:
                        if self._orphaned_by_root.get(root_key) is not session:
                            continue
                        self._orphaned_by_root.pop(root_key, None)
                    cleaned.append(session.to_status_dict())
        return {
            "status": (
                "cleaned"
                if cleaned
                else "reclassified"
                if reclassified
                else "no_orphaned_sessions_cleaned"
            ),
            "cleaned_sessions": cleaned,
            "cleaned_count": len(cleaned),
            "peer_owned_sessions": peer_owned,
            "peer_owned_count": len(peer_owned),
            "reclassified_sessions": reclassified,
            "reclassified_count": len(reclassified),
            "note": (
                "Only strictly contained broker session roots are eligible for cleanup."
            ),
        }

    @lifecycle_serialized
    def recover_session(
        self,
        mcp_session_key: str,
        *,
        recovery_token: str,
    ) -> dict[str, object]:
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", recovery_token):
            raise BrowserBrokerError(
                "Recovery token is invalid or unavailable",
                code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
            )
        token_hash = hashlib.sha256(recovery_token.encode("ascii")).hexdigest()
        with self._lock:
            records = self._valid_session_records_on_disk()
            matches = [
                item for item in records
                if item.recovery_token_hash
                and re.fullmatch(r"[0-9a-f]{64}", item.recovery_token_hash)
                and secrets.compare_digest(item.recovery_token_hash, token_hash)
            ]
            if len(matches) != 1:
                raise BrowserBrokerError(
                    "Recovery token is invalid or unavailable",
                    code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
                )
            candidate = matches[0]
            old_key = candidate.mcp_session_key
            if candidate.state not in {STATE_STOPPED_RETAINED, STATE_EXPIRED}:
                raise BrowserBrokerError(
                    "Recovery requires a stopped retained browser or template draft",
                    code=ERROR_SESSION_NOT_RETAINED,
                )
            if old_key != mcp_session_key and (
                self._current_session_locked(mcp_session_key) is not None
                or any(item.mcp_session_key == mcp_session_key for item in records)
            ):
                raise BrowserBrokerError(
                    "The destination MCP session already has browser ownership",
                    code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
                )
            old_lock = self._lifecycle_lock_for_key(old_key)
            if not old_lock.acquire(blocking=False):
                raise BrowserBrokerError(
                    "The original browser session is busy; retry after it stops",
                    code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
                )
            try:
                if (
                    sum(item.id == candidate.id for item in records) != 1
                    or sum(item.mcp_session_key == old_key for item in records) != 1
                ):
                    raise BrowserBrokerError(
                        "Recovery ownership metadata is ambiguous",
                        code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
                    )
                session = self._retained_by_mcp.get(old_key)
                if session is None:
                    session = self._promote_released_retained_for_key_locked(old_key)
                if session is None or not session.operation_lock.acquire(blocking=False):
                    raise BrowserBrokerError(
                        "The retained browser is unavailable or busy",
                        code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
                    )
                try:
                    # Re-read after acquiring exclusive ownership: a peer may have
                    # rebound the checkpoint between discovery and lease acquisition.
                    metadata_path = self._session_metadata_path(session)
                    reader = (
                        self._template_editor_from_metadata
                        if session.template_role == TEMPLATE_ROLE_EDITING
                        else self._session_from_metadata
                    )
                    checkpoint = reader(metadata_path)
                    if (
                        checkpoint is None
                        or session.id != candidate.id
                        or checkpoint.id != session.id
                        or checkpoint.mcp_session_key != old_key
                        or checkpoint.state not in {STATE_STOPPED_RETAINED, STATE_EXPIRED}
                        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint.recovery_token_hash or "")
                        or not secrets.compare_digest(
                            checkpoint.recovery_token_hash or "", token_hash
                        )
                        or not secrets.compare_digest(
                            session.recovery_token_hash or "", token_hash
                        )
                    ):
                        raise BrowserBrokerError(
                            "Recovery ownership changed; retry with the current token",
                            code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
                        )
                    if old_key != mcp_session_key:
                        new_lease = self._try_acquire_socket_lease(
                            namespace="mcp", identity=mcp_session_key, blocking=False
                        )
                        if new_lease is None:
                            raise BrowserBrokerError(
                                "The destination MCP session is owned by another broker",
                                code=ERROR_SESSION_RECOVERY_UNAVAILABLE,
                            )
                        session.mcp_session_key = mcp_session_key
                        try:
                            replacement_token = self._issue_recovery_token(session)
                        except Exception:
                            session.mcp_session_key = old_key
                            new_lease.close()
                            raise
                        old_lease = session.mcp_ownership_lease
                        session.mcp_ownership_lease = new_lease
                        self._retained_by_mcp.pop(old_key)
                        self._retained_by_mcp[mcp_session_key] = session
                        if old_lease is not None:
                            old_lease.close()
                    else:
                        replacement_token = self._issue_recovery_token(session)
                    return {
                        "status": "recovered",
                        "recovery_token": replacement_token,
                        "browser_session": session.to_status_dict(),
                    }
                finally:
                    session.operation_lock.release()
            finally:
                old_lock.release()

    def _issue_recovery_token(self, session: BrowserSession) -> str:
        token = secrets.token_urlsafe(32)
        previous_hash = session.recovery_token_hash
        session.recovery_token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        try:
            self._write_session_metadata(session)
        except Exception:
            session.recovery_token_hash = previous_hash
            raise
        return token

    @lifecycle_serialized
    def cleanup_session_artifacts(
        self,
        mcp_session_key: str,
    ) -> dict[str, object]:
        with self._lock:
            if mcp_session_key not in self._retained_by_mcp:
                raise BrowserBrokerError(
                    "Stop the current browser before clearing its artifacts",
                    code=ERROR_SESSION_NOT_RETAINED,
                )
            session = self._retained_session_or_error_locked(mcp_session_key)
        with session.operation_lock:
            if session.template_role != TEMPLATE_ROLE_RUNTIME:
                raise BrowserBrokerError(
                    "Artifact cleanup requires an ordinary stopped browser session",
                    code=ERROR_SESSION_NOT_RETAINED,
                )
            if isinstance(self.launcher, DockerSessionLauncher):
                self.launcher.stop(session)
            root = self._validated_session_root(session)
            profile = self._validated_session_profile(session)
            directories = {
                "uploads": self._owned_child_directory(profile, "uploads"),
                "screenshots": self._owned_child_directory(root, "screenshots"),
                "artifacts": self._owned_child_directory(root, "artifacts"),
            }
            usage = {
                name: self._owned_artifact_usage(path)
                for name, path in directories.items()
            }
            for path in directories.values():
                parent_fd = os.open(
                    path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                )
                try:
                    shutil.rmtree(path.name, dir_fd=parent_fd)
                finally:
                    os.close(parent_fd)
            return {
                "status": "cleaned",
                "removed": {
                    name: {"files": count, "bytes": size}
                    for name, (count, size) in usage.items()
                },
                "browser_session": session.to_status_dict(),
            }

    @lifecycle_serialized
    def delete_retained_session(
        self,
        mcp_session_key: str,
    ) -> dict[str, object]:
        with self._lock:
            session = self._retained_session_or_error_locked(mcp_session_key)
        with session.operation_lock:
            with self._lock:
                if (
                    self._retained_by_mcp.get(mcp_session_key) is not session
                    or self._retained_by_id.get(session.id) is not session
                ):
                    raise BrowserBrokerError(
                        "Retained browser session changed before deletion",
                        code=ERROR_SESSION_NOT_FOUND,
                    )
            self._delete_session_runtime(session)
            with self._lock:
                if not self._remove_retained_session_locked(session):
                    raise BrowserBrokerError(
                        "Retained browser session changed during deletion",
                        code=ERROR_SESSION_NOT_FOUND,
                    )
        return {
            "status": "deleted",
            "browser_session": session.to_status_dict(),
        }

    @lifecycle_serialized
    def extend_retained_session(
        self,
        mcp_session_key: str,
        *,
        ttl_seconds: int,
    ) -> dict[str, object]:
        if ttl_seconds <= 0 or ttl_seconds > self.config.max_retention_seconds:
            raise BrowserBrokerError(
                f"ttl_seconds must be between 1 and {self.config.max_retention_seconds}",
                code=ERROR_INVALID_ARGUMENT,
            )
        with self._lock:
            session = self._retained_session_or_error_locked(mcp_session_key)
        with session.operation_lock:
            with self._lock:
                if (
                    self._retained_by_mcp.get(mcp_session_key) is not session
                    or self._retained_by_id.get(session.id) is not session
                ):
                    raise BrowserBrokerError(
                        "Retained browser session changed before extension",
                        code=ERROR_SESSION_NOT_FOUND,
                    )
                session.state = STATE_STOPPED_RETAINED
                session.retention_expires_at = utc_after(ttl_seconds)
                session.touch()
                self._write_session_metadata(session)
                status = session.to_status_dict()
        return {
            "status": "extended",
            "ttl_seconds": ttl_seconds,
            "browser_session": status,
        }

    @admission_serialized
    @optional_template_operation_serialized
    @lifecycle_serialized
    def ensure(
        self,
        mcp_session_key: str,
        *,
        start_url: str | None = None,
        template_name: str | None = None,
        recreate: bool = False,
    ) -> dict[str, object]:
        normalized_start_url = normalize_url(start_url) if start_url else None
        normalized_template_name = validate_template_name(template_name) if template_name is not None else None
        validate_operation_parameters({
            "start_url": normalized_start_url,
            "template_name": normalized_template_name,
            "recreate": recreate,
        })
        if recreate:
            self._discard_current_session(mcp_session_key)
        session = self._ensure_session(mcp_session_key, template_name=normalized_template_name)
        with session.operation_lock:
            navigation = None
            if normalized_start_url:
                self._require_page_action_allowed(session)
                session.invalidate_refs()
                navigation = self.controller.navigate(session, normalized_start_url)
                session.touch()
            return {
                "status": "ready",
                "browser_session": session.to_status_dict(),
                "connect": self._redacted_connect_info(session),
                "navigation": navigation,
                "recovery_token": self._issue_recovery_token(session),
            }

    def list_templates(self) -> dict[str, object]:
        templates_root = self._templates_root()
        templates: list[dict[str, object]] = []
        if templates_root.exists():
            for item in sorted(templates_root.iterdir(), key=lambda path: path.name):
                if (
                    item.is_symlink()
                    or not item.is_dir()
                    or not TEMPLATE_NAME_RE.fullmatch(item.name)
                ):
                    continue
                template_payload = self._template_summary(item.name)
                if template_payload is not None:
                    templates.append(template_payload)
        return {
            "templates_root": str(templates_root),
            "templates": templates,
        }

    def _template_editor_locked(self, template_name: str) -> BrowserSession | None:
        for session in (
            list(self._sessions_by_mcp.values())
            + list(self._retained_by_id.values())
            + list(self._orphaned_by_root.values())
        ):
            if (
                session.template_role in {
                    TEMPLATE_ROLE_EDITING,
                    TEMPLATE_ROLE_STAGING,
                }
                and session.template_name == template_name
            ):
                return session
        return None

    def _template_has_unpublished_roots(self, template_name: str) -> bool:
        for collection_name in ("drafts", "staging"):
            collection_root = self._template_collection_root(
                template_name,
                collection_name,
                create=False,
            )
            if collection_root.exists() and any(collection_root.iterdir()):
                return True
        return False

    def _rollback_failed_template_start(
        self,
        mcp_session_key: str,
        session: BrowserSession,
    ) -> None:
        with session.operation_lock:
            try:
                self.launcher.stop(session)
            except Exception:
                session.state = STATE_UNKNOWN
                session.touch()
                raise
            try:
                self._revoke_novnc_access(session)
            except Exception:
                session.state = STATE_UNKNOWN
                session.touch()
                raise
            session.state = STATE_EXPIRED
            session.touch()
            try:
                self._delete_session_runtime(session)
            except Exception:
                session.state = STATE_UNKNOWN
                session.touch()
                raise
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is session:
                    self._sessions_by_mcp.pop(mcp_session_key, None)

    def _cleanup_or_quarantine_unpublished_profile_locked(
        self,
        *,
        session_id: str,
        mcp_session_key: str,
        profile_dir: Path,
        template_name: str | None = None,
        template_version_id: str | None = None,
        template_role: str = TEMPLATE_ROLE_RUNTIME,
    ) -> None:
        root = profile_dir.parent
        if not root.exists():
            return
        try:
            shutil.rmtree(root)
        except Exception:
            orphan = BrowserSession(
                id=session_id,
                mcp_session_key=mcp_session_key,
                profile_dir=profile_dir,
                cdp_port=0,
                novnc_port=0,
                novnc_backend_port=0,
                container_name="",
                created_at=utc_now(),
                last_used_at=utc_now(),
                state=STATE_ORPHANED,
                template_name=template_name,
                template_version_id=template_version_id,
                template_role=template_role,
                token_exposure_mode="none",
            )
            if not self._try_acquire_session_lease(orphan, create_root=True):
                orphan.state = STATE_PEER_OWNED
            if template_role in {TEMPLATE_ROLE_EDITING, TEMPLATE_ROLE_STAGING}:
                editor_lease = self._try_acquire_template_editor_lease(
                    template_name or ""
                )
                if editor_lease is not None:
                    orphan.template_editor_lease = editor_lease
            self._orphaned_by_root[str(root.resolve())] = orphan
            raise

    @admission_serialized
    @template_operation_serialized
    @lifecycle_serialized
    def create_template(
        self,
        mcp_session_key: str,
        *,
        template_name: str,
        description: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        name = validate_template_name(template_name)
        with self._lock:
            existing = self._sessions_by_mcp.get(mcp_session_key)
            if existing is not None:
                raise BrowserBrokerError(
                    "Stop or seal the current browser session before creating a template",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            if mcp_session_key in self._retained_by_mcp:
                raise BrowserBrokerError(
                    "Delete the current retained browser session before creating a template",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            editor = self._template_editor_locked(name)
            if editor is not None:
                raise BrowserBrokerError(
                    f"Template already has an active editing session: {name}",
                    code=ERROR_TEMPLATE_EXISTS,
                )
            template_dir = self._template_dir(name)
            if (
                template_dir.exists()
                and not overwrite
                and self._template_summary(name) is not None
            ):
                raise BrowserBrokerError(
                    f"Template already exists: {name}",
                    code=ERROR_TEMPLATE_EXISTS,
                )
            if self._template_has_unpublished_roots(name):
                raise BrowserBrokerError(
                    f"Template already has an editing or staged draft: {name}",
                    code=ERROR_TEMPLATE_EXISTS,
                )
            editor_lease = self._try_acquire_template_editor_lease(name)
            if editor_lease is None:
                raise BrowserBrokerError(
                    f"Template is being edited by another broker process: {name}",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            try:
                drafts_root = self._template_collection_root(
                    name,
                    "drafts",
                    create=True,
                )
                draft_id = f"draft-{secrets.token_hex(6)}"
                profile_dir = drafts_root / draft_id / "profile"
                session = self._start_browser_session_locked(
                    mcp_session_key=mcp_session_key,
                    profile_dir=profile_dir,
                    template_name=name,
                    template_version_id=draft_id,
                    template_role=TEMPLATE_ROLE_EDITING,
                    template_editor_lease=editor_lease,
                )
            except Exception:
                if editor_lease.fileno() != -1 and not any(
                    candidate.template_editor_lease is editor_lease
                    for candidate in self._sessions_by_mcp.values()
                ):
                    editor_lease.close()
                raise
            draft_metadata = {
                "name": name,
                "version_id": draft_id,
                "state": TEMPLATE_STATE_EDITING,
                "description": (description or "").strip(),
                "created_at": session.created_at,
                "updated_at": session.created_at,
                "overwrite_existing": overwrite,
            }
        try:
            self._write_template_metadata(
                drafts_root / draft_id / "metadata.json",
                draft_metadata,
            )
            self.controller.wait_until_ready(
                session,
                timeout_sec=self.config.startup_timeout_sec,
            )
            with self._lock:
                session.state = STATE_RUNNING
                session.touch()
                self._write_session_metadata(session)
        except Exception:
            self._rollback_failed_template_start(mcp_session_key, session)
            raise
        finally:
            session.operation_lock.release()
        with session.operation_lock:
            return {
                "status": "editing",
                "template": draft_metadata,
                "recovery_token": self._issue_recovery_token(session),
                "browser_session": session.to_status_dict(),
                "connect": self._redacted_connect_info(session),
            }

    def _template_seal_conflict(self, mcp_session_key: str) -> BrowserBrokerError:
        with self._lock:
            current = self._current_session_locked(mcp_session_key)
        role = current.template_role if current else "none"
        state = current.state if current else "none"
        if role == TEMPLATE_ROLE_RUNTIME:
            message = "Ordinary runtime changes stay in its retained profile; use browser_stop/browser_ensure, or browser_session_recover after reconnect. Template create/refresh requires an empty session."
            next_actions = ["browser_ensure"] if state in {STATE_STOPPED_RETAINED, STATE_EXPIRED} else ["browser_stop", "browser_ensure"]
        elif role == TEMPLATE_ROLE_EDITING:
            message = "Seal this session's current_template_name. After reconnect, recover the retained draft with its recovery token before sealing; another template name cannot transfer editor ownership."
            next_actions = ["browser_template_seal"]
        else:
            message = "Create or refresh a template in an empty session, or recover your retained draft using its recovery token before sealing."
            next_actions = ["browser_template_create", "browser_template_refresh", "browser_session_recover"]
        return BrowserBrokerError(
            "Only the current template editor can seal. " + message,
            code=ERROR_TEMPLATE_CONFLICT,
            details={"current_role": role, "current_state": state,
                     "current_template_name": current.template_name if current else None,
                     "next_actions": next_actions},
        )

    @template_operation_serialized
    @lifecycle_serialized
    def seal_template(
        self,
        mcp_session_key: str,
        *,
        template_name: str,
        description: str | None = None,
    ) -> dict[str, object]:
        name = validate_template_name(template_name)
        with self._lock:
            session = self._sessions_by_mcp.get(mcp_session_key)
            if session is None:
                retained = self._retained_by_mcp.get(mcp_session_key)
                if (
                    retained is not None
                    and retained.template_role == TEMPLATE_ROLE_EDITING
                    and retained.template_name == name
                    and retained.state in {STATE_STOPPED_RETAINED, STATE_EXPIRED}
                ):
                    self._retained_by_mcp.pop(mcp_session_key, None)
                    self._retained_by_id.pop(retained.id, None)
                    self._sessions_by_mcp[mcp_session_key] = retained
                    session = retained
        if session is None:
            raise self._template_seal_conflict(mcp_session_key)
        with session.operation_lock:
            with self._lock:
                if (
                    self._sessions_by_mcp.get(mcp_session_key) is not session
                    or session.template_role != TEMPLATE_ROLE_EDITING
                    or session.template_name != name
                ):
                    raise self._template_seal_conflict(mcp_session_key)
            if session.ownership_lease is None and not self._try_acquire_session_lease(
                session
            ):
                raise BrowserBrokerError(
                    "Template draft is owned by another broker process",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            if session.template_editor_lease is None:
                editor_lease = self._try_acquire_template_editor_lease(name)
                if editor_lease is None:
                    raise BrowserBrokerError(
                        "Template is being edited by another broker process",
                        code=ERROR_TEMPLATE_CONFLICT,
                    )
                session.template_editor_lease = editor_lease
            if session.state != STATE_STOPPED_RETAINED:
                self.launcher.stop(session)
                try:
                    self._revoke_novnc_access(session)
                except Exception:
                    session.state = STATE_UNKNOWN
                    session.touch()
                    raise
                session.state = STATE_STOPPED_RETAINED
            if not session.retention_expires_at:
                session.retention_expires_at = utc_after(
                    self.config.retention_ttl_seconds
                )
            session.touch()
            self._write_session_metadata(session)
            template_dir = self._template_dir(name)
            draft_metadata_path = session.profile_dir.parent / "metadata.json"
            try:
                draft_payload = json.loads(
                    draft_metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                draft_payload = {}
            overwrite_existing = bool(
                isinstance(draft_payload, Mapping)
                and draft_payload.get("overwrite_existing") is True
            )
            version_id = self._sealed_version_for_source_draft(
                template_dir,
                session.template_version_id,
            )
            if version_id is None:
                version_id = (
                    f"v-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
                    f"{secrets.token_hex(3)}"
                )
                versions_root = self._template_collection_root(
                    name,
                    "versions",
                    create=True,
                )
                staging_root = self._template_collection_root(
                    name,
                    "staging",
                    create=True,
                )
                version_dir = versions_root / version_id
                staging_dir = (
                    staging_root / f"{version_id}-{secrets.token_hex(6)}"
                )
                now = utc_now()
                metadata = {
                    "name": name,
                    "version_id": version_id,
                    "state": TEMPLATE_STATE_SEALED,
                    "description": (description or "").strip(),
                    "created_at": now,
                    "updated_at": now,
                    "source_draft_id": session.template_version_id,
                    "chrome_channel": "current-runtime",
                    "runtime_image": self.config.image_name,
                    "locale": "",
                    "timezone": "",
                    "viewport": self.config.viewport,
                    "screen": self.config.screen,
                    "network_policy": self.config.network_policy,
                    "language_hints": [],
                    "overwrite_existing": overwrite_existing,
                }
                try:
                    staging_dir.mkdir(parents=True, exist_ok=False)
                    self._copy_profile(session.profile_dir, staging_dir / "profile")
                    self._write_template_metadata(
                        staging_dir / "metadata.json",
                        metadata,
                    )
                    staging_dir.replace(version_dir)
                except Exception:
                    if staging_dir.exists():
                        self._cleanup_or_quarantine_unpublished_profile_locked(
                            session_id=f"orphan-{secrets.token_hex(6)}",
                            mcp_session_key=mcp_session_key,
                            profile_dir=staging_dir / "profile",
                            template_name=name,
                            template_version_id=staging_dir.name,
                            template_role=TEMPLATE_ROLE_STAGING,
                        )
                    raise
            if overwrite_existing:
                self._prune_overwritten_template_versions(
                    template_dir,
                    keep_version_id=version_id,
                )
            template = self._read_template(name, version_id=version_id)
            draft_dir = session.profile_dir.parent
            if draft_dir.exists():
                shutil.rmtree(draft_dir)
            response = {
                "status": "sealed",
                "template": template.to_dict(),
                "browser_session_stopped": True,
            }
            self._release_session_lease(session)
            self._release_mcp_lease(session)
            self._release_template_editor_lease(session)
            self._release_port_leases(session)
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is session:
                    self._sessions_by_mcp.pop(mcp_session_key, None)
        return response

    @admission_serialized
    @template_operation_serialized
    @lifecycle_serialized
    def refresh_template(
        self,
        mcp_session_key: str,
        *,
        template_name: str,
        version_id: str | None = None,
    ) -> dict[str, object]:
        name = validate_template_name(template_name)
        with self._lock:
            existing = self._sessions_by_mcp.get(mcp_session_key)
            if existing is not None:
                raise BrowserBrokerError(
                    "Stop or seal the current browser session before refreshing a template",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            if mcp_session_key in self._retained_by_mcp:
                raise BrowserBrokerError(
                    "Delete the current retained browser session before refreshing a template",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            editor = self._template_editor_locked(name)
            if editor is not None or self._template_has_unpublished_roots(name):
                raise BrowserBrokerError(
                    f"Template already has an editing or staged draft: {name}",
                    code=ERROR_TEMPLATE_EXISTS,
                )
            source = self._read_template(name, version_id=version_id)
            self._require_template_compatible(source)
            editor_lease = self._try_acquire_template_editor_lease(name)
            if editor_lease is None:
                raise BrowserBrokerError(
                    f"Template is being edited by another broker process: {name}",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            try:
                draft_id = f"draft-{secrets.token_hex(6)}"
                drafts_root = self._template_collection_root(
                    name,
                    "drafts",
                    create=True,
                )
                profile_dir = drafts_root / draft_id / "profile"
                session = self._start_browser_session_locked(
                    mcp_session_key=mcp_session_key,
                    profile_dir=profile_dir,
                    template_name=name,
                    template_version_id=draft_id,
                    template_role=TEMPLATE_ROLE_EDITING,
                    profile_source=source.profile_dir,
                    template_editor_lease=editor_lease,
                )
            except Exception:
                if editor_lease.fileno() != -1 and not any(
                    candidate.template_editor_lease is editor_lease
                    for candidate in self._sessions_by_mcp.values()
                ):
                    editor_lease.close()
                raise
            draft_metadata = {
                "name": name,
                "version_id": draft_id,
                "state": TEMPLATE_STATE_EDITING,
                "description": str(source.metadata.get("description") or ""),
                "created_at": session.created_at,
                "updated_at": session.created_at,
                "source_version_id": source.version_id,
            }
        try:
            self._write_template_metadata(
                drafts_root / draft_id / "metadata.json",
                draft_metadata,
            )
            self.controller.wait_until_ready(
                session,
                timeout_sec=self.config.startup_timeout_sec,
            )
            with self._lock:
                session.state = STATE_RUNNING
                session.touch()
                self._write_session_metadata(session)
        except Exception:
            self._rollback_failed_template_start(mcp_session_key, session)
            raise
        finally:
            session.operation_lock.release()
        with session.operation_lock:
            return {
                "status": "editing",
                "template": draft_metadata,
                "recovery_token": self._issue_recovery_token(session),
                "browser_session": session.to_status_dict(),
                "connect": self._redacted_connect_info(session),
            }

    @template_operation_serialized
    def delete_template(
        self,
        *,
        template_name: str,
    ) -> dict[str, object]:
        name = validate_template_name(template_name)
        editor_lease = self._try_acquire_template_editor_lease(name)
        if editor_lease is None:
            raise BrowserBrokerError(
                "Seal or discard the template-editing session before deletion",
                code=ERROR_TEMPLATE_CONFLICT,
            )
        try:
            with self._lock:
                if (
                    self._template_editor_locked(name) is not None
                    or self._template_has_unpublished_roots(name)
                ):
                    raise BrowserBrokerError(
                        "Seal or clean the template draft/staging state before deletion",
                        code=ERROR_TEMPLATE_CONFLICT,
                    )
                destination = self._template_dir(name)
                if not destination.exists():
                    return {
                        "status": "not_found",
                        "template_name": name,
                    }
                shutil.rmtree(destination)
            return {
                "status": "deleted",
                "template_name": name,
                "state": TEMPLATE_STATE_DELETED,
            }
        finally:
            editor_lease.close()

    @lifecycle_serialized
    def connect_info(self, mcp_session_key: str) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            session.touch()
            return {
                "browser_session": session.to_status_dict(),
                "connect": self.connect_info_for_session(session),
            }

    @admission_serialized
    def navigate(
        self,
        mcp_session_key: str,
        url: str,
    ) -> dict[str, object]:
        normalized_url = normalize_url(url)
        session = self._ensure_session(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            validate_operation_parameters({"url": normalized_url})
            session.invalidate_refs()
            result = self.controller.navigate(session, normalized_url)
            session.touch()
            return {
                "status": "navigated",
                "browser_session": session.to_status_dict(),
                "page": result,
            }

    @admission_serialized
    def snapshot(
        self,
        mcp_session_key: str,
        *,
        max_text_chars: int = 20000,
        max_elements: int = 200,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_observation_allowed(session)
            if max_text_chars <= 0 or max_text_chars > MAX_SNAPSHOT_TEXT_CHARS:
                raise BrowserBrokerError(
                    f"max_text_chars must be between 1 and {MAX_SNAPSHOT_TEXT_CHARS}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            if max_elements <= 0 or max_elements > MAX_SNAPSHOT_ELEMENTS:
                raise BrowserBrokerError(
                    f"max_elements must be between 1 and {MAX_SNAPSHOT_ELEMENTS}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            result = self.controller.snapshot(
                session,
                max_text_chars=max_text_chars,
                max_elements=max_elements,
            )
            if not isinstance(result, Mapping):
                raise BrowserBrokerError("Snapshot controller returned a non-object payload")
            result = normalize_snapshot_payload(
                result,
                max_text_chars=max_text_chars,
                max_elements=max_elements,
            )
            session.record_snapshot_refs(result)
            session.touch()
            return {
                "browser_session": session.to_status_dict(),
                "page": result,
            }

    @admission_serialized
    def artifact(self, mcp_session_key: str, **arguments: Any) -> dict:
        return artifacts.capture(self, mcp_session_key, **arguments)

    @admission_serialized
    def evaluate(
        self,
        mcp_session_key: str,
        *,
        expression: str,
        await_promise: bool = True,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            validate_operation_parameters({"expression": expression, "await_promise": await_promise})
            result = self.controller.evaluate(
                session,
                expression=expression,
                await_promise=await_promise,
            )
            session.touch()
            return {
                "browser_session": session.to_status_dict(),
                "result": result,
            }

    @admission_serialized
    def screenshot(
        self,
        mcp_session_key: str,
        *,
        image_format: str = "png",
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        normalized_format = (image_format or "png").strip().lower()
        if normalized_format not in {"png", "jpeg"}:
            raise BrowserBrokerError(
                "format must be png or jpeg",
                code=ERROR_INVALID_ARGUMENT,
        )
        with session.operation_lock:
            self._require_page_observation_allowed(session)
            session_root = self._validated_session_root(session)
            screenshots_dir = self._owned_child_directory(
                session_root,
                "screenshots",
            )
            screenshot_count, screenshot_total_bytes = self._owned_artifact_usage(
                screenshots_dir
            )
            if (
                screenshot_count >= MAX_SCREENSHOT_ARTIFACTS
                or screenshot_total_bytes >= MAX_SCREENSHOT_TOTAL_BYTES
            ):
                raise BrowserBrokerError(
                    "Screenshot artifact quota is exhausted; stop/delete the session to clean it"
                )
            quality = 80 if normalized_format == "jpeg" else None
            screenshot_data = self.controller.screenshot(
                session,
                image_format=normalized_format,
                quality=quality,
            )
            if not isinstance(screenshot_data, str) or len(screenshot_data) > (
                ((MAX_SCREENSHOT_BYTES + 2) // 3) * 4
            ):
                raise BrowserBrokerError(
                    f"CDP screenshot response exceeds {MAX_SCREENSHOT_BYTES} decoded bytes"
                )
            try:
                image_bytes = base64.b64decode(screenshot_data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise BrowserBrokerError("CDP screenshot response is not valid base64") from exc
            if len(image_bytes) > MAX_SCREENSHOT_BYTES:
                raise BrowserBrokerError(
                    f"CDP screenshot response exceeds {MAX_SCREENSHOT_BYTES} decoded bytes"
                )
            if screenshot_total_bytes + len(image_bytes) > MAX_SCREENSHOT_TOTAL_BYTES:
                raise BrowserBrokerError(
                    "Screenshot artifact byte quota would be exceeded"
                )
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            extension = "jpeg" if normalized_format == "jpeg" else "png"
            screenshot_name = f"{timestamp}-{secrets.token_hex(6)}.{extension}"
            screenshot_path = screenshots_dir / screenshot_name
            tmp_name = f".{screenshot_name}.{secrets.token_hex(6)}.tmp"
            directory_fd = os.open(
                screenshots_dir,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
            try:
                artifact_fd = os.open(
                    tmp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
                with os.fdopen(artifact_fd, "wb") as artifact:
                    artifact.write(image_bytes)
                    artifact.flush()
                    os.fsync(artifact.fileno())
                os.replace(
                    tmp_name,
                    screenshot_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
            finally:
                try:
                    os.unlink(tmp_name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
                os.close(directory_fd)
            session.touch()
            return {
                "browser_session": session.to_status_dict(),
                "screenshot": {
                    "path": self._display_path(screenshot_path),
                    "format": normalized_format,
                    "capture": "viewport",
                    "size_bytes": len(image_bytes),
                },
            }

    @admission_serialized
    def click(
        self,
        mcp_session_key: str,
        *,
        ref: str,
        pacing: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            self._set_pacing(session, pacing)
            ref_data = session.resolve_ref(ref)
            validate_operation_parameters({"ref": ref, "pacing": session.pacing_mode})
            result = self.controller.click(session, ref_data)
            session.touch()
            return {
                "status": "clicked",
                "browser_session": session.to_status_dict(),
                "ref": ref,
                "result": result,
            }

    @admission_serialized
    def type_text(
        self,
        mcp_session_key: str,
        *,
        ref: str,
        text: str,
        clear: bool = False,
        pacing: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            self._set_pacing(session, pacing)
            ref_data = session.resolve_ref(ref)
            validate_operation_parameters({
                "ref": ref,
                "text": text,
                "clear": clear,
                "pacing": session.pacing_mode,
            })
            result = self.controller.type_text(session, ref_data, text=text, clear=clear)
            session.touch()
            return {
                "status": "typed",
                "browser_session": session.to_status_dict(),
                "ref": ref,
                "result": result,
            }

    @admission_serialized
    def upload_file(
        self,
        mcp_session_key: str,
        *,
        ref: str,
        path: str,
        pacing: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            self._set_pacing(session, pacing)
            ref_data = session.resolve_ref(ref)
            source = self._resolve_upload_source(path)
            source_digest, source_size = self._read_bounded_upload_source(source)
            validate_operation_parameters({
                "ref": ref,
                "source_path": str(source),
                "source_sha256": source_digest,
                "source_size_bytes": source_size,
                "pacing": session.pacing_mode,
            })
            staged = self._stage_upload_file(
                session,
                str(source),
                expected_sha256=source_digest,
                expected_size_bytes=source_size,
            )
            result = self.controller.upload_file(
                session,
                ref_data,
                file_path=staged.container_path,
            )
            session.touch()
            return {
                "status": "uploaded",
                "browser_session": session.to_status_dict(),
                "ref": ref,
                "file_name": staged.source_name,
                "upload_file_name": staged.upload_name,
                "file_size_bytes": staged.size_bytes,
                "result": result,
            }

    @admission_serialized
    def backend_request(
        self,
        mcp_session_key: str,
        *,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        max_response_chars: int = DEFAULT_BACKEND_RESPONSE_MAX_CHARS,
        timeout_ms: int = DEFAULT_BACKEND_REQUEST_TIMEOUT_MS,
        pacing: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            self._set_pacing(session, pacing)
            normalized_method = validate_backend_request_method(method)
            normalized_path = validate_backend_request_path(path)
            if (
                max_response_chars <= 0
                or max_response_chars > MAX_BACKEND_RESPONSE_MAX_CHARS
            ):
                raise BrowserBrokerError(
                    "max_response_chars must be between 1 and "
                    f"{MAX_BACKEND_RESPONSE_MAX_CHARS}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            if timeout_ms <= 0 or timeout_ms > MAX_BACKEND_REQUEST_TIMEOUT_MS:
                raise BrowserBrokerError(
                    f"timeout_ms must be between 1 and {MAX_BACKEND_REQUEST_TIMEOUT_MS}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            validate_operation_parameters({
                "method": normalized_method,
                "path": normalized_path,
                "body": body,
                "max_response_chars": max_response_chars,
                "timeout_ms": timeout_ms,
                "pacing": session.pacing_mode,
            })
            result = self.controller.backend_request(
                session,
                method=normalized_method,
                path=normalized_path,
                body=body,
                max_response_chars=max_response_chars,
                timeout_ms=timeout_ms,
            )
            session.touch()
            return {
                "status": "requested",
                "browser_session": session.to_status_dict(),
                "method": normalized_method,
                "path": normalized_path,
                "response": result,
            }

    @admission_serialized
    def press_key(
        self,
        mcp_session_key: str,
        *,
        key: str,
        pacing: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            self._set_pacing(session, pacing)
            validate_operation_parameters({"key": key, "pacing": session.pacing_mode})
            result = self.controller.press_key(session, key)
            session.touch()
            return {
                "status": "pressed",
                "browser_session": session.to_status_dict(),
                "result": result,
            }

    @admission_serialized
    def wait(
        self,
        mcp_session_key: str,
        *,
        seconds: float | None = None,
        text: str | None = None,
        timeout_ms: int = 5000,
        pacing: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            self._set_pacing(session, pacing)
            if seconds is not None and (
                not math.isfinite(seconds)
                or seconds < 0
                or seconds > MAX_WAIT_SECONDS
            ):
                raise BrowserBrokerError(
                    f"seconds must be finite and between 0 and {MAX_WAIT_SECONDS:g}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            if timeout_ms <= 0 or timeout_ms > MAX_WAIT_TIMEOUT_MS:
                raise BrowserBrokerError(
                    f"timeout_ms must be between 1 and {MAX_WAIT_TIMEOUT_MS}",
                    code=ERROR_INVALID_ARGUMENT,
                )
            result = self.controller.wait(
                session,
                seconds=seconds,
                text=text,
                timeout_ms=timeout_ms,
            )
            session.touch()
            return {
                "status": "waited",
                "browser_session": session.to_status_dict(),
                "result": result,
            }

    def _stage_upload_file(
        self,
        session: BrowserSession,
        path: str,
        *,
        expected_sha256: str | None = None,
        expected_size_bytes: int,
    ) -> StagedUploadFile:
        source = self._resolve_upload_source(path)
        source_size = expected_size_bytes
        profile_dir = self._validated_session_profile(session)
        uploads_dir = self._owned_child_directory(profile_dir, "uploads")
        upload_count, upload_total_bytes = self._owned_artifact_usage(uploads_dir)
        if upload_count >= MAX_STAGED_UPLOADS:
            raise BrowserBrokerError(
                "Staged upload count quota is exhausted; stop the browser and use browser_session_cleanup_artifacts"
            )
        if upload_total_bytes + source_size > MAX_STAGED_UPLOAD_TOTAL_BYTES:
            raise BrowserBrokerError("Staged upload byte quota would be exceeded")
        staged_dir_name = secrets.token_hex(8)
        staged_dir = uploads_dir / staged_dir_name
        upload_name = self._unique_upload_file_name(source.name)
        staged_host_path = staged_dir / upload_name
        uploads_fd = os.open(
            uploads_dir,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        staged_fd: int | None = None
        staged_created = False
        try:
            os.mkdir(staged_dir_name, mode=0o700, dir_fd=uploads_fd)
            staged_created = True
            staged_fd = os.open(
                staged_dir_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=uploads_fd,
            )
            destination_fd = os.open(
                upload_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=staged_fd,
            )
            with os.fdopen(destination_fd, "wb") as destination_stream:
                actual_sha256, size_bytes = self._read_bounded_upload_source(
                    source,
                    destination_stream=destination_stream,
                )
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
            if (
                size_bytes != expected_size_bytes
                or (expected_sha256 and actual_sha256 != expected_sha256)
            ):
                raise BrowserBrokerError(
                    "Upload source changed while preparing the file; retry with the current source",
                    code=ERROR_INVALID_ARGUMENT,
                )
        except Exception:
            if staged_fd is not None:
                try:
                    os.unlink(upload_name, dir_fd=staged_fd)
                except FileNotFoundError:
                    pass
            if staged_created:
                try:
                    os.rmdir(staged_dir_name, dir_fd=uploads_fd)
                except OSError:
                    pass
            raise
        finally:
            if staged_fd is not None:
                os.close(staged_fd)
            os.close(uploads_fd)
        return StagedUploadFile(
            source_name=source.name,
            upload_name=upload_name,
            container_path=f"/profile/uploads/{staged_dir_name}/{upload_name}",
            size_bytes=size_bytes,
        )

    def _resolve_upload_source(self, path: str) -> Path:
        source = Path(path).expanduser()
        if not source.is_absolute():
            raise BrowserBrokerError(
                "Upload path must be absolute and refer to the task workspace file",
                code=ERROR_INVALID_ARGUMENT,
            )
        try:
            source = source.resolve(strict=True)
        except OSError as exc:
            raise BrowserBrokerError(
                f"Upload file does not exist: {path}",
                code=ERROR_INVALID_ARGUMENT,
            ) from exc
        if not source.is_file():
            raise BrowserBrokerError(
                f"Upload path is not a regular file: {path}",
                code=ERROR_INVALID_ARGUMENT,
            )
        return source

    def _read_bounded_upload_source(
        self,
        path: Path,
        *,
        destination_stream: Any | None = None,
    ) -> tuple[str, int]:
        digest = hashlib.sha256()
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        try:
            source_fd = os.open(path, flags)
        except OSError as exc:
            raise BrowserBrokerError(
                "Upload source could not be opened as a no-follow regular file",
                code=ERROR_INVALID_ARGUMENT,
            ) from exc
        size_bytes = 0
        try:
            source_stat = os.fstat(source_fd)
            if not stat.S_ISREG(source_stat.st_mode):
                raise BrowserBrokerError(
                    "Upload source is not a regular file",
                    code=ERROR_INVALID_ARGUMENT,
                )
            if source_stat.st_size > MAX_UPLOAD_BYTES:
                raise BrowserBrokerError(
                    f"Upload file exceeds {MAX_UPLOAD_BYTES} bytes",
                    code=ERROR_INVALID_ARGUMENT,
                )
            while True:
                read_size = min(
                    1024 * 1024,
                    MAX_UPLOAD_BYTES - size_bytes + 1,
                )
                try:
                    chunk = os.read(source_fd, read_size)
                except BlockingIOError as exc:
                    raise BrowserBrokerError(
                        "Upload source would block during bounded inspection",
                        code=ERROR_INVALID_ARGUMENT,
                    ) from exc
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > MAX_UPLOAD_BYTES:
                    raise BrowserBrokerError(
                        f"Upload file exceeds {MAX_UPLOAD_BYTES} bytes",
                        code=ERROR_INVALID_ARGUMENT,
                    )
                digest.update(chunk)
                if destination_stream is not None:
                    destination_stream.write(chunk)
        except OSError as exc:
            raise BrowserBrokerError(
                "Upload source could not be read safely",
                code=ERROR_INVALID_ARGUMENT,
            ) from exc
        finally:
            os.close(source_fd)
        return digest.hexdigest(), size_bytes

    def _unique_upload_file_name(self, source_name: str) -> str:
        token = secrets.token_hex(6)
        clean_name = source_name.strip() or "upload"
        max_clean_len = 240 - len(token)
        if len(clean_name) > max_clean_len:
            clean_name = clean_name[-max_clean_len:]
        return f"{token}-{clean_name}"

    @admission_serialized
    def new_tab(
        self,
        mcp_session_key: str,
        *,
        url: str | None = None,
    ) -> dict[str, object]:
        normalized_url = normalize_url(url) if url else None
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            validate_operation_parameters({"url": normalized_url})
            result = self.controller.new_tab(session, normalized_url)
            session.invalidate_refs()
            session.touch()
            return {
                "status": "new_tab",
                "browser_session": session.to_status_dict(),
                "tab": result,
            }

    @admission_serialized
    def switch_tab(
        self,
        mcp_session_key: str,
        *,
        tab_id: str,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            validate_operation_parameters({"tab_id": tab_id})
            result = self.controller.switch_tab(session, tab_id)
            session.invalidate_refs()
            session.touch()
            return {
                "status": "switched_tab",
                "browser_session": session.to_status_dict(),
                "tab": result,
            }

    @admission_serialized
    def close_tab(
        self,
        mcp_session_key: str,
        *,
        tab_id: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            self._require_page_action_allowed(session)
            validate_operation_parameters({"tab_id": tab_id})
            result = self.controller.close_tab(session, tab_id)
            session.invalidate_refs()
            session.touch()
            return {
                "status": "closed_tab",
                "browser_session": session.to_status_dict(),
                "tab": result,
            }

    @lifecycle_serialized
    def pause_for_human(
        self,
        mcp_session_key: str,
        *,
        reason: str | None = None,
    ) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is not session:
                    raise BrowserBrokerError(
                        "Browser session stopped before the pause could be applied",
                        code=ERROR_SESSION_STOPPED,
                    )
                if session.state not in {STATE_RUNNING, STATE_PAUSED_OBSERVED}:
                    raise BrowserBrokerError(
                        "Only a running or already-paused browser session can be paused",
                        code=ERROR_SESSION_STOPPED,
                    )
            session.state = STATE_PAUSED_OBSERVED
            session.touch()
            return {
                "status": "paused_for_human",
                "reason": (reason or "").strip(),
                "browser_session": session.to_status_dict(),
            }

    @lifecycle_serialized
    def resume(self, mcp_session_key: str) -> dict[str, object]:
        session = self._session_or_error(mcp_session_key)
        with session.operation_lock:
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is not session:
                    raise BrowserBrokerError(
                        "Browser session stopped before resume could be applied",
                        code=ERROR_SESSION_STOPPED,
                    )
                if session.state != STATE_PAUSED_OBSERVED:
                    raise BrowserBrokerError(
                        "Only a paused browser session can be resumed",
                        code=ERROR_SESSION_STOPPED,
                    )
            session.state = STATE_RUNNING
            session.touch()
            response = {
                "status": "resumed",
                "browser_session": session.to_status_dict(),
            }
        return response

    @lifecycle_serialized
    def stop(self, mcp_session_key: str) -> dict[str, object]:
        with self._lock:
            session = self._sessions_by_mcp.get(mcp_session_key)
            retained = self._retained_by_mcp.get(mcp_session_key)
        if session is None:
            if retained is not None:
                if retained.template_role == TEMPLATE_ROLE_EDITING:
                    raise BrowserBrokerError(
                        "Template-editing sessions must be sealed or discarded, not retained",
                        code=ERROR_TEMPLATE_CONFLICT,
                    )
                self._write_session_metadata(retained)
                return {
                    "status": "stopped_retained",
                    "browser_session": retained.to_status_dict(),
                }
            return {"status": "not_running"}
        with session.operation_lock:
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is not session:
                    retained = self._retained_by_mcp.get(mcp_session_key)
                    if retained is not None:
                        return {
                            "status": "stopped_retained",
                            "browser_session": retained.to_status_dict(),
                        }
                    return {"status": "not_running"}
            if session.template_role == TEMPLATE_ROLE_EDITING:
                raise BrowserBrokerError(
                    "Seal the template or discard its editing session instead of stopping it",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            self.launcher.stop(session)
            try:
                self._revoke_novnc_access(session)
            except Exception:
                session.state = STATE_UNKNOWN
                session.touch()
                raise
            session.state = STATE_STOPPED_RETAINED
            session.retention_expires_at = utc_after(self.config.retention_ttl_seconds)
            session.touch()
            self._release_port_leases(session)
            self._write_session_metadata(session)
            with self._lock:
                self._sessions_by_mcp.pop(mcp_session_key, None)
                self._retained_by_mcp[mcp_session_key] = session
                self._retained_by_id[session.id] = session
            response = {
                "status": "stopped_retained",
                "browser_session": session.to_status_dict(),
            }
        return response

    @lifecycle_serialized
    def shutdown_transport_session(
        self,
        mcp_session_key: str,
    ) -> dict[str, object]:
        with self._lock:
            session = self._sessions_by_mcp.get(mcp_session_key)
            retained = self._retained_by_mcp.get(mcp_session_key)
        candidate = session or retained
        if candidate is not None and candidate.template_role == TEMPLATE_ROLE_EDITING:
            if session is not None:
                self._quiesce_template_editor_for_shutdown(mcp_session_key)
            return {"status": "template_draft_retained"}
        return self.stop(mcp_session_key)

    @lifecycle_serialized
    def _quiesce_template_editor_for_shutdown(self, mcp_session_key: str) -> None:
        with self._lock:
            session = self._sessions_by_mcp.get(mcp_session_key)
        if session is None:
            return
        with session.operation_lock:
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is not session:
                    return
                if session.template_role != TEMPLATE_ROLE_EDITING:
                    raise BrowserBrokerError(
                        "Only template-editing sessions use draft-preserving shutdown"
                    )
            self.launcher.stop(session)
            try:
                self._revoke_novnc_access(session)
            except Exception:
                session.state = STATE_UNKNOWN
                session.touch()
                raise
            session.state = STATE_STOPPED_RETAINED
            session.retention_expires_at = utc_after(
                self.config.retention_ttl_seconds
            )
            session.touch()
            self._release_port_leases(session)
            self._write_session_metadata(session)
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is session:
                    self._sessions_by_mcp.pop(mcp_session_key, None)
                    self._retained_by_mcp[mcp_session_key] = session
                    self._retained_by_id[session.id] = session

    def shutdown_all_sessions(self) -> dict[str, object]:
        self.begin_shutdown()
        self._wait_for_admission_drain()
        with self._shutdown_internal_operations():
            return self._shutdown_all_sessions_admission_drained()

    def _shutdown_all_sessions_admission_drained(self) -> dict[str, object]:
        with self._lock:
            active = {
                key: session for key, session in self._sessions_by_mcp.items()
            }
        stopped: list[str] = []
        failures: list[dict[str, str]] = []
        for mcp_session_key, session in sorted(active.items()):
            try:
                if session.template_role == TEMPLATE_ROLE_EDITING:
                    self._quiesce_template_editor_for_shutdown(mcp_session_key)
                else:
                    self.stop(mcp_session_key)
            except Exception as exc:  # noqa: BLE001 - preserve remaining ownership.
                failures.append(
                    {
                        "mcp_session_key": mcp_session_key,
                        "error": str(exc),
                    }
                )
            else:
                stopped.append(mcp_session_key)
        if failures:
            raise BrowserBrokerError(
                "One or more browser sessions could not be shut down safely",
                details={
                    "stopped_session_count": len(stopped),
                    "shutdown_failures": failures,
                },
            )
        return {
            "status": "stopped_retained",
            "stopped_session_keys": stopped,
            "stopped_session_count": len(stopped),
        }

    def release_shutdown_leases(self) -> None:
        if int(getattr(self._admission_local, "depth", 0)):
            raise BrowserBrokerError(
                "Cannot release browser ownership from inside an active ownership operation"
            )
        self._wait_for_admission_drain()
        with self._lock:
            if not self._closing:
                raise BrowserBrokerError(
                    "Browser ownership leases can only be released during shutdown"
                )
            sessions = self._all_sessions_locked()
            self._sessions_by_mcp.clear()
            self._retained_by_mcp.clear()
            self._retained_by_id.clear()
            self._orphaned_by_root.clear()
        for session in sessions:
            self._release_session_lease(session)
            self._release_mcp_lease(session)
            self._release_template_editor_lease(session)
            self._release_port_leases(session)

    @lifecycle_serialized
    def _discard_current_session(self, mcp_session_key: str) -> None:
        with self._lock:
            session = self._sessions_by_mcp.get(mcp_session_key)
            retained = self._retained_by_mcp.get(mcp_session_key)
        if session is not None:
            with session.operation_lock:
                with self._lock:
                    if self._sessions_by_mcp.get(mcp_session_key) is not session:
                        session = None
                if session is None:
                    return self._discard_current_session(mcp_session_key)
                self.launcher.stop(session)
                try:
                    self._revoke_novnc_access(session)
                except Exception:
                    session.state = STATE_UNKNOWN
                    session.touch()
                    raise
                session.state = STATE_EXPIRED
                self._delete_session_runtime(session)
                with self._lock:
                    if self._sessions_by_mcp.get(mcp_session_key) is session:
                        self._sessions_by_mcp.pop(mcp_session_key, None)
        if retained is not None:
            with retained.operation_lock:
                self._delete_session_runtime(retained)
                with self._lock:
                    self._remove_retained_session_locked(retained)

    def _delete_session_runtime(self, session: BrowserSession) -> None:
        session_root = self._validated_session_root(session)
        if (
            session.state in {STATE_STOPPED_RETAINED, STATE_EXPIRED}
            and isinstance(self.launcher, DockerSessionLauncher)
            and session.container_name
        ):
            # A previous broker can die after restarting a retained container but
            # before deleting its stopped-session metadata. Verify and stop that
            # exact immutable container identity before touching the mounted root.
            self.launcher.stop(session)
        if session_root.exists():
            try:
                shutil.rmtree(session_root)
            except Exception:
                if session.retention_expires_at is None:
                    session.state = STATE_UNKNOWN
                    session.touch()
                session.profile_dir.mkdir(parents=True, exist_ok=True)
                self._write_session_metadata(session)
                raise
        self._release_session_lease(session)
        self._release_mcp_lease(session)
        self._release_template_editor_lease(session)
        self._release_port_leases(session)

    def _diagnostics_for_session(self, session: BrowserSession | None) -> dict[str, object]:
        if session is None:
            return {
                "state": None,
                "cdp_exposure_mode": "none",
                "novnc_exposure_mode": "none",
                "network_policy": network_policy_diagnostics(self.config.network_policy),
                "retention": "none",
                "runtime": "no_browser_session",
            }
        return {
            "state": session.state,
            "pacing_mode": session.pacing_mode,
            "template_name": session.template_name,
            "retention_expires_at": session.retention_expires_at,
            "cdp_exposure_mode": "host_loopback_same_os_authority",
            "novnc_exposure_mode": session.token_exposure_mode,
            "network_policy": network_policy_diagnostics(self.config.network_policy),
            "technical_reachability": {
                "container_known": bool(session.container_name),
                "profile_known": session.profile_dir.exists(),
            },
        }

    def _redacted_connect_info(self, session: BrowserSession) -> dict[str, object]:
        return {
            "redacted": True,
            "state": session.state,
            "tool": "browser_connect_info",
        }

    def connect_info_for_session(self, session: BrowserSession) -> dict[str, object]:
        with session.operation_lock:
            with self._lock:
                is_current = (
                    self._sessions_by_mcp.get(session.mcp_session_key) is session
                )
            if not is_current or session.state not in {
                STATE_RUNNING,
                STATE_PAUSED_OBSERVED,
            }:
                raise BrowserBrokerError(
                    "Browser connection info is unavailable while the session is not active",
                    code=ERROR_SESSION_STOPPED,
                )
            self._ensure_novnc_access(session)
            host = self.config.public_host
            novnc_query = urllib.parse.urlencode(
                {
                    "autoconnect": "1",
                    "resize": "scale",
                    "rb_token": session.novnc_token or "",
                }
            )
            novnc_url = f"http://{host}:{session.novnc_port}/vnc.html?{novnc_query}"
            ssh_tunnel_command = None
            if self.config.ssh_host:
                ssh_tunnel_command = (
                    f"ssh -N -L {session.novnc_port}:127.0.0.1:{session.novnc_port} "
                    f"{self.config.ssh_host}"
                )
            return {
                "novnc_url": novnc_url,
                "novnc_token_expires_at": session.novnc_token_expires_at,
                "cdp": {
                    "exposure": "host_loopback_same_os_authority",
                    "note": (
                        "CDP is not returned through MCP, but its loopback Docker port is "
                        "reachable by processes sharing the broker's host network authority."
                    ),
                },
                "host_ports": {
                    "novnc": session.novnc_port,
                },
                "ssh_tunnel_required": self.config.ssh_host is not None,
                "ssh_tunnel_command": ssh_tunnel_command,
                "note": (
                    "Open the noVNC URL directly when the broker runs on this host; "
                    "use the SSH tunnel command first when the broker runs on a remote host."
                ),
            }

    def _ensure_novnc_access(self, session: BrowserSession) -> None:
        proxy = self._novnc_proxies.get(session.id)
        if (
            session.novnc_token_active
            and session.novnc_token
            and proxy is not None
            and not is_expired(session.novnc_token_expires_at)
        ):
            return
        self._revoke_novnc_access(session)
        now = utc_now()
        token = secrets.token_urlsafe(32)
        expires_at = utc_after(self.config.novnc_token_ttl_seconds)
        proxy = self.novnc_proxy_factory(
            listen_host="127.0.0.1",
            listen_port=session.novnc_port,
            backend_host="127.0.0.1",
            backend_port=session.novnc_backend_port,
            token=token,
            expires_at=expires_at,
        )
        try:
            proxy.start()
        except Exception:
            self._novnc_proxies[session.id] = proxy
            try:
                self._revoke_novnc_access(session)
            except BrowserBrokerError as revoke_exc:
                raise BrowserBrokerError(
                    "noVNC proxy startup failed and partial-proxy cleanup also failed"
                ) from revoke_exc
            raise
        session.novnc_token = token
        session.novnc_token_issued_at = now
        session.novnc_token_expires_at = expires_at
        session.novnc_token_active = True
        session.token_exposure_mode = "token_gated_proxy"
        self._novnc_proxies[session.id] = proxy

    def _revoke_novnc_access(self, session: BrowserSession) -> None:
        proxy = self._novnc_proxies.get(session.id)
        session.novnc_token_active = False
        session.novnc_token = None
        session.novnc_token_issued_at = None
        session.novnc_token_expires_at = None
        session.token_exposure_mode = "revoked"
        if proxy is None:
            return
        try:
            proxy.stop()
        except Exception as exc:
            raise BrowserBrokerError(
                "Failed to stop the noVNC proxy after revoking its credential"
            ) from exc
        if self._novnc_proxies.get(session.id) is proxy:
            self._novnc_proxies.pop(session.id, None)

    def _require_page_action_allowed(self, session: BrowserSession) -> None:
        if session.state == STATE_PAUSED_OBSERVED:
            raise BrowserBrokerError(
                "Page actions are blocked while paused for human takeover",
                code=ERROR_PAUSED,
            )
        if session.state != STATE_RUNNING:
            raise BrowserBrokerError(
                f"Page actions require a running browser session; current state is {session.state}",
                code=ERROR_SESSION_NOT_FOUND,
            )

    def _require_page_observation_allowed(self, session: BrowserSession) -> None:
        if session.state not in {STATE_RUNNING, STATE_PAUSED_OBSERVED}:
            raise BrowserBrokerError(
                f"Page observation requires a running browser session; current state is {session.state}",
                code=ERROR_SESSION_NOT_FOUND,
            )

    def _set_pacing(self, session: BrowserSession, pacing: str | None) -> None:
        value = (pacing or PACING_MODE_HUMAN).strip().lower()
        if value in {"", "default"}:
            value = PACING_MODE_HUMAN
        if value not in {PACING_MODE_HUMAN, "fast", "batch"}:
            raise BrowserBrokerError("pacing must be human, fast, or batch", code=ERROR_INVALID_ARGUMENT)
        session.pacing_mode = value

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config.workspace_root))
        except ValueError:
            return str(path)

    def _require_session_capacity_locked(
        self, *, creating_new: bool, mcp_session_key: str
    ) -> None:
        def needs_total_slot() -> bool:
            return (
                creating_new
                and self._owned_session_count_locked() >= self.config.max_total_sessions
            )

        def needs_active_slot() -> bool:
            return len(self._sessions_by_mcp) >= self.config.max_active_sessions

        if not needs_total_slot() and not needs_active_slot():
            return
        counts: Counter[str] = Counter()
        candidates = sorted(
            self._sessions_by_mcp.values(),
            key=lambda item: (item.last_used_at, item.id),
        )
        for session in candidates:
            if not needs_total_slot() and not needs_active_slot():
                break
            if (
                session.mcp_session_key == mcp_session_key
                or session.template_role != TEMPLATE_ROLE_RUNTIME
                or session.state == STATE_PEER_OWNED
            ):
                continue
            lifecycle_lock = self._lifecycle_lock_for_key(session.mcp_session_key)
            # Admission holds the inventory lock: never wait on another caller's locks.
            if not lifecycle_lock.acquire(blocking=False):
                continue
            try:
                if not session.operation_lock.acquire(blocking=False):
                    continue
                try:
                    outcome = None
                    if (
                        not needs_total_slot()
                        and needs_active_slot()
                        and self._sessions_by_mcp.get(session.mcp_session_key) is session
                    ):
                        if self._idle_reclaimable(session):
                            self.stop(session.mcp_session_key)
                            outcome = "idle_stopped"
                    if outcome is not None:
                        counts[outcome] += 1
                        self._automatic_cleanup_counts[outcome] += 1
                        log_runtime_event(
                            configure_runtime_logger(), logging.INFO,
                            "Automatic browser capacity cleanup completed",
                            correlation_id=mcp_session_key,
                            mcp_session_key=session.mcp_session_key,
                            status=outcome,
                        )
                except (BrowserBrokerError, OSError, subprocess.SubprocessError):
                    counts["failed"] += 1
                    self._automatic_cleanup_counts["failed"] += 1
                    log_runtime_event(
                        configure_runtime_logger(), logging.WARNING,
                        "Automatic browser capacity cleanup failed; ownership retained for retry",
                        correlation_id=mcp_session_key,
                        mcp_session_key=session.mcp_session_key,
                        status="failed",
                    )
                finally:
                    session.operation_lock.release()
            finally:
                lifecycle_lock.release()
        if needs_total_slot() or needs_active_slot():
            raise BrowserBrokerError(
                "Browser session capacity is unavailable after automatic cleanup; "
                "stop busy browsers or explicitly delete unneeded retained profiles to free capacity",
                code=ERROR_SESSION_CAPACITY_EXHAUSTED,
                details={
                    "capacity": {
                        "active": len(self._sessions_by_mcp),
                        "max_active": self.config.max_active_sessions,
                        "total": self._owned_session_count_locked(),
                        "max_total": self.config.max_total_sessions,
                    },
                    "automatic_cleanup": { "counts": dict(counts)},
                },
            )

    def _idle_reclaimable(self, session: BrowserSession) -> bool:
        if session.state == STATE_STOPPED_RETAINED:
            # Retry a completed stop whose retained checkpoint could not be persisted.
            return True
        if (
            not self.config.idle_timeout_seconds
            or session.state != STATE_RUNNING
            or (session.novnc_token_active and not is_expired(session.novnc_token_expires_at))
        ):
            return False
        try:
            last_used = datetime.fromisoformat(session.last_used_at)
        except ValueError:
            return False
        return last_used.tzinfo is not None and (
            datetime.now(timezone.utc) - last_used
        ).total_seconds() >= self.config.idle_timeout_seconds

    def _allocate_session_ports_locked(
        self,
    ) -> tuple[int, int, int, list[socket.socket]]:
        used_ports = self._used_ports_locked()
        leases: list[socket.socket] = []
        try:
            cdp_port, cdp_lease = self._acquire_available_port_lease(
                self.config.cdp_port_range,
                excluded_ports=used_ports,
            )
            leases.append(cdp_lease)
            used_ports.add(cdp_port)
            novnc_port, novnc_lease = self._acquire_available_port_lease(
                self.config.novnc_port_range,
                excluded_ports=used_ports,
            )
            leases.append(novnc_lease)
            used_ports.add(novnc_port)
            novnc_backend_port, backend_lease = self._acquire_available_port_lease(
                self.config.novnc_port_range,
                excluded_ports=used_ports,
            )
            leases.append(backend_lease)
            return cdp_port, novnc_port, novnc_backend_port, leases
        except Exception:
            for lease in leases:
                lease.close()
            raise
    def _start_browser_session_locked(
        self,
        *,
        mcp_session_key: str,
        profile_dir: Path,
        template_name: str | None = None,
        template_version_id: str | None = None,
        template_role: str = TEMPLATE_ROLE_RUNTIME,
        session_id: str | None = None,
        ownership_lease: Any | None = None,
        profile_source: Path | None = None,
        template_editor_lease: Any | None = None,
    ) -> BrowserSession:
        if (
            mcp_session_key not in self._sessions_by_mcp
            and mcp_session_key not in self._retained_by_mcp
        ):
            self._promote_released_retained_for_key_locked(mcp_session_key)
        if (
            mcp_session_key in self._sessions_by_mcp
            or mcp_session_key in self._retained_by_mcp
        ):
            raise BrowserBrokerError(
                "MCP session already owns a browser session",
                code=ERROR_BROKER,
            )
        self._require_session_capacity_locked(
            creating_new=True, mcp_session_key=mcp_session_key
        )
        resolved_session_id = session_id or secrets.token_hex(6)
        (
            cdp_port,
            novnc_port,
            novnc_backend_port,
            port_ownership_leases,
        ) = self._allocate_session_ports_locked()
        now = utc_now()
        session = BrowserSession(
            id=resolved_session_id,
            mcp_session_key=mcp_session_key,
            profile_dir=profile_dir,
            cdp_port=cdp_port,
            novnc_port=novnc_port,
            novnc_backend_port=novnc_backend_port,
            container_name=(
                f"tokimon-rb-{self.config.installation_id}-{resolved_session_id}"
            ),
            created_at=now,
            last_used_at=now,
            state=STATE_STARTING,
            template_name=template_name,
            template_version_id=template_version_id,
            template_role=template_role,
            ownership_lease=ownership_lease,
            template_editor_lease=template_editor_lease,
            port_ownership_leases=port_ownership_leases,
        )
        if not self._try_acquire_mcp_lease(session):
            self._release_port_leases(session)
            raise BrowserBrokerError(
                "MCP session key is currently owned by another broker process",
                code=ERROR_BROKER,
            )
        try:
            has_session_lease = self._try_acquire_session_lease(
                session,
                create_root=True,
            )
        except Exception:
            self._release_mcp_lease(session)
            self._release_port_leases(session)
            raise
        if not has_session_lease:
            self._release_mcp_lease(session)
            self._release_port_leases(session)
            raise BrowserBrokerError(
                "Browser profile root is currently owned by another broker process",
                code=ERROR_BROKER,
            )
        self._sessions_by_mcp[mcp_session_key] = session
        try:
            self._write_session_metadata(session)
        except Exception:
            try:
                self._delete_session_runtime(session)
            finally:
                if session.ownership_lease is None:
                    self._sessions_by_mcp.pop(mcp_session_key, None)
            raise
        if profile_source is not None:
            try:
                self._copy_profile(profile_source, profile_dir)
            except Exception:
                session.state = STATE_UNKNOWN
                session.touch()
                try:
                    self._delete_session_runtime(session)
                except Exception:
                    raise
                finally:
                    if session.ownership_lease is None:
                        self._sessions_by_mcp.pop(mcp_session_key, None)
                raise
        session.operation_lock.acquire()
        try:
            session.container_name = self._launch_session_container(session)
        except Exception as exc:
            session.state = STATE_UNKNOWN
            session.touch()
            if (
                isinstance(exc, BrowserBrokerError)
                and exc.details.get("cleanup_verified") is False
            ):
                session.operation_lock.release()
                raise
            try:
                self._delete_session_runtime(session)
                if self._sessions_by_mcp.get(mcp_session_key) is session:
                    self._sessions_by_mcp.pop(mcp_session_key, None)
            finally:
                session.operation_lock.release()
            raise
        return session

    @admission_serialized
    @optional_template_operation_serialized
    @lifecycle_serialized
    def _ensure_session(
        self,
        mcp_session_key: str,
        *,
        template_name: str | None = None,
    ) -> BrowserSession:
        if not mcp_session_key:
            raise BrowserBrokerError("MCP session key is required", code=ERROR_INVALID_ARGUMENT)
        normalized_template_name = (
            validate_template_name(template_name) if template_name is not None else None
        )
        with self._lock:
            existing = self._sessions_by_mcp.get(mcp_session_key)
            if (
                existing is None
                and mcp_session_key not in self._retained_by_mcp
            ):
                self._promote_released_retained_for_key_locked(mcp_session_key)
                existing = self._sessions_by_mcp.get(mcp_session_key)
            if existing is not None and existing.state == STATE_STOPPED_RETAINED:
                if existing.template_role == TEMPLATE_ROLE_EDITING:
                    raise BrowserBrokerError(
                        "Retry template seal or discard the stopped editing session",
                        code=ERROR_TEMPLATE_CONFLICT,
                    )
                self._write_session_metadata(existing)
                self._sessions_by_mcp.pop(mcp_session_key, None)
                self._retained_by_mcp[mcp_session_key] = existing
                self._retained_by_id[existing.id] = existing
                existing = None
            if existing is not None:
                if (
                    normalized_template_name is not None
                    and existing.template_name != normalized_template_name
                ):
                    raise BrowserBrokerError(
                        "template_name only applies when creating a new browser session; "
                        "stop the current session before using a different template",
                        code=ERROR_SESSION_TEMPLATE_MISMATCH,
                )
                existing.touch()
                return existing
            retained = self._retained_by_mcp.get(mcp_session_key)
            if retained is not None:
                if retained.template_role == TEMPLATE_ROLE_EDITING:
                    raise BrowserBrokerError(
                        "Seal or explicitly discard the retained template-editing session",
                        code=ERROR_TEMPLATE_CONFLICT,
                    )
                if (
                    normalized_template_name is not None
                    and retained.template_name != normalized_template_name
                ):
                    raise BrowserBrokerError(
                        "template_name only applies when creating a new browser session; "
                        "delete the retained session before using a different template",
                        code=ERROR_SESSION_TEMPLATE_MISMATCH,
                    )
                self._require_session_capacity_locked(
                    creating_new=False, mcp_session_key=mcp_session_key
                )
                self._restore_retained_template_version_locked(retained)
                (
                    cdp_port,
                    novnc_port,
                    novnc_backend_port,
                    port_ownership_leases,
                ) = self._allocate_session_ports_locked()
                now = utc_now()
                session = BrowserSession(
                    id=retained.id,
                    mcp_session_key=mcp_session_key,
                    profile_dir=retained.profile_dir,
                    cdp_port=cdp_port,
                    novnc_port=novnc_port,
                    novnc_backend_port=novnc_backend_port,
                    container_name=(
                        f"tokimon-rb-{self.config.installation_id}-{retained.id}"
                    ),
                    recovery_token_hash=retained.recovery_token_hash,
                    container_id=retained.container_id,
                    created_at=retained.created_at or now,
                    last_used_at=now,
                    state=STATE_STARTING,
                    template_name=retained.template_name,
                    template_version_id=retained.template_version_id,
                    template_role=retained.template_role,
                    retention_expires_at=None,
                    ownership_lease=retained.ownership_lease,
                    mcp_ownership_lease=retained.mcp_ownership_lease,
                    template_editor_lease=retained.template_editor_lease,
                    port_ownership_leases=port_ownership_leases,
                )
                retained.ownership_lease = None
                retained.mcp_ownership_lease = None
                retained.template_editor_lease = None
                session.operation_lock.acquire()
                self._retained_by_mcp.pop(mcp_session_key, None)
                self._retained_by_id.pop(retained.id, None)
                self._sessions_by_mcp[mcp_session_key] = session
                starting_checkpoint_written = False
                start_cleanup_can_restore = False
                try:
                    self._write_session_metadata(session)
                    starting_checkpoint_written = True
                    if isinstance(self.launcher, DockerSessionLauncher):
                        self.launcher.stop(session)
                    start_cleanup_can_restore = True
                    session.container_name = self._launch_session_container(session)
                except Exception as exc:
                    if (
                        starting_checkpoint_written
                        and (
                            not start_cleanup_can_restore
                            or (
                                isinstance(exc, BrowserBrokerError)
                                and exc.details.get("cleanup_verified") is False
                            )
                        )
                    ):
                        session.state = STATE_UNKNOWN
                        session.touch()
                        session.operation_lock.release()
                        raise
                    self._sessions_by_mcp.pop(mcp_session_key, None)
                    retained.ownership_lease = session.ownership_lease
                    retained.mcp_ownership_lease = session.mcp_ownership_lease
                    retained.template_editor_lease = session.template_editor_lease
                    session.ownership_lease = None
                    session.mcp_ownership_lease = None
                    session.template_editor_lease = None
                    self._release_port_leases(session)
                    self._retained_by_mcp[mcp_session_key] = retained
                    self._retained_by_id[retained.id] = retained
                    session.operation_lock.release()
                    if starting_checkpoint_written:
                        try:
                            self._write_session_metadata(retained)
                        except Exception:
                            retained.state = STATE_UNKNOWN
                            retained.retention_expires_at = None
                            retained.touch()
                            self._retained_by_mcp.pop(mcp_session_key, None)
                            self._retained_by_id.pop(retained.id, None)
                            self._orphaned_by_root[
                                str(retained.profile_dir.parent.resolve())
                            ] = retained
                            raise
                    raise
                start_retained = True
            else:
                start_retained = False
            if not start_retained:
                template = (
                    self._read_template(normalized_template_name)
                    if normalized_template_name is not None
                    else None
                )
                if template is not None:
                    self._require_template_compatible(template)
                session_id = secrets.token_hex(6)
                profile_dir = self._sessions_root() / session_id / "profile"
                session = self._start_browser_session_locked(
                    mcp_session_key=mcp_session_key,
                    profile_dir=profile_dir,
                    template_name=normalized_template_name,
                    template_version_id=template.version_id if template else None,
                    session_id=session_id,
                    profile_source=template.profile_dir if template else None,
                )
        try:
            self.controller.wait_until_ready(
                session,
                timeout_sec=self.config.startup_timeout_sec,
            )
            with self._lock:
                session.state = STATE_RUNNING
                session.touch()
                self._write_session_metadata(session)
        except Exception:
            with self._lock:
                session.state = STATE_UNKNOWN
                session.touch()
            self.launcher.stop(session)
            if start_retained:
                session.state = STATE_STOPPED_RETAINED
                if not session.retention_expires_at:
                    session.retention_expires_at = utc_after(
                        self.config.retention_ttl_seconds
                    )
                session.touch()
                self._release_port_leases(session)
                self._write_session_metadata(session)
            else:
                session.state = STATE_EXPIRED
                session.touch()
                try:
                    self._delete_session_runtime(session)
                except Exception:
                    session.state = STATE_UNKNOWN
                    session.touch()
                    raise
            with self._lock:
                if self._sessions_by_mcp.get(mcp_session_key) is session:
                    self._sessions_by_mcp.pop(mcp_session_key, None)
                if start_retained:
                    self._retained_by_mcp[mcp_session_key] = session
                    self._retained_by_id[session.id] = session
            raise
        finally:
            session.operation_lock.release()
        return session

    def _owned_runtime_collection_root(self, collection_name: str) -> Path:
        runtime_root = self.config.runtime_root
        if runtime_root.exists():
            if runtime_root.is_symlink() or not runtime_root.is_dir():
                raise BrowserBrokerError(
                    "Browser runtime root must be a real owned directory"
                )
        else:
            runtime_root.mkdir(parents=True, exist_ok=True)
        collection_root = runtime_root / collection_name
        if collection_root.exists():
            if collection_root.is_symlink() or not collection_root.is_dir():
                raise BrowserBrokerError(
                    f"Browser {collection_name} root must be a real owned directory"
                )
        else:
            collection_root.mkdir(mode=0o700, exist_ok=True)
            if collection_root.is_symlink() or not collection_root.is_dir():
                raise BrowserBrokerError(
                    f"Browser {collection_name} root must be a real owned directory"
                )
        if collection_root.resolve().parent != runtime_root.resolve():
            raise BrowserBrokerError(
                f"Browser {collection_name} root resolves outside the runtime root"
            )
        return collection_root

    def _sessions_root(self) -> Path:
        return self._owned_runtime_collection_root("sessions")

    def _templates_root(self) -> Path:
        return self._owned_runtime_collection_root("templates")

    def _template_dir(self, name: str) -> Path:
        normalized = validate_template_name(name)
        templates_root = self._templates_root()
        if templates_root.exists() and (
            templates_root.is_symlink() or not templates_root.is_dir()
        ):
            raise BrowserBrokerError(
                "Template root must be a real owned directory",
                code=ERROR_TEMPLATE_CONFLICT,
            )
        template_dir = templates_root / normalized
        if template_dir.is_symlink():
            raise BrowserBrokerError(
                "Template directory symlinks are not permitted",
                code=ERROR_TEMPLATE_CONFLICT,
            )
        if (
            template_dir.exists()
            and not template_dir.is_dir()
        ) or template_dir.resolve().parent != templates_root.resolve():
            raise BrowserBrokerError(
                "Template directory resolves outside the owned template root",
                code=ERROR_TEMPLATE_CONFLICT,
            )
        return template_dir

    def _template_collection_root(
        self,
        template_name: str,
        collection_name: str,
        *,
        create: bool,
    ) -> Path:
        if collection_name not in {"drafts", "staging", "versions"}:
            raise BrowserBrokerError("Unknown template collection")
        template_dir = self._template_dir(template_name)
        collection_root = template_dir / collection_name
        if collection_root.exists():
            if collection_root.is_symlink() or not collection_root.is_dir():
                raise BrowserBrokerError(
                    f"Template {collection_name} root must be a real owned directory",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
        elif create:
            template_dir.mkdir(parents=True, exist_ok=True)
            collection_root.mkdir(mode=0o700, exist_ok=True)
            if collection_root.is_symlink() or not collection_root.is_dir():
                raise BrowserBrokerError(
                    f"Template {collection_name} root must be a real owned directory",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
        if collection_root.resolve().parent != template_dir.resolve():
            raise BrowserBrokerError(
                f"Template {collection_name} root resolves outside its template",
                code=ERROR_TEMPLATE_CONFLICT,
            )
        return collection_root

    def _try_acquire_socket_lease(
        self,
        *,
        namespace: str,
        identity: str,
        blocking: bool,
    ) -> socket.socket | None:
        lease_identity = (
            f"global-host-port\0{identity}"
            if namespace == "host-port"
            else (
                f"{self.config.installation_id}\0"
                f"{self.config.runtime_root.resolve()}\0{identity}"
            )
        )
        digest = hashlib.sha256(lease_identity.encode("utf-8")).hexdigest()
        address = f"\0tokimon-rb-{namespace}-{digest}"
        deadline = time.monotonic() + MAX_LEASE_WAIT_SECONDS
        while True:
            lease = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                lease.bind(address)
                return lease
            except OSError as exc:
                lease.close()
                if exc.errno != errno.EADDRINUSE:
                    raise BrowserBrokerError(
                        f"Could not acquire {namespace} browser ownership lease"
                    ) from exc
                if not blocking:
                    return None
                if self._shutdown_event.is_set() or time.monotonic() >= deadline:
                    return None
                self._shutdown_event.wait(0.01)

    def _try_acquire_session_lease(
        self,
        session: BrowserSession,
        *,
        create_root: bool = False,
        allow_missing_root: bool = False,
    ) -> bool:
        if session.ownership_lease is not None:
            return True
        session_root = session.profile_dir.parent
        if create_root:
            session_root.mkdir(parents=True, exist_ok=True)
        if session_root.is_symlink() or (
            not session_root.is_dir() and not allow_missing_root
        ):
            return False
        lease = self._try_acquire_socket_lease(
            namespace="session",
            identity=str(session_root.resolve()),
            blocking=False,
        )
        if lease is None:
            return False
        session.ownership_lease = lease
        return True

    def _release_session_lease(self, session: BrowserSession) -> None:
        lease = session.ownership_lease
        if lease is None:
            return
        session.ownership_lease = None
        lease.close()

    def _try_acquire_mcp_lease(self, session: BrowserSession) -> bool:
        if session.mcp_ownership_lease is not None:
            return True
        lease = self._try_acquire_socket_lease(
            namespace="mcp",
            identity=session.mcp_session_key,
            blocking=False,
        )
        if lease is None:
            return False
        session.mcp_ownership_lease = lease
        return True

    def _release_mcp_lease(self, session: BrowserSession) -> None:
        lease = session.mcp_ownership_lease
        if lease is None:
            return
        session.mcp_ownership_lease = None
        lease.close()

    def _acquire_available_port_lease(
        self,
        port_range: tuple[int, int],
        *,
        excluded_ports: set[int],
        attempts: int = 300,
    ) -> tuple[int, socket.socket]:
        low, high = port_range
        for _ in range(attempts):
            port = random.randint(low, high)
            if port in excluded_ports:
                continue
            lease = self._try_acquire_socket_lease(
                namespace="host-port",
                identity=str(port),
                blocking=False,
            )
            if lease is None:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    lease.close()
                    continue
            return port, lease
        raise BrowserBrokerError(f"No leased port is available in range {low}-{high}")

    def _release_port_leases(self, session: BrowserSession) -> None:
        leases = session.port_ownership_leases
        session.port_ownership_leases = []
        for lease in leases:
            lease.close()

    def _try_acquire_retained_leases(self, session: BrowserSession) -> bool:
        acquired_mcp = session.mcp_ownership_lease is None
        if not self._try_acquire_mcp_lease(session):
            return False
        acquired_session = session.ownership_lease is None
        try:
            has_session_lease = self._try_acquire_session_lease(session)
        except Exception:
            if acquired_mcp:
                self._release_mcp_lease(session)
            raise
        if not has_session_lease:
            if acquired_mcp:
                self._release_mcp_lease(session)
            return False
        if session.template_role in {
            TEMPLATE_ROLE_EDITING,
            TEMPLATE_ROLE_STAGING,
        } and session.template_editor_lease is None:
            try:
                editor_lease = self._try_acquire_template_editor_lease(
                    session.template_name or ""
                )
            except Exception:
                if acquired_session:
                    self._release_session_lease(session)
                if acquired_mcp:
                    self._release_mcp_lease(session)
                raise
            if editor_lease is None:
                if acquired_session:
                    self._release_session_lease(session)
                if acquired_mcp:
                    self._release_mcp_lease(session)
                return False
            session.template_editor_lease = editor_lease
        return True

    def _try_acquire_orphan_leases(self, session: BrowserSession) -> bool:
        acquired_session = session.ownership_lease is None
        if not self._try_acquire_session_lease(session):
            return False
        if session.template_role in {
            TEMPLATE_ROLE_EDITING,
            TEMPLATE_ROLE_STAGING,
        } and session.template_editor_lease is None:
            try:
                editor_lease = self._try_acquire_template_editor_lease(
                    session.template_name or ""
                )
            except Exception:
                if acquired_session:
                    self._release_session_lease(session)
                raise
            if editor_lease is None:
                if acquired_session:
                    self._release_session_lease(session)
                return False
            session.template_editor_lease = editor_lease
        return True

    @contextlib.contextmanager
    def _template_operation(self, template_name: str) -> Any:
        name = validate_template_name(template_name)
        held = getattr(self._template_operation_local, "held", None)
        if held is None:
            held = {}
            self._template_operation_local.held = held
        existing = held.get(name)
        if existing is not None:
            held[name] = (existing[0] + 1, existing[1])
            try:
                yield
            finally:
                depth, existing_lease = held[name]
                held[name] = (depth - 1, existing_lease)
            return
        lease = self._try_acquire_socket_lease(
            namespace="template-operation",
            identity=name,
            blocking=True,
        )
        if lease is None:
            raise BrowserBrokerError(
                "Template operation is busy or broker shutdown interrupted lease acquisition",
                code=ERROR_TEMPLATE_CONFLICT,
            )
        try:
            held[name] = (1, lease)
            yield
        finally:
            held.pop(name, None)
            lease.close()

    def _try_acquire_template_editor_lease(
        self,
        template_name: str,
    ) -> Any | None:
        return self._try_acquire_socket_lease(
            namespace="template-editor",
            identity=validate_template_name(template_name),
            blocking=False,
        )

    def _release_template_editor_lease(self, session: BrowserSession) -> None:
        lease = session.template_editor_lease
        if lease is None:
            return
        session.template_editor_lease = None
        lease.close()

    def _read_template(self, name: str, *, version_id: str | None = None) -> BrowserTemplate:
        normalized = validate_template_name(name)
        template_dir = self._template_dir(normalized)
        legacy_layout = False
        templates_root = self._templates_root().resolve()
        if template_dir.is_symlink() or template_dir.resolve().parent != templates_root:
            raise BrowserBrokerError(
                "Template resolves outside the owned template root",
                code=ERROR_INVALID_ARGUMENT,
            )
        if version_id is None:
            version_dir = self._latest_template_version_dir(template_dir)
        else:
            normalized_version_id = validate_template_version_id(version_id)
            versions_root = self._template_collection_root(
                normalized,
                "versions",
                create=False,
            ).resolve()
            version_candidate = versions_root / normalized_version_id
            if version_candidate.is_symlink():
                raise BrowserBrokerError(
                    "Template version symlinks are not permitted",
                    code=ERROR_INVALID_ARGUMENT,
                )
            version_dir = version_candidate.resolve()
            if version_dir.parent != versions_root:
                raise BrowserBrokerError(
                    "Template version resolves outside its sealed versions directory",
                    code=ERROR_INVALID_ARGUMENT,
                )
        if version_dir is None:
            # Backward-compatible read for templates created by the old copy-profile flow.
            version_dir = template_dir
            legacy_layout = True
        profile_dir = version_dir / "profile"
        try:
            resolved_version_dir = version_dir.resolve(strict=True)
            resolved_profile_dir = profile_dir.resolve(strict=True)
        except OSError:
            raise BrowserBrokerError(
                f"Template not found: {normalized}",
                code=ERROR_TEMPLATE_NOT_FOUND,
            )
        if (
            version_dir.is_symlink()
            or profile_dir.is_symlink()
            or not resolved_profile_dir.is_dir()
            or resolved_profile_dir.parent != resolved_version_dir
        ):
            raise BrowserBrokerError(
                "Template profile is not a directly owned sealed profile",
                code=ERROR_INVALID_ARGUMENT,
            )
        profile_dir = resolved_profile_dir
        metadata_path = version_dir / "metadata.json"
        metadata: dict[str, object] = {}
        if not legacy_layout and not metadata_path.is_file():
            raise BrowserBrokerError(
                "Sealed template version is missing its published metadata",
                code=ERROR_INVALID_ARGUMENT,
            )
        if metadata_path.exists():
            if (
                metadata_path.is_symlink()
                or metadata_path.resolve().parent != resolved_version_dir
                or not metadata_path.is_file()
            ):
                raise BrowserBrokerError(
                    "Template metadata is not directly owned by the sealed version",
                    code=ERROR_INVALID_ARGUMENT,
                )
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise BrowserBrokerError(
                    f"Template metadata is not valid JSON: {metadata_path}",
                    code=ERROR_INVALID_ARGUMENT,
                ) from exc
            if not isinstance(payload, dict):
                raise BrowserBrokerError(
                    "Template metadata must be a JSON object",
                    code=ERROR_INVALID_ARGUMENT,
                )
            metadata = dict(payload)
        if not legacy_layout and metadata.get("state") != TEMPLATE_STATE_SEALED:
            raise BrowserBrokerError(
                "Template version metadata is not in the sealed state",
                code=ERROR_INVALID_ARGUMENT,
            )
        metadata_name = str(metadata.get("name") or normalized)
        metadata_version_id = str(metadata.get("version_id") or version_dir.name)
        if metadata_name != normalized or metadata_version_id != version_dir.name:
            raise BrowserBrokerError(
                "Template metadata identity does not match its owned filesystem path",
                code=ERROR_INVALID_ARGUMENT,
            )
        metadata.setdefault("name", normalized)
        metadata.setdefault("version_id", version_dir.name)
        metadata.setdefault("state", TEMPLATE_STATE_SEALED)
        metadata.setdefault("created_at", "")
        metadata.setdefault("updated_at", "")
        metadata.setdefault("description", "")
        return BrowserTemplate(
            name=normalized,
            version_id=str(metadata.get("version_id") or version_dir.name),
            state=str(metadata.get("state") or TEMPLATE_STATE_SEALED),
            template_dir=template_dir,
            profile_dir=profile_dir,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    def _require_template_compatible(self, template: BrowserTemplate) -> None:
        sealed_runtime_image = str(template.metadata.get("runtime_image") or "")
        if sealed_runtime_image and sealed_runtime_image != self.config.image_name:
            raise BrowserBrokerError(
                "Template was sealed for a different configured browser image: "
                f"{sealed_runtime_image}",
                code=ERROR_TEMPLATE_CONFLICT,
                details={
                    "template_runtime_image": sealed_runtime_image,
                    "configured_runtime_image": self.config.image_name,
                },
            )

    def _latest_template_version_dir(self, template_dir: Path) -> Path | None:
        versions_root = self._template_collection_root(
            template_dir.name,
            "versions",
            create=False,
        )
        if not versions_root.exists():
            return None
        versions: list[Path] = []
        for path in versions_root.iterdir():
            metadata_path = path / "metadata.json"
            if (
                path.is_symlink()
                or not TEMPLATE_VERSION_ID_RE.fullmatch(path.name)
                or not path.is_dir()
                or (path / "profile").is_symlink()
                or not (path / "profile").is_dir()
                or metadata_path.is_symlink()
                or not metadata_path.is_file()
            ):
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("state") == TEMPLATE_STATE_SEALED:
                versions.append(path)
        if not versions:
            return None
        return sorted(versions, key=lambda path: path.name)[-1]

    def _sealed_version_for_source_draft(
        self,
        template_dir: Path,
        source_draft_id: str | None,
    ) -> str | None:
        if not source_draft_id:
            return None
        versions_root = self._template_collection_root(
            template_dir.name,
            "versions",
            create=False,
        )
        if not versions_root.exists():
            return None
        for version_dir in sorted(versions_root.iterdir(), reverse=True):
            if (
                version_dir.is_symlink()
                or not TEMPLATE_VERSION_ID_RE.fullmatch(version_dir.name)
            ):
                continue
            metadata_path = version_dir / "metadata.json"
            if (
                (version_dir / "profile").is_symlink()
                or not (version_dir / "profile").is_dir()
                or metadata_path.is_symlink()
                or not metadata_path.is_file()
            ):
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(payload, dict)
                and payload.get("state") == TEMPLATE_STATE_SEALED
                and payload.get("source_draft_id") == source_draft_id
            ):
                return version_dir.name
        return None

    def _prune_overwritten_template_versions(
        self,
        template_dir: Path,
        *,
        keep_version_id: str,
    ) -> None:
        keep = validate_template_version_id(keep_version_id)
        versions_root = self._template_collection_root(
            template_dir.name,
            "versions",
            create=False,
        )
        if versions_root.exists():
            for version_dir in list(versions_root.iterdir()):
                if version_dir.name == keep:
                    continue
                if version_dir.is_symlink() or not version_dir.is_dir():
                    raise BrowserBrokerError(
                        "Template overwrite found an unsafe sealed-version entry",
                        code=ERROR_TEMPLATE_CONFLICT,
                    )
                shutil.rmtree(version_dir)
        legacy_profile = template_dir / "profile"
        if legacy_profile.exists():
            if legacy_profile.is_symlink() or not legacy_profile.is_dir():
                raise BrowserBrokerError(
                    "Template overwrite found an unsafe legacy profile",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            shutil.rmtree(legacy_profile)
        legacy_metadata = template_dir / "metadata.json"
        if legacy_metadata.exists():
            if legacy_metadata.is_symlink() or not legacy_metadata.is_file():
                raise BrowserBrokerError(
                    "Template overwrite found unsafe legacy metadata",
                    code=ERROR_TEMPLATE_CONFLICT,
                )
            legacy_metadata.unlink()

    def _write_template_metadata(self, metadata_path: Path, payload: Mapping[str, object]) -> None:
        write_json_atomic(metadata_path, payload)

    def _template_summary(self, name: str) -> dict[str, object] | None:
        normalized = validate_template_name(name)
        template_dir = self._template_dir(normalized)
        versions: list[dict[str, object]] = []
        versions_root = self._template_collection_root(
            normalized,
            "versions",
            create=False,
        )
        if versions_root.exists():
            for version_dir in sorted(versions_root.iterdir(), key=lambda path: path.name):
                if (
                    version_dir.is_symlink()
                    or not TEMPLATE_VERSION_ID_RE.fullmatch(version_dir.name)
                    or (version_dir / "profile").is_symlink()
                    or not (version_dir / "profile").is_dir()
                    or (version_dir / "metadata.json").is_symlink()
                ):
                    continue
                versions.append(self._read_template(normalized, version_id=version_dir.name).to_dict())
        elif (template_dir / "profile").exists():
            versions.append(self._read_template(normalized).to_dict())
        drafts: list[dict[str, object]] = []
        drafts_root = self._template_collection_root(
            normalized,
            "drafts",
            create=False,
        )
        if drafts_root.exists():
            for draft_dir in sorted(drafts_root.iterdir(), key=lambda path: path.name):
                if (
                    draft_dir.is_symlink()
                    or not draft_dir.is_dir()
                    or not draft_dir.name.startswith("draft-")
                ):
                    continue
                metadata_path = draft_dir / "metadata.json"
                if metadata_path.exists() and not metadata_path.is_symlink():
                    try:
                        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        payload = {}
                else:
                    payload = {}
                if isinstance(payload, dict):
                    drafts.append(dict(payload))
        if not versions and not drafts:
            return None
        return {
            "name": normalized,
            "template_dir": str(template_dir),
            "versions": versions,
            "drafts": drafts,
            "latest_version_id": versions[-1]["version_id"] if versions else None,
        }

    def _copy_profile(self, source: Path, destination: Path) -> None:
        validate_copy_source_tree(
            source, label="Browser profile copy source", ignore=profile_copy_ignore
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.copy-{secrets.token_hex(6)}"
        try:
            shutil.copytree(source, staging, ignore=profile_copy_ignore)
            if destination.exists():
                shutil.rmtree(destination)
            staging.replace(destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def _validated_session_root(self, session: BrowserSession) -> Path:
        profile_dir = session.profile_dir.resolve()
        session_root = profile_dir.parent
        if profile_dir != session_root / "profile":
            raise BrowserBrokerError(
                "Browser profile path is not an owned session profile",
                code=ERROR_BROKER,
            )
        if session.template_role in {TEMPLATE_ROLE_EDITING, TEMPLATE_ROLE_STAGING}:
            if not session.template_name or not session.template_version_id:
                raise BrowserBrokerError(
                    "Template-editing session ownership metadata is incomplete",
                    code=ERROR_BROKER,
                )
            collection_name = (
                "drafts"
                if session.template_role == TEMPLATE_ROLE_EDITING
                else "staging"
            )
            drafts_root = self._template_collection_root(
                session.template_name,
                collection_name,
                create=False,
            ).resolve()
            if (
                session_root.parent != drafts_root
                or session_root.name != session.template_version_id
            ):
                raise BrowserBrokerError(
                    "Template-editing profile resolves outside its owned draft",
                    code=ERROR_BROKER,
                )
        else:
            sessions_root = self._sessions_root().resolve()
            if session_root.parent != sessions_root or session_root.name != session.id:
                raise BrowserBrokerError(
                    "Browser profile resolves outside its owned session root",
                    code=ERROR_BROKER,
                )
        return session_root

    def _validated_session_profile(self, session: BrowserSession) -> Path:
        session_root = self._validated_session_root(session)
        profile_dir = session_root / "profile"
        if (
            profile_dir.is_symlink()
            or not profile_dir.is_dir()
            or profile_dir.resolve(strict=True).parent != session_root
        ):
            raise BrowserBrokerError(
                "Browser profile must be a real directory directly owned by its session",
                code=ERROR_BROKER,
            )
        return profile_dir

    def _owned_child_directory(self, parent: Path, child_name: str) -> Path:
        try:
            resolved_parent = parent.resolve(strict=True)
        except OSError as exc:
            raise BrowserBrokerError(
                "Owned browser artifact parent is unavailable",
                code=ERROR_BROKER,
            ) from exc
        child = parent / child_name
        if child.is_symlink():
            raise BrowserBrokerError(
                f"Owned browser artifact directory cannot be a symlink: {child_name}",
                code=ERROR_BROKER,
            )
        if child.exists():
            if not child.is_dir():
                raise BrowserBrokerError(
                    f"Owned browser artifact path is not a directory: {child_name}",
                    code=ERROR_BROKER,
                )
        else:
            child.mkdir(mode=0o700)
        try:
            resolved_child = child.resolve(strict=True)
        except OSError as exc:
            raise BrowserBrokerError(
                f"Owned browser artifact directory is unavailable: {child_name}",
                code=ERROR_BROKER,
            ) from exc
        if resolved_child.parent != resolved_parent or child.is_symlink():
            raise BrowserBrokerError(
                f"Owned browser artifact directory escapes its parent: {child_name}",
                code=ERROR_BROKER,
            )
        return child

    def _owned_artifact_usage(self, root: Path) -> tuple[int, int]:
        file_count = 0
        total_bytes = 0
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in directory_names:
                candidate = base / name
                if candidate.is_symlink():
                    raise BrowserBrokerError(
                        "Owned browser artifact tree contains a directory symlink",
                        code=ERROR_BROKER,
                    )
            for name in file_names:
                candidate = base / name
                try:
                    metadata = candidate.lstat()
                except OSError as exc:
                    raise BrowserBrokerError(
                        "Owned browser artifact tree contains an unreadable entry",
                        code=ERROR_BROKER,
                    ) from exc
                if not stat.S_ISREG(metadata.st_mode):
                    raise BrowserBrokerError(
                        "Owned browser artifact tree contains a link or special file",
                        code=ERROR_BROKER,
                    )
                file_count += 1
                total_bytes += metadata.st_size
        return file_count, total_bytes

    def _session_metadata_path(self, session: BrowserSession) -> Path:
        session_root = self._validated_session_root(session)
        if session.template_role in {TEMPLATE_ROLE_EDITING, TEMPLATE_ROLE_STAGING}:
            return session_root / "session-metadata.json"
        return session_root / "metadata.json"

    def _launch_session_container(self, session: BrowserSession) -> str:
        arguments = {
            "session_id": session.id,
            "profile_dir": session.profile_dir,
            "cdp_port": session.cdp_port,
            "novnc_backend_port": session.novnc_backend_port,
        }
        if isinstance(self.launcher, DockerSessionLauncher):
            session.container_id = None
            self._write_session_metadata(session)

            def record_container_identity(container_id: str) -> None:
                session.container_id = container_id
                self._write_session_metadata(session)

            return self.launcher.start(**arguments, container_started=record_container_identity)
        return self.launcher.start(**arguments)

    def _write_session_metadata(self, session: BrowserSession) -> None:
        metadata_path = self._session_metadata_path(session)
        payload = {
            "id": session.id,
            "mcp_session_key": session.mcp_session_key,
            "recovery_token_hash": session.recovery_token_hash,
            "state": session.state,
            "created_at": session.created_at,
            "last_used_at": session.last_used_at,
            "template_name": session.template_name,
            "template_version_id": session.template_version_id,
            "template_role": session.template_role,
            "retention_expires_at": session.retention_expires_at,
            "profile_dir": str(session.profile_dir),
            "container_id": session.container_id,
            "metadata_version": 1,
        }
        write_json_atomic(metadata_path, payload)

    def _session_from_metadata(self, metadata_path: Path) -> BrowserSession | None:
        try:
            sessions_root = self._sessions_root().resolve()
            record_root = metadata_path.parent.resolve(strict=True)
            resolved_metadata_path = metadata_path.resolve(strict=True)
        except OSError:
            return None
        if (
            metadata_path.is_symlink()
            or record_root.parent != sessions_root
            or resolved_metadata_path.parent != record_root
        ):
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("metadata_version") != 1:
            return None
        container_id = payload.get("container_id")
        if container_id is not None and (
            not isinstance(container_id, str)
            or not DOCKER_FULL_CONTAINER_ID_RE.fullmatch(container_id)
        ):
            return None
        session_id = str(payload.get("id") or metadata_path.parent.name)
        mcp_session_key = str(payload.get("mcp_session_key") or "")
        if session_id != record_root.name or not mcp_session_key:
            return None
        raw_profile_dir = Path(
            str(payload.get("profile_dir") or record_root / "profile")
        )
        try:
            profile_dir = raw_profile_dir.resolve(strict=True)
            expected_profile_dir = (record_root / "profile").resolve(strict=True)
        except OSError:
            return None
        if (
            profile_dir != expected_profile_dir
            or profile_dir.parent != record_root
            or not profile_dir.is_dir()
        ):
            return None
        state = str(payload.get("state") or STATE_STOPPED_RETAINED)
        if state == "paused_private":  # Retired checkpoints remain orphaned.
            state = STATE_UNKNOWN
        retained_states = {STATE_STOPPED_RETAINED, STATE_EXPIRED}
        recoverable_active_states = {
            STATE_STARTING,
            STATE_RUNNING,
            STATE_PAUSED_OBSERVED,
            STATE_UNKNOWN,
        }
        if state not in retained_states | recoverable_active_states:
            return None
        template_role = (
            empty_to_none(str(payload.get("template_role") or ""))
            or TEMPLATE_ROLE_RUNTIME
        )
        if template_role != TEMPLATE_ROLE_RUNTIME:
            return None
        retention_expires_at = empty_to_none(
            str(payload.get("retention_expires_at") or "")
        )
        if state in retained_states:
            if retention_expires_at is None:
                return None
            try:
                parsed_expiry = datetime.fromisoformat(retention_expires_at)
            except ValueError:
                return None
            if parsed_expiry.tzinfo is None:
                return None
            parsed_expiry = parsed_expiry.astimezone(timezone.utc)
            if parsed_expiry > datetime.now(timezone.utc) + timedelta(
                seconds=self.config.max_retention_seconds
            ):
                return None
            if state == STATE_STOPPED_RETAINED and is_expired(
                retention_expires_at
            ):
                state = STATE_EXPIRED
        elif retention_expires_at is not None:
            return None
        return BrowserSession(
            id=session_id,
            mcp_session_key=mcp_session_key,
            profile_dir=profile_dir,
            cdp_port=0,
            novnc_port=0,
            novnc_backend_port=0,
            container_name=(
                f"tokimon-rb-{self.config.installation_id}-{session_id}"
            ),
            recovery_token_hash=empty_to_none(str(payload.get("recovery_token_hash") or "")),
            container_id=container_id,
            created_at=str(payload.get("created_at") or ""),
            last_used_at=str(payload.get("last_used_at") or ""),
            template_name=empty_to_none(str(payload.get("template_name") or "")),
            template_version_id=empty_to_none(str(payload.get("template_version_id") or "")),
            template_role=template_role,
            state=state,
            retention_expires_at=retention_expires_at,
            token_exposure_mode="none",
        )

    def _template_editor_from_metadata(
        self,
        metadata_path: Path,
    ) -> BrowserSession | None:
        try:
            templates_root = self._templates_root().resolve()
            draft_root = metadata_path.parent.resolve(strict=True)
            resolved_metadata_path = metadata_path.resolve(strict=True)
        except OSError:
            return None
        if (
            metadata_path.name != "session-metadata.json"
            or metadata_path.is_symlink()
            or resolved_metadata_path.parent != draft_root
            or draft_root.parent.name != "drafts"
            or draft_root.parent.parent.parent != templates_root
            or not TEMPLATE_NAME_RE.fullmatch(draft_root.parent.parent.name)
            or not draft_root.name.startswith("draft-")
        ):
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("metadata_version") != 1:
            return None
        container_id = payload.get("container_id")
        if container_id is not None and (
            not isinstance(container_id, str)
            or not DOCKER_FULL_CONTAINER_ID_RE.fullmatch(container_id)
        ):
            return None
        session_id = str(payload.get("id") or "")
        mcp_session_key = str(payload.get("mcp_session_key") or "")
        if not session_id or not mcp_session_key:
            return None
        template_name = str(payload.get("template_name") or "")
        template_version_id = str(payload.get("template_version_id") or "")
        if (
            payload.get("template_role") != TEMPLATE_ROLE_EDITING
            or template_name != draft_root.parent.parent.name
            or template_version_id != draft_root.name
        ):
            return None
        try:
            profile_dir = Path(str(payload.get("profile_dir") or "")).resolve(
                strict=True
            )
            expected_profile_dir = (draft_root / "profile").resolve(strict=True)
        except OSError:
            return None
        if profile_dir != expected_profile_dir or profile_dir.parent != draft_root:
            return None
        state = str(payload.get("state") or "")
        if state == "paused_private":  # Retired checkpoints remain orphaned.
            state = STATE_UNKNOWN
        retained_states = {STATE_STOPPED_RETAINED, STATE_EXPIRED}
        recoverable_active_states = {
            STATE_STARTING,
            STATE_RUNNING,
            STATE_PAUSED_OBSERVED,
            STATE_UNKNOWN,
        }
        if state not in retained_states | recoverable_active_states:
            return None
        retention_expires_at = empty_to_none(
            str(payload.get("retention_expires_at") or "")
        )
        if state in retained_states:
            if retention_expires_at is None:
                return None
            try:
                parsed_expiry = datetime.fromisoformat(retention_expires_at)
            except ValueError:
                return None
            if parsed_expiry.tzinfo is None:
                return None
            parsed_expiry = parsed_expiry.astimezone(timezone.utc)
            if parsed_expiry > datetime.now(timezone.utc) + timedelta(
                seconds=self.config.max_retention_seconds
            ):
                return None
            if state == STATE_STOPPED_RETAINED and is_expired(
                retention_expires_at
            ):
                state = STATE_EXPIRED
        elif retention_expires_at is not None:
            return None
        return BrowserSession(
            id=session_id,
            mcp_session_key=mcp_session_key,
            profile_dir=profile_dir,
            cdp_port=0,
            novnc_port=0,
            novnc_backend_port=0,
            container_name=(
                f"tokimon-rb-{self.config.installation_id}-{session_id}"
            ),
            recovery_token_hash=empty_to_none(str(payload.get("recovery_token_hash") or "")),
            container_id=container_id,
            created_at=str(payload.get("created_at") or ""),
            last_used_at=str(payload.get("last_used_at") or ""),
            template_name=template_name,
            template_version_id=template_version_id,
            template_role=TEMPLATE_ROLE_EDITING,
            state=state,
            retention_expires_at=retention_expires_at,
            token_exposure_mode="none",
        )

    def _valid_session_records_on_disk(self) -> list[BrowserSession]:
        sessions: list[BrowserSession] = []
        sessions_root = self._sessions_root()
        for record_root in sorted(sessions_root.iterdir(), key=lambda path: path.name):
            if not record_root.is_dir() or record_root.is_symlink():
                continue
            metadata_path = record_root / "metadata.json"
            if not metadata_path.exists():
                continue
            session = self._session_from_metadata(metadata_path)
            if session is not None:
                sessions.append(session)

        templates_root = self._templates_root()
        for template_dir in sorted(
            templates_root.iterdir(), key=lambda path: path.name
        ):
            if (
                not template_dir.is_dir()
                or template_dir.is_symlink()
                or not TEMPLATE_NAME_RE.fullmatch(template_dir.name)
            ):
                continue
            try:
                drafts_root = self._template_collection_root(
                    template_dir.name,
                    "drafts",
                    create=False,
                )
            except BrowserBrokerError:
                continue
            if not drafts_root.exists():
                continue
            for draft_root in sorted(
                drafts_root.iterdir(), key=lambda path: path.name
            ):
                if not draft_root.is_dir() or draft_root.is_symlink():
                    continue
                metadata_path = draft_root / "session-metadata.json"
                if not metadata_path.exists():
                    continue
                session = self._template_editor_from_metadata(metadata_path)
                if session is not None:
                    sessions.append(session)
        return sessions

    def _promote_retained_candidate_locked(
        self,
        candidate: BrowserSession,
    ) -> BrowserSession | None:
        root_key = str(candidate.profile_dir.parent.resolve())
        orphan = self._orphaned_by_root.get(root_key)
        if (
            candidate.mcp_session_key in self._sessions_by_mcp
            or candidate.mcp_session_key in self._retained_by_mcp
            or candidate.id in self._retained_by_id
        ):
            return None

        if orphan is not None:
            candidate.ownership_lease = orphan.ownership_lease
            candidate.mcp_ownership_lease = orphan.mcp_ownership_lease
            candidate.template_editor_lease = orphan.template_editor_lease
            orphan.ownership_lease = None
            orphan.mcp_ownership_lease = None
            orphan.template_editor_lease = None
        try:
            acquired = self._try_acquire_retained_leases(candidate)
        except Exception:
            if orphan is not None:
                orphan.ownership_lease = candidate.ownership_lease
                orphan.mcp_ownership_lease = candidate.mcp_ownership_lease
                orphan.template_editor_lease = candidate.template_editor_lease
                candidate.ownership_lease = None
                candidate.mcp_ownership_lease = None
                candidate.template_editor_lease = None
            else:
                self._release_mcp_lease(candidate)
                self._release_session_lease(candidate)
                self._release_template_editor_lease(candidate)
            raise
        if not acquired:
            if orphan is not None:
                orphan.ownership_lease = candidate.ownership_lease
                orphan.mcp_ownership_lease = candidate.mcp_ownership_lease
                orphan.template_editor_lease = candidate.template_editor_lease
                candidate.ownership_lease = None
                candidate.mcp_ownership_lease = None
                candidate.template_editor_lease = None
                orphan.state = STATE_PEER_OWNED
            else:
                self._release_session_lease(candidate)
                self._release_template_editor_lease(candidate)
                candidate.state = STATE_PEER_OWNED
                self._orphaned_by_root[root_key] = candidate
            return None

        self._orphaned_by_root.pop(root_key, None)
        self._retained_by_mcp[candidate.mcp_session_key] = candidate
        self._retained_by_id[candidate.id] = candidate
        return candidate

    def _promote_released_retained_for_key_locked(
        self,
        mcp_session_key: str,
    ) -> BrowserSession | None:
        all_records = self._valid_session_records_on_disk()
        key_counts = Counter(
            session.mcp_session_key for session in all_records
        )
        id_counts = Counter(session.id for session in all_records)
        matching_records = [
            session
            for session in all_records
            if session.mcp_session_key == mcp_session_key
        ]
        candidates = [
            session
            for session in matching_records
            if session.state in {STATE_STOPPED_RETAINED, STATE_EXPIRED}
        ]
        roots = {
            str(session.profile_dir.parent.resolve()): session
            for session in candidates
        }
        candidates = list(roots.values())
        if not candidates:
            if matching_records:
                raise BrowserBrokerError(
                    "An active or unknown browser ownership checkpoint exists for "
                    "this MCP session key; session cleanup is required",
                    code=ERROR_BROKER,
                )
            return None
        if len(candidates) > 1 or any(
            key_counts[session.mcp_session_key] > 1
            or id_counts[session.id] > 1
            for session in candidates
        ):
            raise BrowserBrokerError(
                "Retained browser ownership metadata has a duplicate MCP key or "
                "session id; session cleanup is required",
                code=ERROR_BROKER,
            )
        promoted = self._promote_retained_candidate_locked(candidates[0])
        if promoted is None:
            raise BrowserBrokerError(
                "The retained browser session is currently owned by another broker process",
                code=ERROR_BROKER,
            )
        return promoted

    def _reconcile_retained_sessions(self) -> None:
        sessions_root = self._sessions_root()
        templates_root = self._templates_root()
        with self._lock:
            parsed_valid: list[BrowserSession] = []
            runtime_roots = [
                item
                for item in sorted(sessions_root.iterdir(), key=lambda path: path.name)
                if item.is_dir() and not item.is_symlink()
            ]
            for record_root in runtime_roots:
                metadata_path = record_root / "metadata.json"
                if metadata_path.exists():
                    session = self._session_from_metadata(metadata_path)
                    if session is not None:
                        parsed_valid.append(session)

            template_item_roots: list[tuple[Path, str, str]] = []
            for template_dir in sorted(
                templates_root.iterdir(), key=lambda path: path.name
            ):
                if (
                    not template_dir.is_dir()
                    or template_dir.is_symlink()
                    or not TEMPLATE_NAME_RE.fullmatch(template_dir.name)
                ):
                    continue
                for collection_name, template_role in (
                    ("drafts", TEMPLATE_ROLE_EDITING),
                    ("staging", TEMPLATE_ROLE_STAGING),
                ):
                    try:
                        collection_root = self._template_collection_root(
                            template_dir.name,
                            collection_name,
                            create=False,
                        )
                    except BrowserBrokerError:
                        # A collection symlink or other containment violation is
                        # intentionally not traversed. Mutating operations will
                        # preserve the runtime for explicit recovery.
                        continue
                    if not collection_root.exists():
                        continue
                    for item_root in sorted(
                        collection_root.iterdir(), key=lambda path: path.name
                    ):
                        if not item_root.is_dir() or item_root.is_symlink():
                            continue
                        template_item_roots.append(
                            (item_root, template_dir.name, template_role)
                        )
                        if template_role != TEMPLATE_ROLE_EDITING:
                            continue
                        metadata_path = item_root / "session-metadata.json"
                        if not metadata_path.exists():
                            continue
                        session = self._template_editor_from_metadata(metadata_path)
                        if session is not None:
                            parsed_valid.append(session)

            key_counts = Counter(
                session.mcp_session_key for session in parsed_valid
            )
            id_counts = Counter(session.id for session in parsed_valid)
            for session in parsed_valid:
                root_key = str(session.profile_dir.parent.resolve())
                if (
                    key_counts[session.mcp_session_key] > 1
                    or id_counts[session.id] > 1
                ):
                    session.state = STATE_ORPHANED
                    if not self._try_acquire_orphan_leases(session):
                        session.state = STATE_PEER_OWNED
                    self._orphaned_by_root[root_key] = session
                    continue
                if session.state not in {
                    STATE_STOPPED_RETAINED,
                    STATE_EXPIRED,
                }:
                    if self._try_acquire_retained_leases(session):
                        session.state = STATE_UNKNOWN
                    else:
                        session.state = STATE_PEER_OWNED
                    self._orphaned_by_root[root_key] = session
                    continue
                if not self._try_acquire_retained_leases(session):
                    session.state = STATE_PEER_OWNED
                    self._orphaned_by_root[root_key] = session
                    continue
                self._retained_by_mcp[session.mcp_session_key] = session
                self._retained_by_id[session.id] = session
                if session.state == STATE_EXPIRED:
                    self._write_session_metadata(session)

            known_roots = {
                str(session.profile_dir.parent.resolve())
                for session in (
                    list(self._retained_by_id.values())
                    + list(self._orphaned_by_root.values())
                )
            }
            for record_root in runtime_roots:
                root_key = str(record_root.resolve())
                if root_key in known_roots:
                    continue
                orphan = BrowserSession(
                    id=record_root.name,
                    mcp_session_key=f"orphan:{record_root.name}",
                    profile_dir=record_root / "profile",
                    cdp_port=0,
                    novnc_port=0,
                    novnc_backend_port=0,
                    container_name="",
                    created_at="",
                    last_used_at="",
                    state=STATE_ORPHANED,
                    token_exposure_mode="none",
                )
                if not self._try_acquire_orphan_leases(orphan):
                    orphan.state = STATE_PEER_OWNED
                self._orphaned_by_root[root_key] = orphan
            for item_root, template_name, template_role in template_item_roots:
                root_key = str(item_root.resolve())
                if root_key in known_roots or root_key in self._orphaned_by_root:
                    continue
                orphan = BrowserSession(
                    id=f"orphan-{item_root.name}",
                    mcp_session_key=f"orphan:{template_name}:{item_root.name}",
                    profile_dir=item_root / "profile",
                    cdp_port=0,
                    novnc_port=0,
                    novnc_backend_port=0,
                    container_name="",
                    created_at="",
                    last_used_at="",
                    state=STATE_ORPHANED,
                    template_name=template_name,
                    template_version_id=item_root.name,
                    template_role=template_role,
                    token_exposure_mode="none",
                )
                if not self._try_acquire_orphan_leases(orphan):
                    orphan.state = STATE_PEER_OWNED
                self._orphaned_by_root[root_key] = orphan

    def _current_session_locked(self, mcp_session_key: str) -> BrowserSession | None:
        return self._sessions_by_mcp.get(mcp_session_key) or self._retained_by_mcp.get(
            mcp_session_key
        )

    def _session_is_indexed_locked(self, session: BrowserSession) -> bool:
        return (
            self._sessions_by_mcp.get(session.mcp_session_key) is session
            or (
                self._retained_by_mcp.get(session.mcp_session_key) is session
                and self._retained_by_id.get(session.id) is session
            )
            or self._orphaned_by_root.get(
                str(session.profile_dir.parent.resolve())
            )
            is session
        )

    def _all_sessions_locked(self) -> list[BrowserSession]:
        sessions = (
            list(self._sessions_by_mcp.values())
            + list(self._retained_by_id.values())
            + list(self._orphaned_by_root.values())
        )
        return sorted(sessions, key=lambda item: (item.created_at, item.id))

    def _owned_session_count_locked(self) -> int:
        return sum(session.state != STATE_PEER_OWNED for session in self._all_sessions_locked())

    def _retained_session_or_error_locked(self, mcp_session_key: str) -> BrowserSession:
        session = self._retained_by_mcp.get(mcp_session_key)
        if session is None:
            raise BrowserBrokerError(
                "No retained browser session exists for this MCP session",
                code=ERROR_SESSION_NOT_FOUND,
            )
        if session.state == STATE_STOPPED_RETAINED and is_expired(
            session.retention_expires_at
        ):
            session.state = STATE_EXPIRED
            self._write_session_metadata(session)
        return session

    def _remove_retained_session_locked(self, session: BrowserSession) -> bool:
        if (
            self._retained_by_mcp.get(session.mcp_session_key) is not session
            or self._retained_by_id.get(session.id) is not session
        ):
            return False
        self._retained_by_mcp.pop(session.mcp_session_key, None)
        self._retained_by_id.pop(session.id, None)
        return True

    def _restore_retained_template_version_locked(self, session: BrowserSession) -> None:
        if not session.template_name or session.template_version_id:
            return
        try:
            template = self._read_template(session.template_name)
        except BrowserBrokerError:
            return
        session.template_version_id = template.version_id

    def _session_or_error(self, mcp_session_key: str) -> BrowserSession:
        with self._lock:
            session = self._sessions_by_mcp.get(mcp_session_key)
        if session is None:
            raise BrowserBrokerError(
                "No browser session exists for this MCP session; call browser_ensure first",
                code=ERROR_SESSION_NOT_FOUND,
            )
        return session

    def _used_ports_locked(self) -> set[int]:
        ports: set[int] = set()
        for session in self._sessions_by_mcp.values():
            ports.add(session.cdp_port)
            ports.add(session.novnc_port)
            ports.add(session.novnc_backend_port)
        return ports


def mcp_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "browser_whoami",
            "description": "Report the current MCP session and associated browser session.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_ensure",
            "description": (
                "Start or reuse the current MCP session browser. Keep the returned recovery_token "
                "in this task's transient context for browser_session_recover after a reconnect. "
                "Explicit recreation deletes and replaces its runtime."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_url": {"type": "string"},
                    "template_name": {"type": "string"},
                    "recreate": {"type": "boolean"},
                },
            },
        },
        {
            "name": "browser_current_session_status",
            "description": "Return redacted status for the current MCP session's browser session.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_template_list",
            "description": "List reusable browser profile templates available to this broker.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_template_create",
            "description": "Start a template-editing browser for human setup. Seal it before use.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["template_name"],
                "properties": {
                    "template_name": {"type": "string"},
                    "description": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
            },
        },
        {
            "name": "browser_template_seal",
            "description": "Stop the current template editor and seal its profile as an immutable cloneable version.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["template_name"],
                "properties": {"template_name": {"type": "string"}, "description": {"type": "string"}},
            },
        },
        {
            "name": "browser_template_refresh",
            "description": "Create a new editing draft from a sealed template version.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["template_name"],
                "properties": {"template_name": {"type": "string"}, "version_id": {"type": "string"}},
            },
        },
        {
            "name": "browser_template_delete",
            "description": "Delete a named reusable browser profile template from local storage.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["template_name"],
                "properties": {"template_name": {"type": "string"}},
            },
        },
        {
            "name": "browser_connect_info",
            "description": "Return token-gated noVNC and optional SSH tunnel details for human viewing or takeover.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_session_list",
            "description": "List redacted broker-known browser sessions.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_navigate",
            "description": "Navigate this session's browser to a URL.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
        },
        {
            "name": "browser_snapshot",
            "description": "Return page URL, title, text, and a compact interactive element list.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_text_chars": {"type": "integer", "minimum": 1, "maximum": 100000},
                    "max_elements": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
            },
        },
        {
            "name": "browser_artifact",
            "description": "Save rendered page content as sanitized HTML (kind=html), or download a linked file of any type (kind=download, ref; same-origin, no redirects). Returns file metadata, not content.",
            "inputSchema": {"type": "object", "additionalProperties": False, "required": ["kind"], "properties": {
                "kind": {"type": "string", "enum": ["html", "download"]},
                "scope_selector": {"type": "string"}, "ref": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 16777216},
            }},
        },
        {
            "name": "browser_evaluate",
            "description": "Evaluate JavaScript in the active page and return its value.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}, "await_promise": {"type": "boolean"}},
            },
        },
        {
            "name": "browser_screenshot",
            "description": "Capture the current viewport to a session screenshot artifact and return its path.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"format": {"type": "string", "enum": ["png", "jpeg"]}},
            },
        },
        {
            "name": "browser_click",
            "description": "Click an element by snapshot ref.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref"],
                "properties": {
                    "ref": {"type": "string"},
                    "pacing": {"type": "string", "enum": ["human", "fast", "batch"]},
                },
            },
        },
        {
            "name": "browser_type",
            "description": "Type into an element by snapshot ref, using human pacing by default.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "text"],
                "properties": {
                    "ref": {"type": "string"},
                    "text": {"type": "string"},
                    "clear": {"type": "boolean"},
                    "pacing": {"type": "string", "enum": ["human", "fast", "batch"]},
                },
            },
        },
        {
            "name": "browser_upload_file",
            "description": "Stage a bounded file from an absolute host path in the task workspace and upload it through an input ref.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "path"],
                "properties": {
                    "ref": {"type": "string"},
                    "path": {"type": "string"},
                    "pacing": {"type": "string", "enum": ["human", "fast", "batch"]},
                },
            },
        },
        {
            "name": "browser_backend_request",
            "description": "Run a same-origin backend request from the active page context.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["method", "path"],
                "properties": {
                    "method": {"type": "string"},
                    "path": {"type": "string"},
                    "body": {"type": "object"},
                    "max_response_chars": {"type": "integer", "minimum": 1, "maximum": 10000000},
                    "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 60000},
                    "pacing": {"type": "string", "enum": ["human", "fast", "batch"]},
                },
            },
        },
        {
            "name": "browser_press_key",
            "description": "Press a key in the active page or a referenced element.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key"],
                "properties": {
                    "key": {"type": "string"},
                    "pacing": {"type": "string", "enum": ["human", "fast", "batch"]},
                },
            },
        },
        {
            "name": "browser_wait",
            "description": "Wait for time or text in the active page.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "seconds": {"type": "number", "minimum": 0, "maximum": 10.0},
                    "text": {"type": "string"},
                    "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 10000},
                    "pacing": {"type": "string", "enum": ["human", "fast", "batch"]},
                },
            },
        },
        {
            "name": "browser_new_tab",
            "description": "Open and activate a new tab.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {"url": {"type": "string"}}},
        },
        {
            "name": "browser_switch_tab",
            "description": "Switch to a tab in this session's browser.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tab_id"],
                "properties": {"tab_id": {"type": "string"}},
            },
        },
        {
            "name": "browser_close_tab",
            "description": "Close a tab in this session's browser.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"tab_id": {"type": "string"}},
            },
        },
        {
            "name": "browser_pause_for_human",
            "description": (
                "Pause actions for human takeover; keep page observation available."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reason": {"type": "string"},
                },
            },
        },
        {
            "name": "browser_resume",
            "description": "Resume this session after human takeover.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_session_extend_retention",
            "description": "Extend retention for the current stopped retained browser session.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ttl_seconds"],
                "properties": {"ttl_seconds": {"type": "integer", "minimum": 1}},
            },
        },
        {
            "name": "browser_session_recover",
            "description": (
                "Recover this task's stopped browser or template draft after MCP reconnect, "
                "using its last recovery_token. The new MCP session must be empty. "
                "Keep the replacement recovery_token returned by this call. "
                "Then use browser_ensure to restart an ordinary browser or seal the retained draft."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["recovery_token"],
                "properties": {"recovery_token": {"type": "string"}},
            },
        },
        {
            "name": "browser_session_cleanup_artifacts",
            "description": (
                "Clear staged uploads, screenshots and saved artifacts from the current stopped browser. "
                "Use browser_stop first; login data and browser-managed downloads are preserved."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
        {
            "name": "browser_session_delete_retained",
            "description": "Delete this session's retained browser profile.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_session_cleanup_expired",
            "description": "Delete expired retained browser sessions owned by this broker.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_session_cleanup_orphaned",
            "description": "Clean verified orphaned browser resources after reconciliation.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_diagnostics",
            "description": "Return redacted technical diagnostics for the current browser session.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "browser_stop",
            "description": "Stop this session's browser and retain its profile for later restoration.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
    ]


class MCPServer:
    def __init__(self, broker: RemoteBrowserBroker | None = None) -> None:
        self.broker = broker or RemoteBrowserBroker()

    def handle(
        self,
        request: Mapping[str, Any],
        *,
        mcp_session_key: str,
    ) -> dict[str, Any] | None:
        return handle_request(
            request, server_name=SERVER_NAME, server_version=SERVER_VERSION,
            protocol_version=SUPPORTED_PROTOCOL_VERSION,
            list_tools=mcp_tools,
            call_tool=lambda params: self._call_tool(params, mcp_session_key),
        )

    def _call_tool(
        self,
        params: Mapping[str, Any],
        mcp_session_key: str,
    ) -> dict[str, Any]:
        name = str(params.get("name", ""))
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return tool_result({"error": "arguments must be an object"}, is_error=True)
        try:
            if name == "browser_artifact":
                schema = next(tool["inputSchema"] for tool in mcp_tools() if tool["name"] == name)
                if set(arguments) - set(schema["properties"]) or set(schema.get("required", [])) - set(arguments):
                    raise BrowserBrokerError("Unsupported or missing tool arguments", code=ERROR_INVALID_ARGUMENT)
                types = {"string": str, "integer": int}
                for key, value in arguments.items():
                    expected = schema["properties"][key]["type"]
                    if not isinstance(value, types[expected]) or (expected == "integer" and isinstance(value, bool)):
                        raise BrowserBrokerError(f"Invalid type for {key}", code=ERROR_INVALID_ARGUMENT)
                return tool_result(self.broker.artifact(mcp_session_key, **arguments))
            if name == "browser_whoami":
                return tool_result(self.broker.whoami(mcp_session_key))
            if name == "browser_ensure":
                return tool_result(
                    self.broker.ensure(
                        mcp_session_key,
                        start_url=optional_string(arguments, "start_url"),
                        template_name=optional_string(arguments, "template_name"),
                        recreate=optional_bool(arguments, "recreate", False),
                    )
                )
            if name == "browser_current_session_status":
                return tool_result(self.broker.current_session_status(mcp_session_key))
            if name == "browser_template_list":
                return tool_result(self.broker.list_templates())
            if name == "browser_template_create":
                return tool_result(
                    self.broker.create_template(
                        mcp_session_key,
                        template_name=required_string(arguments, "template_name"),
                        description=optional_string(arguments, "description"),
                        overwrite=optional_bool(arguments, "overwrite", False),
                    )
                )
            if name == "browser_template_seal":
                return tool_result(
                    self.broker.seal_template(
                        mcp_session_key,
                        template_name=required_string(arguments, "template_name"),
                        description=optional_string(arguments, "description"),
                    )
                )
            if name == "browser_template_refresh":
                return tool_result(
                    self.broker.refresh_template(
                        mcp_session_key,
                        template_name=required_string(arguments, "template_name"),
                        version_id=optional_string(arguments, "version_id"),
                    )
                )
            if name == "browser_template_delete":
                return tool_result(
                    self.broker.delete_template(
                        template_name=required_string(arguments, "template_name"),
                    )
                )
            if name == "browser_connect_info":
                return tool_result(self.broker.connect_info(mcp_session_key))
            if name == "browser_session_list":
                return tool_result(self.broker.session_list())
            if name == "browser_navigate":
                return tool_result(
                    self.broker.navigate(
                        mcp_session_key,
                        required_string(arguments, "url"),
                    )
                )
            if name == "browser_snapshot":
                return tool_result(
                    self.broker.snapshot(
                        mcp_session_key,
                        max_text_chars=optional_positive_int(arguments, "max_text_chars", 20000),
                        max_elements=optional_positive_int(arguments, "max_elements", 200),
                    )
                )
            if name == "browser_evaluate":
                return tool_result(
                    self.broker.evaluate(
                        mcp_session_key,
                        expression=required_string(arguments, "expression"),
                        await_promise=optional_bool(arguments, "await_promise", True),
                    )
                )
            if name == "browser_screenshot":
                if "full_page" in arguments:
                    raise BrowserBrokerError(
                        "full_page screenshot capture is not supported by browser_screenshot; "
                        "only viewport capture is available",
                        code=ERROR_INVALID_ARGUMENT,
                    )
                unsupported = set(arguments) - {"format"}
                if unsupported:
                    raise BrowserBrokerError(
                        "Unsupported browser_screenshot argument(s): "
                        + ", ".join(sorted(unsupported)),
                        code=ERROR_INVALID_ARGUMENT,
                    )
                return tool_result(
                    self.broker.screenshot(
                        mcp_session_key,
                        image_format=optional_string(arguments, "format") or "png",
                    )
                )
            if name == "browser_click":
                return tool_result(
                    self.broker.click(
                        mcp_session_key,
                        ref=required_string(arguments, "ref"),
                        pacing=optional_string(arguments, "pacing"),
                    )
                )
            if name == "browser_type":
                return tool_result(
                    self.broker.type_text(
                        mcp_session_key,
                        ref=required_string(arguments, "ref"),
                        text=required_literal_string(arguments, "text"),
                        clear=optional_bool(arguments, "clear", False),
                        pacing=optional_string(arguments, "pacing"),
                    )
                )
            if name == "browser_upload_file":
                return tool_result(
                    self.broker.upload_file(
                        mcp_session_key,
                        ref=required_string(arguments, "ref"),
                        path=required_string(arguments, "path"),
                        pacing=optional_string(arguments, "pacing"),
                    )
                )
            if name == "browser_backend_request":
                properties = next(tool["inputSchema"]["properties"] for tool in mcp_tools() if tool["name"] == name)
                if set(arguments) - set(properties):
                    raise BrowserBrokerError("Unsupported tool arguments", code=ERROR_INVALID_ARGUMENT)
                return tool_result(
                    self.broker.backend_request(
                        mcp_session_key,
                        method=required_string(arguments, "method"),
                        path=required_string(arguments, "path"),
                        body=optional_mapping(arguments, "body"),
                        max_response_chars=optional_positive_int(
                            arguments,
                            "max_response_chars",
                            DEFAULT_BACKEND_RESPONSE_MAX_CHARS,
                        ),
                        timeout_ms=optional_positive_int(
                            arguments,
                            "timeout_ms",
                            DEFAULT_BACKEND_REQUEST_TIMEOUT_MS,
                        ),
                        pacing=optional_string(arguments, "pacing"),
                    )
                )
            if name == "browser_press_key":
                return tool_result(
                    self.broker.press_key(
                        mcp_session_key,
                        key=required_string(arguments, "key"),
                        pacing=optional_string(arguments, "pacing"),
                    )
                )
            if name == "browser_wait":
                return tool_result(
                    self.broker.wait(
                        mcp_session_key,
                        seconds=optional_non_negative_float(arguments, "seconds"),
                        text=optional_string(arguments, "text"),
                        timeout_ms=optional_positive_int(arguments, "timeout_ms", 5000),
                        pacing=optional_string(arguments, "pacing"),
                    )
                )
            if name == "browser_new_tab":
                return tool_result(
                    self.broker.new_tab(
                        mcp_session_key,
                        url=optional_string(arguments, "url"),
                    )
                )
            if name == "browser_switch_tab":
                return tool_result(
                    self.broker.switch_tab(
                        mcp_session_key,
                        tab_id=required_string(arguments, "tab_id"),
                    )
                )
            if name == "browser_close_tab":
                return tool_result(
                    self.broker.close_tab(
                        mcp_session_key,
                        tab_id=optional_string(arguments, "tab_id"),
                    )
                )
            if name == "browser_pause_for_human":
                if set(arguments) - {"reason"}:
                    raise BrowserBrokerError(
                        "browser_pause_for_human accepts only an optional reason",
                        code=ERROR_INVALID_ARGUMENT,
                    )
                return tool_result(
                    self.broker.pause_for_human(
                        mcp_session_key,
                        reason=optional_string(arguments, "reason"),
                    )
                )
            if name == "browser_resume":
                return tool_result(self.broker.resume(mcp_session_key))
            if name == "browser_session_extend_retention":
                return tool_result(
                    self.broker.extend_retained_session(
                        mcp_session_key,
                        ttl_seconds=required_positive_int(arguments, "ttl_seconds"),
                    )
                )
            if name == "browser_session_recover":
                if set(arguments) - {"recovery_token"}:
                    raise BrowserBrokerError(
                        "browser_session_recover accepts only recovery_token",
                        code=ERROR_INVALID_ARGUMENT,
                    )
                return tool_result(
                    self.broker.recover_session(
                        mcp_session_key,
                        recovery_token=required_string(arguments, "recovery_token"),
                    )
                )
            if name == "browser_session_cleanup_artifacts":
                return tool_result(self.broker.cleanup_session_artifacts(mcp_session_key))
            if name == "browser_session_delete_retained":
                return tool_result(self.broker.delete_retained_session(mcp_session_key))
            if name == "browser_session_cleanup_expired":
                return tool_result(self.broker.cleanup_expired_sessions())
            if name == "browser_session_cleanup_orphaned":
                return tool_result(self.broker.cleanup_orphaned_sessions())
            if name == "browser_diagnostics":
                return tool_result(self.broker.diagnostics(mcp_session_key))
            if name == "browser_stop":
                return tool_result(self.broker.stop(mcp_session_key))
            return tool_result({"error": f"Unknown tool: {name}"}, is_error=True)
        except BrowserBrokerError as exc:
            return tool_result(exc.to_payload(), is_error=True)


def required_literal_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise BrowserBrokerError(f"{name} must be a string", code=ERROR_INVALID_ARGUMENT)
    return value


def required_string(arguments: Mapping[str, Any], name: str) -> str:
    value = optional_string(arguments, name)
    if value is None:
        raise BrowserBrokerError(f"{name} is required", code=ERROR_INVALID_ARGUMENT)
    return value


def optional_string(arguments: Mapping[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def optional_bool(arguments: Mapping[str, Any], name: str, default: bool) -> bool:
    value = arguments.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise BrowserBrokerError(f"{name} must be a boolean", code=ERROR_INVALID_ARGUMENT)


def optional_mapping(arguments: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    raise BrowserBrokerError(f"{name} must be an object", code=ERROR_INVALID_ARGUMENT)


def optional_positive_int(arguments: Mapping[str, Any], name: str, default: int) -> int:
    value = arguments.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BrowserBrokerError(f"{name} must be an integer", code=ERROR_INVALID_ARGUMENT) from exc
    if parsed <= 0:
        raise BrowserBrokerError(
            f"{name} must be greater than zero",
            code=ERROR_INVALID_ARGUMENT,
        )
    return parsed


def optional_non_negative_float(arguments: Mapping[str, Any], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BrowserBrokerError(f"{name} must be a number", code=ERROR_INVALID_ARGUMENT) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise BrowserBrokerError(
            f"{name} must be a finite non-negative number",
            code=ERROR_INVALID_ARGUMENT,
        )
    return parsed


def required_positive_int(arguments: Mapping[str, Any], name: str) -> int:
    if arguments.get(name) is None:
        raise BrowserBrokerError(f"{name} is required", code=ERROR_INVALID_ARGUMENT)
    return optional_positive_int(arguments, name, 1)


def encode_sse(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(payload, ensure_ascii=False)
    return f"event: message\ndata: {text}\n\n".encode("utf-8")


def make_handler(
    server: MCPServer,
) -> type[BaseHTTPRequestHandler]:
    class RemoteBrowserMCPHandler(BaseHTTPRequestHandler):
        server_version = "TokimonRemoteBrowserBroker/0.1"

        def do_GET(self) -> None:  # noqa: N802 - http.server API.
            if not self._local_request_allowed():
                return
            if self.path == "/health":
                log_runtime_event(
                    configure_runtime_logger(),
                    logging.INFO,
                    "health_requested",
                    correlation_id=self._session_key(None),
                    method="GET",
                    path=self.path,
                    client=self.client_address[0],
                )
                self._send_json(health_payload(server.broker.config))
                return
            parsed_path = urllib.parse.urlparse(self.path)
            if parsed_path.path == "/mcp":
                session_key = self._session_key(None)
                log_runtime_event(
                    configure_runtime_logger(),
                    logging.INFO,
                    "event_stream_opened",
                    correlation_id=session_key,
                    method="GET",
                    path=parsed_path.path,
                    client=self.client_address[0],
                    mcp_session_key=session_key,
                )
                self._send_event_stream(mcp_session_key=session_key)
                return
            self.send_error(404, "Not found")

        def do_DELETE(self) -> None:  # noqa: N802 - http.server API.
            if not self._local_request_allowed():
                return
            parsed_path = urllib.parse.urlparse(self.path)
            if parsed_path.path != "/mcp":
                self.send_error(404, "Not found")
                return
            session_key = self._session_key(None)
            log_runtime_event(
                configure_runtime_logger(),
                logging.INFO,
                "session_shutdown",
                correlation_id=session_key,
                method="DELETE",
                path=parsed_path.path,
                client=self.client_address[0],
                mcp_session_key=session_key,
            )
            try:
                server.broker.shutdown_transport_session(session_key)
            except BrowserBrokerError as exc:
                body = json.dumps(exc.to_payload(), ensure_ascii=False).encode("utf-8")
                self.send_response(409)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(202)
            self.send_header("mcp-session-id", session_key)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - http.server API.
            if not self._local_request_allowed():
                return
            parsed_path = urllib.parse.urlparse(self.path)
            if parsed_path.path != "/mcp":
                self.send_error(404, "Not found")
                return
            try:
                length = bounded_content_length(
                    self.headers.get("Content-Length"),
                    maximum=MAX_MCP_REQUEST_BYTES,
                )
                raw = read_http_body_with_deadline(self, length)
            except TimeoutError:
                self.send_error(408, "Request body deadline expired")
                return
            except BrowserBrokerError as exc:
                self.send_error(
                    413 if exc.details.get("request_too_large") else 400,
                    str(exc),
                )
                return
            try:
                request = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                session_key = self._session_key(None)
                log_runtime_event(
                    configure_runtime_logger(),
                    logging.WARNING,
                    "invalid_jsonrpc_request",
                    correlation_id=session_key,
                    method="POST",
                    path=parsed_path.path,
                    client=self.client_address[0],
                    mcp_session_key=session_key,
                )
                self._send_sse(
                    error_response(None, -32700, str(exc)),
                    mcp_session_key=session_key,
                )
                return
            if not isinstance(request, dict):
                session_key = self._session_key(None)
                log_runtime_event(
                    configure_runtime_logger(),
                    logging.WARNING,
                    "invalid_jsonrpc_request",
                    correlation_id=session_key,
                    method="POST",
                    path=parsed_path.path,
                    client=self.client_address[0],
                    mcp_session_key=session_key,
                )
                self._send_sse(
                    error_response(None, -32600, "Request must be an object"),
                    mcp_session_key=session_key,
                )
                return
            session_key = self._session_key(request)
            log_runtime_event(
                configure_runtime_logger(),
                logging.INFO,
                "request_received",
                correlation_id=session_key,
                request_id=request.get("id"),
                method=request.get("method"),
                path=parsed_path.path,
                client=self.client_address[0],
                mcp_session_key=session_key,
            )
            result = server.handle(
                request,
                mcp_session_key=session_key,
            )
            if result is None:
                self.send_response(202)
                self.send_header("mcp-session-id", session_key)
                self.end_headers()
                return
            self._send_sse(result, mcp_session_key=session_key)

        def _local_request_allowed(self) -> bool:
            try:
                host = urllib.parse.urlsplit("//" + self.headers.get("Host", "")).hostname
            except ValueError:
                host = None
            if (
                not is_loopback_host(self.client_address[0])
                or not is_loopback_host(str(self.server.server_address[0]))
                or not is_loopback_host(host or "")
                or self.headers.get("Origin") is not None
            ):
                self.send_error(403, "Browser MCP requires a local non-browser client")
                return False
            return True

        def _session_key(self, request: Mapping[str, Any] | None) -> str:
            header = self.headers.get("mcp-session-id")
            if header:
                return header.strip()
            return f"mcp_{secrets.token_hex(12)}"

        def _send_sse(self, payload: Mapping[str, Any], *, mcp_session_key: str) -> None:
            body = encode_sse(payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("mcp-session-id", mcp_session_key)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_event_stream(self, *, mcp_session_key: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("mcp-session-id", mcp_session_key)
            self.end_headers()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    if server.broker.wait_for_shutdown(15):
                        self.close_connection = True
                        return
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True
                return

        def _send_json(self, payload: Mapping[str, Any]) -> None:
            body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            log_runtime_event(
                configure_runtime_logger(),
                logging.INFO,
                "http_request",
                correlation_id=self.headers.get("mcp-session-id") or "http-request",
                method=self.command,
                path=self.path,
                status=format % args,
                client=self.client_address[0],
                mcp_session_key=self.headers.get("mcp-session-id"),
            )

    return RemoteBrowserMCPHandler


def serve(
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    server: MCPServer | None = None,
) -> int:
    resolved_server = server or MCPServer()
    mcp_session_key = f"mcp_{secrets.token_hex(12)}"
    logger = configure_runtime_logger()
    log_runtime_event(
        logger,
        logging.INFO,
        "server_start",
        correlation_id=mcp_session_key,
        method="serve",
        mcp_session_key=mcp_session_key,
        runtime="stdio-mcp",
    )
    shutdown_failed = False
    broker_shutdown_succeeded = False
    try:
        while True:
            line = input_stream.readline(MAX_MCP_REQUEST_BYTES + 1)
            if line == "":
                break
            oversized = (
                len(line) > MAX_MCP_REQUEST_BYTES
                or len(line.encode("utf-8")) > MAX_MCP_REQUEST_BYTES
            )
            if oversized:
                while line and not line.endswith("\n"):
                    line = input_stream.readline(MAX_MCP_REQUEST_BYTES + 1)
                result = error_response(
                    None,
                    -32600,
                    f"Request exceeds {MAX_MCP_REQUEST_BYTES} bytes",
                )
                print(
                    json.dumps(result, ensure_ascii=False),
                    file=output_stream,
                    flush=True,
                )
                continue
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                result = error_response(None, -32700, str(exc))
            else:
                if not isinstance(request, dict):
                    result = error_response(None, -32600, "Request must be an object")
                else:
                    log_runtime_event(
                        logger,
                        logging.INFO,
                        "request_received",
                        correlation_id=mcp_session_key,
                        request_id=request.get("id"),
                        method=request.get("method"),
                        mcp_session_key=mcp_session_key,
                        runtime="stdio-mcp",
                    )
                    result = resolved_server.handle(request, mcp_session_key=mcp_session_key)
            if result is not None:
                print(json.dumps(result, ensure_ascii=False), file=output_stream, flush=True)
    finally:
        if isinstance(resolved_server, MCPServer):
            resolved_server.broker.begin_shutdown()
        if isinstance(resolved_server, MCPServer):
            try:
                with resolved_server.broker._shutdown_internal_operations():
                    resolved_server.broker.shutdown_transport_session(mcp_session_key)
            except Exception as exc:  # noqa: BLE001 - continue draining other sessions.
                shutdown_failed = True
                log_runtime_event(
                    logger,
                    logging.ERROR,
                    "stdio_session_shutdown_failed",
                    correlation_id=mcp_session_key,
                    method="serve",
                    status=str(exc),
                    mcp_session_key=mcp_session_key,
                    runtime="stdio-mcp",
                )
            try:
                resolved_server.broker.shutdown_all_sessions()
            except Exception as exc:  # noqa: BLE001 - report and preserve owned failures.
                shutdown_failed = True
                log_runtime_event(
                    logger,
                    logging.ERROR,
                    "server_shutdown_failed",
                    correlation_id=mcp_session_key,
                    method="serve",
                    status=str(exc),
                    mcp_session_key=mcp_session_key,
                    runtime="stdio-mcp",
                )
            else:
                broker_shutdown_succeeded = True
            if broker_shutdown_succeeded:
                resolved_server.broker.release_shutdown_leases()
    return 2 if shutdown_failed else 0


def serve_http(*, config: BrokerConfig | None = None, broker: RemoteBrowserBroker | None = None) -> int:
    resolved_config = config or BrokerConfig.from_env()
    resolved_broker = broker or RemoteBrowserBroker(config=resolved_config)
    server = MCPServer(resolved_broker)
    httpd: ThreadingHTTPServer | None = None
    broker_shutdown_succeeded = False
    try:
        httpd = ThreadingHTTPServer(
            (resolved_config.bind_host, resolved_config.broker_port),
            make_handler(server),
        )
        httpd.daemon_threads = False
        httpd.block_on_close = True
        log_runtime_event(
            configure_runtime_logger(),
            logging.INFO,
            "server_start",
            correlation_id="server-start",
            method="serve-http",
            path=(
                f"http://{resolved_config.bind_host}:"
                f"{resolved_config.broker_port}/mcp"
            ),
        )
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        resolved_broker.begin_shutdown()
        if httpd is not None:
            try:
                httpd.server_close()
            except Exception as exc:  # noqa: BLE001 - browser cleanup must still run.
                log_runtime_event(
                    configure_runtime_logger(),
                    logging.ERROR,
                    "http_server_close_failed",
                    correlation_id="server-shutdown",
                    method="serve-http",
                    status=str(exc),
                )
        try:
            resolved_broker.shutdown_all_sessions()
        except Exception as exc:  # noqa: BLE001 - preserve failed ownership for recovery.
            log_runtime_event(
                configure_runtime_logger(),
                logging.ERROR,
                "server_shutdown_failed",
                correlation_id="server-shutdown",
                method="serve-http",
                status=str(exc),
            )
        else:
            broker_shutdown_succeeded = True
        if broker_shutdown_succeeded:
            resolved_broker.release_shutdown_leases()
    return 0


def docker_containers_with_label(
    config: BrokerConfig,
    label: str | None,
) -> tuple[list[dict[str, object]], str | None]:
    list_command = [config.docker_command, "ps", "-aq"]
    if label:
        list_command.extend(["--filter", f"label={label}"])
    try:
        listed = subprocess.run(
            list_command,
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if listed.returncode != 0:
        return [], (listed.stderr or listed.stdout or "").strip()
    container_ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not container_ids:
        return [], None
    try:
        inspected = subprocess.run(
            [config.docker_command, "inspect", *container_ids],
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if inspected.returncode != 0:
        return [], (inspected.stderr or inspected.stdout or "").strip()
    try:
        payload = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        return [], f"Docker inspect output was not JSON: {exc}"
    containers: list[dict[str, object]] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, Mapping):
            continue
        config_payload = item.get("Config") if isinstance(item.get("Config"), Mapping) else {}
        labels = config_payload.get("Labels") if isinstance(config_payload, Mapping) else {}
        state = item.get("State") if isinstance(item.get("State"), Mapping) else {}
        mounts_payload = item.get("Mounts") if isinstance(item.get("Mounts"), list) else []
        mount_sources = [
            str(mount.get("Source") or "")
            for mount in mounts_payload
            if isinstance(mount, Mapping) and mount.get("Source")
        ]
        containers.append(
            {
                "id": str(item.get("Id") or ""),
                "name": str(item.get("Name") or "").lstrip("/"),
                "image": str(config_payload.get("Image") or item.get("Image") or ""),
                "state": str(state.get("Status") or "unknown") if isinstance(state, Mapping) else "unknown",
                "labels": labels if isinstance(labels, Mapping) else {},
                "mount_sources": mount_sources,
            }
        )
    return containers, None


def docker_resource_inventory(config: BrokerConfig) -> dict[str, object]:
    tokimon_containers, tokimon_error = docker_containers_with_label(
        config,
        "tokimon.remote-browser-broker=true",
    )
    owned: list[dict[str, object]] = []
    foreign: list[dict[str, object]] = []
    for container in tokimon_containers:
        labels = container.get("labels")
        installation = (
            str(labels.get("tokimon.remote-browser-installation") or "")
            if isinstance(labels, Mapping)
            else ""
        )
        if installation == config.installation_id:
            owned.append(container)
        else:
            foreign.append(container)
    warnings: list[str] = []
    if foreign:
        warnings.append("foreign_or_unscoped_tokimon_browser_containers_present")
    if tokimon_error:
        warnings.append("tokimon_container_inventory_error")
    return {
        "owned_container_count": len(owned),
        "foreign_tokimon_container_count": len(foreign),
        "owned_containers": owned,
        "foreign_tokimon_containers": foreign,
        "warnings": warnings,
        "errors": {"tokimon": tokimon_error},
    }


def docker_image_present(config: BrokerConfig, image_name: str) -> tuple[bool, str | None]:
    try:
        inspected = subprocess.run(
            [config.docker_command, "image", "inspect", image_name],
            capture_output=True,
            text=True,
            timeout=DOCKER_INSPECT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if inspected.returncode == 0:
        return True, None
    return False, (inspected.stderr or inspected.stdout or "").strip()


def automatic_cleanup_policy(config: BrokerConfig) -> dict[str, object]:
    return {
        "enabled": config.idle_timeout_seconds > 0,
        "trigger": "session_capacity_pressure",
        "profile_deletion": "explicit_only",
        "idle_timeout_seconds": config.idle_timeout_seconds,
        "max_active_sessions": config.max_active_sessions,
        "max_total_sessions": config.max_total_sessions,
    }


def health_payload(config: BrokerConfig | None = None) -> dict[str, object]:
    resolved = config or BrokerConfig.from_env()
    docker_path = shutil.which(resolved.docker_command)
    has_image = False
    image_error = None
    if docker_path:
        has_image, image_error = docker_image_present(resolved, resolved.image_name)
        resources = docker_resource_inventory(resolved)
    else:
        resources = {
            "owned_container_count": 0,
            "foreign_tokimon_container_count": 0,
            "owned_containers": [],
            "foreign_tokimon_containers": [],
            "warnings": ["docker_missing"],
            "errors": {"tokimon": None},
        }
    return {
        "ok": bool(docker_path and IMAGE_CONTEXT_DIR.exists()),
        "server": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "protocol_version": SUPPORTED_PROTOCOL_VERSION,
        },
        "installation": {
            "id": resolved.installation_id,
            "owner_user": resolved.owner_user,
            "workspace_root": str(resolved.workspace_root),
            "default_http_port": default_broker_port(resolved.installation_id),
        },
        "docker": {
            "command": resolved.docker_command,
            "path": docker_path,
            "image": resolved.image_name,
            "image_present": has_image,
            "image_error": image_error,
            "resources": resources,
        },
        "image_context": str(IMAGE_CONTEXT_DIR),
        "runtime_root": str(resolved.runtime_root),
        "templates_root": str(resolved.runtime_root / "templates"),
        "broker_url": f"http://{resolved.bind_host}:{resolved.broker_port}/mcp",
        "port_ranges": {
            "cdp": f"{resolved.cdp_port_range[0]}-{resolved.cdp_port_range[1]}",
            "novnc": f"{resolved.novnc_port_range[0]}-{resolved.novnc_port_range[1]}",
        },
        "network_policy": network_policy_diagnostics(resolved.network_policy),
        "retention": {
            "default_ttl_seconds": resolved.retention_ttl_seconds,
            "max_retention_seconds": resolved.max_retention_seconds,
        },
        "automatic_cleanup": automatic_cleanup_policy(resolved),
        "ssh_host": resolved.ssh_host,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the remote browser broker MCP server.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Run the stdio MCP broker")

    serve_parser = subparsers.add_parser("serve-http", help="Run the HTTP MCP broker")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    health_parser = subparsers.add_parser("health", help="Show broker health")
    health_parser.add_argument("--json", action="store_true")
    subparsers.add_parser("build-image", help="Build the Docker image used by browser sessions")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BrokerConfig.from_env()
    if args.command == "serve":
        return serve()
    if args.command == "serve-http":
        if args.host is not None:
            config = replace_config(config, bind_host=args.host)
        if args.port is not None:
            config = replace_config(config, broker_port=args.port)
        return serve_http(config=config)
    if args.command == "health":
        payload = health_payload(config)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"{SERVER_NAME}: {'ok' if payload['ok'] else 'unavailable'}")
            print(f"Broker URL: {payload['broker_url']}")
            print(f"Docker: {payload['docker']['path'] or 'missing'}")
            print(f"Image present: {payload['docker']['image_present']}")
        return 0 if payload["ok"] else 1
    if args.command == "build-image":
        DockerSessionLauncher(config).build_image()
        print(json.dumps({"ok": True, "image": config.image_name}, indent=2))
        return 0
    return 1


def replace_config(config: BrokerConfig, **overrides: object) -> BrokerConfig:
    data = {
        "workspace_root": config.workspace_root,
        "runtime_root": config.runtime_root,
        "installation_id": config.installation_id,
        "owner_user": config.owner_user,
        "image_name": config.image_name,
        "docker_command": config.docker_command,
        "bind_host": config.bind_host,
        "broker_port": config.broker_port,
        "cdp_port_range": config.cdp_port_range,
        "novnc_port_range": config.novnc_port_range,
        "ssh_host": config.ssh_host,
        "public_host": config.public_host,
        "viewport": config.viewport,
        "screen": config.screen,
        "startup_timeout_sec": config.startup_timeout_sec,
        "retention_ttl_seconds": config.retention_ttl_seconds,
        "max_retention_seconds": config.max_retention_seconds,
        "max_active_sessions": config.max_active_sessions,
        "max_total_sessions": config.max_total_sessions,
        "idle_timeout_seconds": config.idle_timeout_seconds,
        "novnc_token_ttl_seconds": config.novnc_token_ttl_seconds,
        "network_policy": config.network_policy,
    }
    data.update(overrides)
    return BrokerConfig(**data)


if __name__ == "__main__":
    raise SystemExit(main())
