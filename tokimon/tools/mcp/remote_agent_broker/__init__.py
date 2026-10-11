#!/usr/bin/env python3
"""Remote agent broker MCP server.

The broker runs shell commands and installed AI CLI agents in a trusted remote
account.  Long-running work is represented as asynchronous jobs with reviewable
artifacts under .tokimon/tools/mcp/remote-agent/.
"""

from __future__ import annotations

import argparse
import codecs
import ipaddress
import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from tokimon.tools.mcp._protocol import response, error_response, handle_request, json_wire_text
from tokimon.tools._runtime import configure_logger, format_log_record, resolve_executable
from tokimon.runtime import resolve_env_runtime_path, runtime_root as repository_runtime_root, source_root
from tokimon.settings import (
    TokimonSettingsError,
    resolve_agent_execution_options,
    resolve_agent_options,
)
from tokimon.tools.mcp.remote_agent_broker.dispatch import ToolCallDispatcher, mcp_tools
from tokimon.tools.mcp.remote_agent_broker.model import (
    ACTIVE_STATUSES,
    AGENT_RUNNER_NAMES,
    CLAUDE_RUNNER_NAME,
    COPILOT_RUNNER_NAME,
    RUNNER_NAMES,
    SHELL_RUNNER_NAME,
    TERMINAL_STATUSES,
    RemoteAgentBrokerError,
    RequestContext,
    parse_utc,
    read_file_tail,
    redact_text,
    require_positive_int,
    utc_now,
)
from tokimon.tools.mcp.remote_agent_broker.storage import (
    JobArtifactManager,
    JobCleanupManager,
    JobStateStore,
)

WORKSPACE_ROOT = source_root()
REMOTE_AGENT_RUNTIME_RELATIVE_PATH = Path("tools") / "mcp" / "remote-agent"
DEFAULT_RUNTIME_ROOT = repository_runtime_root(WORKSPACE_ROOT) / REMOTE_AGENT_RUNTIME_RELATIVE_PATH

SERVER_NAME = "tokimon-remote-agent-broker"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSION = "2025-03-26"
LOGGER_NAME = "tokimon.tools.mcp.remote_agent_broker"

__all__ = [
    "BrokerConfig",
    "MCPServer",
    "RemoteAgentBroker",
    "RemoteAgentBrokerError",
    "RequestContext",
    "build_parser",
    "main",
    "mcp_tools",
    "serve",
    "serve_http",
]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8938
DEFAULT_TTL_DAYS = 7
DEFAULT_MAX_READ_BYTES = 65536
DEFAULT_SHELL_SYNC_TIMEOUT_SEC = 30
DEFAULT_CANCEL_GRACE_SEC = 5
STREAM_READ_CHARS = 4096
POST_EXIT_STREAM_DRAIN_SEC = 2.5
POST_EXIT_PROCESS_GROUP_TERM_SEC = 0.5
POST_EXIT_PROCESS_GROUP_KILL_SEC = 0.5
OUTPUT_STATE_FLUSH_INTERVAL_SEC = 0.25
MAX_STREAM_REDACTION_BUFFER_CHARS = 65536
MAX_HTTP_REQUEST_BYTES = 1024 * 1024
HTTP_READ_TIMEOUT_SEC = 10


@dataclass(frozen=True)
class BrokerConfig:
    workspace_root: Path = WORKSPACE_ROOT
    runtime_root: Path = DEFAULT_RUNTIME_ROOT
    bind_host: str = DEFAULT_HOST
    broker_port: int = DEFAULT_PORT
    ssh_host: str | None = None
    default_ttl_days: int = DEFAULT_TTL_DAYS
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES
    shell_command: str = "/bin/sh"
    shell_sync_timeout_sec: int = DEFAULT_SHELL_SYNC_TIMEOUT_SEC

    def __post_init__(self) -> None:
        require_port(self.broker_port, name="broker_port")
        require_positive_int(self.default_ttl_days, name="default_ttl_days")
        require_positive_int(self.max_read_bytes, name="max_read_bytes")
        require_positive_int(
            self.shell_sync_timeout_sec,
            name="shell_sync_timeout_sec",
        )

    @classmethod
    def from_env(cls, *, env: Mapping[str, str] | None = None) -> "BrokerConfig":
        effective_env = dict(os.environ if env is None else env)
        runtime_root = resolve_env_runtime_path(
            default_relative_path=REMOTE_AGENT_RUNTIME_RELATIVE_PATH,
            env_var="TOKIMON_RA_RUNTIME_ROOT",
            env=effective_env,
            workspace_root=WORKSPACE_ROOT,
        )
        return cls(
            runtime_root=runtime_root,
            bind_host=effective_env.get("TOKIMON_RA_HOST", DEFAULT_HOST),
            broker_port=environment_int(
                effective_env,
                "TOKIMON_RA_PORT",
                DEFAULT_PORT,
            ),
            ssh_host=empty_to_none(effective_env.get("TOKIMON_RA_SSH_HOST")),
            default_ttl_days=environment_int(
                effective_env,
                "TOKIMON_RA_DEFAULT_TTL_DAYS",
                DEFAULT_TTL_DAYS,
            ),
            max_read_bytes=environment_int(
                effective_env,
                "TOKIMON_RA_MAX_READ_BYTES",
                DEFAULT_MAX_READ_BYTES,
            ),
            shell_command=effective_env.get("SHELL", "/bin/sh") or "/bin/sh",
            shell_sync_timeout_sec=environment_int(
                effective_env,
                "TOKIMON_RA_SHELL_SYNC_TIMEOUT_SEC",
                DEFAULT_SHELL_SYNC_TIMEOUT_SEC,
            ),
        )


@dataclass(frozen=True)
class CommandSpec:
    argv: list[str]
    stdin_text: str | None = None
    close_stdin: bool = False
    display_command: list[str] | None = None


@dataclass
class JobRuntime:
    process: subprocess.Popen[str]
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    supports_interactive_input: bool
    monitor_thread: threading.Thread | None = None
    stdin_thread: threading.Thread | None = None
    last_output_monotonic: float = 0.0
    last_output_at: str | None = None
    last_output_state_write_monotonic: float = 0.0
    output_state_dirty: bool = False
    input_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    capture_errors: dict[str, str] = field(default_factory=dict)
    capture_abandoned: threading.Event = field(default_factory=threading.Event, repr=False)


def empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def environment_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RemoteAgentBrokerError(f"{name} must be an integer") from exc


def require_port(value: object, *, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise RemoteAgentBrokerError(f"{name} must be an integer from 1 through 65535")
    return value


def encode_sse(payload: Mapping[str, Any]) -> bytes:
    text = json_wire_text(payload)
    return f"event: message\ndata: {text}\n\n".encode("utf-8")


class StructuredLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return format_log_record(
            record, timestamp=datetime.now(timezone.utc).isoformat(),
            service=SERVER_NAME, runtime=getattr(record, "runtime", "http-mcp"),
            fields=('request_id', 'method', 'path', 'status', 'client', 'job_id'),
            format_exception=self.formatException,
        )


def configure_runtime_logger(*, stream: Any | None = None) -> logging.Logger:
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
    job_id: Any = None,
    runtime: str | None = None,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "correlation_id": correlation_id
            or (str(request_id) if request_id is not None else "server"),
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "client": client,
            "job_id": job_id,
            "runtime": runtime or "http-mcp",
        },
    )


def option_string(options: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = options.get(name)
        if value is not None:
            if not isinstance(value, str):
                raise RemoteAgentBrokerError(
                    f"runner_options.{name} must be a string"
                )
            text = value.strip()
            if text:
                return text
    return None


def option_bool(options: Mapping[str, Any], name: str, default: bool = False) -> bool:
    value = options.get(name)
    if value is None:
        return default
    if type(value) is bool:
        return value
    raise RemoteAgentBrokerError(f"runner_options.{name} must be a boolean")


def option_strings(options: Mapping[str, Any], name: str) -> list[str]:
    value = options.get(name)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise RemoteAgentBrokerError(f"runner_options.{name} must be a string or list")


class RunnerAdapter:
    name = ""
    command = ""
    env_var = ""
    supports_interactive_input = False

    def __init__(self, *, env: Mapping[str, str]) -> None:
        self.env = env

    def executable(self) -> str | None:
        return resolve_executable(self.command, self.env_var, self.env)

    def version(self) -> str:
        executable = self.executable()
        if not executable:
            return "unavailable"
        try:
            process = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"unavailable: {exc}"
        output = (process.stdout or process.stderr).strip()
        return output.splitlines()[0] if output else f"exit:{process.returncode}"

    def info(self) -> dict[str, object]:
        executable = self.executable()
        return {
            "agent": self.name,
            "command": self.command,
            "env_var": self.env_var,
            "available": executable is not None,
            "path": executable,
            "version": self.version() if executable else "unavailable",
            "supports_interactive_input": self.supports_interactive_input,
        }

    def build(
        self,
        *,
        cwd: Path,
        prompt: str,
        runner_options: Mapping[str, Any],
    ) -> CommandSpec:
        raise NotImplementedError


class ShellRunner(RunnerAdapter):
    name = SHELL_RUNNER_NAME
    command = "sh"
    env_var = "SHELL"
    supports_interactive_input = True

    def __init__(self, *, env: Mapping[str, str], shell_command: str) -> None:
        super().__init__(env=env)
        self.shell_command = shell_command

    def executable(self) -> str | None:
        resolved = shutil.which(self.shell_command, path=self.env.get("PATH"))
        if resolved:
            return resolved
        path = Path(self.shell_command).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        return shutil.which("sh", path=self.env.get("PATH"))

    def build_shell(self, command: str | list[str]) -> CommandSpec:
        if isinstance(command, list):
            if not command or not all(isinstance(item, str) for item in command):
                raise RemoteAgentBrokerError("command must be a non-empty string array")
            return CommandSpec(argv=list(command))
        if not isinstance(command, str):
            raise RemoteAgentBrokerError("command must be a string or string array")
        text = command.strip()
        if not text:
            raise RemoteAgentBrokerError("command must not be empty")
        executable = self.executable()
        if not executable:
            raise RemoteAgentBrokerError(f"Shell executable not found: {self.shell_command}")
        return CommandSpec(argv=[executable, "-lc", text])


class CopilotRunner(RunnerAdapter):
    name = COPILOT_RUNNER_NAME
    command = "copilot"
    env_var = "COPILOT_CLI"

    def build(
        self,
        *,
        cwd: Path,
        prompt: str,
        runner_options: Mapping[str, Any],
    ) -> CommandSpec:
        executable = self.executable()
        if not executable:
            raise RemoteAgentBrokerError("Copilot CLI not found")
        cmd = [executable, "--no-color", "--no-remote"]
        if option_bool(runner_options, "allow_all", True):
            cmd.append("--allow-all")
        if option_bool(runner_options, "no_ask_user", True):
            cmd.append("--no-ask-user")
        if option_bool(runner_options, "no_custom_instructions", True):
            cmd.append("--no-custom-instructions")
        if option_bool(runner_options, "disable_builtin_mcps", True):
            cmd.append("--disable-builtin-mcps")
        model = option_string(runner_options, "model")
        if model:
            cmd.extend(["--model", model])
        effort = option_string(runner_options, "effort", "reasoning_effort", "effortLevel")
        if effort:
            cmd.extend(["--effort", effort])
        mode = option_string(runner_options, "mode")
        if mode:
            cmd.extend(["--mode", mode])
        output_format = option_string(runner_options, "output_format", "outputFormat")
        if output_format:
            cmd.extend(["--output-format", output_format])
        name = option_string(runner_options, "name")
        if name:
            cmd.extend(["--name", name])
        for directory in option_strings(runner_options, "add_dir"):
            cmd.extend(["--add-dir", directory])
        cmd.extend(["--prompt", prompt])
        display = [*cmd[:-1], "<prompt>"]
        return CommandSpec(argv=cmd, close_stdin=True, display_command=display)


class ClaudeRunner(RunnerAdapter):
    name = CLAUDE_RUNNER_NAME
    command = "claude"
    env_var = "CLAUDE_CODE_CLI"

    def build(
        self,
        *,
        cwd: Path,
        prompt: str,
        runner_options: Mapping[str, Any],
    ) -> CommandSpec:
        executable = self.executable()
        if not executable:
            raise RemoteAgentBrokerError("Claude Code CLI not found")
        cmd = [executable, "--print", "--input-format", "text"]
        output_format = option_string(runner_options, "output_format", "outputFormat") or "text"
        cmd.extend(["--output-format", output_format])
        model = option_string(runner_options, "model")
        if model:
            cmd.extend(["--model", model])
        effort = option_string(runner_options, "effort", "reasoning_effort")
        if effort:
            cmd.extend(["--effort", effort])
        permission_mode = option_string(runner_options, "permission_mode", "permissionMode")
        if permission_mode is None and option_bool(runner_options, "trusted", True):
            permission_mode = "bypassPermissions"
        if permission_mode:
            cmd.extend(["--permission-mode", permission_mode])
        if option_bool(runner_options, "dangerously_skip_permissions", False):
            cmd.append("--dangerously-skip-permissions")
        if option_bool(runner_options, "continue", False):
            cmd.append("--continue")
        resume = option_string(runner_options, "resume")
        if resume:
            cmd.extend(["--resume", resume])
        session_id = option_string(runner_options, "session_id")
        if session_id:
            cmd.extend(["--session-id", session_id])
        for directory in option_strings(runner_options, "add_dir"):
            cmd.extend(["--add-dir", directory])
        return CommandSpec(argv=cmd, stdin_text=prompt, close_stdin=True)


class RemoteAgentBroker:
    def __init__(
        self,
        *,
        config: BrokerConfig | None = None,
        env: Mapping[str, str] | None = None,
        job_store: JobStateStore | None = None,
        artifact_manager: JobArtifactManager | None = None,
        cleanup_manager: JobCleanupManager | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or BrokerConfig.from_env(env=env)
        self.env = dict(os.environ if env is None else env)
        runners = (
            ShellRunner(env=self.env, shell_command=self.config.shell_command),
            CopilotRunner(env=self.env),
            ClaudeRunner(env=self.env),
        )
        self.runners: dict[str, RunnerAdapter] = {runner.name: runner for runner in runners}
        self._lock = threading.RLock()
        self._runtimes: dict[str, JobRuntime] = {}
        self._monitor_clock = monotonic_clock or time.monotonic
        self.job_store = job_store or JobStateStore(
            jobs_root=self.config.runtime_root / "jobs",
            default_ttl_days=self.config.default_ttl_days,
            default_capture_limit_bytes=self.config.max_read_bytes,
        )
        self.artifact_manager = artifact_manager or JobArtifactManager(
            state_store=self.job_store,
            max_read_bytes=self.config.max_read_bytes,
            capture_diff=lambda cwd, base_ref: capture_git_diff(cwd, base_ref=base_ref),
        )
        self.cleanup_manager = cleanup_manager or JobCleanupManager(
            runtime_root=self.config.runtime_root,
            default_ttl_days=self.config.default_ttl_days,
            state_store=self.job_store,
            artifact_manager=self.artifact_manager,
            runtime_lookup=self._runtime,
        )

    def health(self) -> dict[str, object]:
        agents = self.list_agents()["agents"]
        required = {SHELL_RUNNER_NAME}
        available = {
            str(agent["agent"])
            for agent in agents
            if isinstance(agent, Mapping) and bool(agent.get("available"))
        }
        agent_settings = self._agent_settings_health()
        return {
            "ok": required.issubset(available) and bool(agent_settings["ok"]),
            "server": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
                "protocol_version": SUPPORTED_PROTOCOL_VERSION,
            },
            "broker_url": f"http://{self.config.bind_host}:{self.config.broker_port}/mcp",
            "runtime_root": str(self.config.runtime_root),
            "jobs_root": str(self.jobs_root),
            "default_ttl_days": self.config.default_ttl_days,
            "max_read_bytes": self.config.max_read_bytes,
            "ssh_host": self.config.ssh_host,
            "ssh_tunnel_command": self.ssh_tunnel_command(),
            "agents": agents,
            "agent_settings": agent_settings,
        }

    def _agent_settings_health(self) -> dict[str, object]:
        return {"ok": True, "model_settings_required": False}

    def list_agents(self) -> dict[str, object]:
        return {
            "agents": [self.runners[name].info() for name in RUNNER_NAMES]
        }

    def start_agent(
        self,
        *,
        agent: str,
        cwd: str,
        prompt: str,
        runner_options: Mapping[str, Any] | None = None,
        timeout_sec: int | None = None,
        idle_timeout_sec: int | None = None,
        max_wall_time_sec: int | None = None,
        ttl_days: int | None = None,
        capture_limit_bytes: int | None = None,
        request_context: RequestContext | None = None,
    ) -> dict[str, object]:
        if not isinstance(agent, str):
            raise RemoteAgentBrokerError("agent must be a string")
        if not isinstance(cwd, str):
            raise RemoteAgentBrokerError("cwd must be a string")
        if not isinstance(prompt, str):
            raise RemoteAgentBrokerError("prompt must be a string")
        for name, value in (
            ("timeout_sec", timeout_sec),
            ("idle_timeout_sec", idle_timeout_sec),
            ("max_wall_time_sec", max_wall_time_sec),
            ("ttl_days", ttl_days),
            ("capture_limit_bytes", capture_limit_bytes),
        ):
            if value is not None:
                require_positive_int(value, name=name)
        normalized_agent = agent.strip().lower()
        if normalized_agent not in AGENT_RUNNER_NAMES:
            raise RemoteAgentBrokerError("agent must be one of copilot or claude")
        if not prompt.strip():
            raise RemoteAgentBrokerError("prompt is required")
        try:
            options = resolve_agent_options(
                normalized_agent,
                normalize_options(runner_options),
                workspace_root=self.config.workspace_root,
            )
        except TokimonSettingsError as exc:
            raise RemoteAgentBrokerError(str(exc)) from exc
        execution_overrides: dict[str, Any] = {}
        if timeout_sec is not None:
            execution_overrides["idle_timeout_sec"] = timeout_sec
        if idle_timeout_sec is not None:
            execution_overrides["idle_timeout_sec"] = idle_timeout_sec
        if max_wall_time_sec is not None:
            execution_overrides["max_wall_time_sec"] = max_wall_time_sec
        try:
            execution = resolve_agent_execution_options(
                execution_overrides,
                workspace_root=self.config.workspace_root,
            )
        except TokimonSettingsError as exc:
            raise RemoteAgentBrokerError(str(exc)) from exc
        cwd_path = resolve_cwd(cwd)
        spec = self.runners[normalized_agent].build(
            cwd=cwd_path,
            prompt=prompt,
            runner_options=options,
        )
        job_id, job_dir = self.artifact_manager.allocate_job()
        context = request_context or RequestContext.direct_job(job_id)
        job = self.job_store.initial_job(
            job_id=job_id,
            kind="agent",
            agent=normalized_agent,
            cwd=cwd_path,
            command=spec.display_command or spec.argv,
            runner_options=options,
            prompt=prompt,
            ttl_days=ttl_days,
            capture_limit_bytes=capture_limit_bytes,
            timeout_sec=timeout_sec,
            idle_timeout_sec=int(execution["idle_timeout_sec"]),
            max_wall_time_sec=(
                int(execution["max_wall_time_sec"])
                if execution["max_wall_time_sec"] is not None
                else None
            ),
            artifacts=self.artifact_manager.artifact_paths(job_id),
            request_context=context,
        )
        self.job_store.write(job_dir, job)
        self.artifact_manager.append_transcript(job_id, f"# Remote Agent Job {job_id}\n\n")
        self.artifact_manager.append_transcript(job_id, f"- agent: `{normalized_agent}`\n")
        self.artifact_manager.append_transcript(job_id, f"- cwd: `{cwd_path}`\n\n")
        self.artifact_manager.append_transcript(job_id, "## Prompt\n\n")
        self.artifact_manager.append_transcript(
            job_id,
            redact_text(prompt).strip() + "\n\n",
        )
        self._log_job_event("job_accepted", job_id=job_id, context=context)
        self._launch_job(
            job_id,
            job_dir,
            spec,
            cwd_path,
            timeout_sec=(
                int(execution["max_wall_time_sec"])
                if execution["max_wall_time_sec"] is not None
                else None
            ),
            idle_timeout_sec=int(execution["idle_timeout_sec"]),
            poll_interval_sec=float(execution["poll_interval_sec"]),
            supports_interactive_input=(
                self.runners[normalized_agent].supports_interactive_input
            ),
        )
        return self.job_store.public_job(self.job_store.read(job_id))

    def start_shell(
        self,
        *,
        command: str | list[str],
        cwd: str | None = None,
        async_job: bool = False,
        timeout_sec: int | None = None,
        env: Mapping[str, Any] | None = None,
        capture_limit_bytes: int | None = None,
        request_context: RequestContext | None = None,
    ) -> dict[str, object]:
        if type(async_job) is not bool:
            raise RemoteAgentBrokerError("async_job must be a boolean")
        if cwd is not None and not isinstance(cwd, str):
            raise RemoteAgentBrokerError("cwd must be a string")
        if env is not None and not isinstance(env, Mapping):
            raise RemoteAgentBrokerError("env must be an object")
        for name, value in (
            ("timeout_sec", timeout_sec),
            ("capture_limit_bytes", capture_limit_bytes),
        ):
            if value is not None:
                require_positive_int(value, name=name)
        cwd_path = resolve_cwd(cwd or str(self.config.workspace_root))
        process_env = self._process_env(env)
        effective_timeout_sec = (
            timeout_sec
            if timeout_sec is not None
            else None if async_job else self.config.shell_sync_timeout_sec
        )
        shell_runner = self.runners[SHELL_RUNNER_NAME]
        assert isinstance(shell_runner, ShellRunner)
        spec = shell_runner.build_shell(command)
        if not async_job:
            spec = CommandSpec(
                argv=spec.argv,
                stdin_text=spec.stdin_text,
                close_stdin=True,
                display_command=spec.display_command,
            )
        job_id, job_dir = self.artifact_manager.allocate_job()
        context = request_context or RequestContext.direct_job(job_id)
        env_keys = sorted(str(key) for key in (env or {}).keys())
        job = self.job_store.initial_job(
            job_id=job_id,
            kind="shell",
            agent=SHELL_RUNNER_NAME,
            cwd=cwd_path,
            command=spec.display_command or spec.argv,
            runner_options={"env_keys": env_keys} if env_keys else {},
            prompt=None,
            ttl_days=None,
            capture_limit_bytes=capture_limit_bytes,
            timeout_sec=effective_timeout_sec,
            idle_timeout_sec=None,
            max_wall_time_sec=None,
            artifacts=self.artifact_manager.artifact_paths(job_id),
            request_context=context,
        )
        self.job_store.write(job_dir, job)
        self.artifact_manager.append_transcript(job_id, f"# Remote Shell Job {job_id}\n\n")
        self.artifact_manager.append_transcript(job_id, f"- cwd: `{cwd_path}`\n")
        self.artifact_manager.append_transcript(
            job_id,
            f"- command: `{redact_text(' '.join(spec.argv))}`\n\n",
        )
        self._log_job_event("job_accepted", job_id=job_id, context=context)
        self._launch_job(
            job_id,
            job_dir,
            spec,
            cwd_path,
            timeout_sec=effective_timeout_sec,
            idle_timeout_sec=None,
            poll_interval_sec=0.1,
            process_env=process_env,
            supports_interactive_input=shell_runner.supports_interactive_input,
        )
        if async_job:
            return self.job_store.public_job(self.job_store.read(job_id))
        wait_timeout = effective_timeout_sec or self.config.shell_sync_timeout_sec
        runtime = self._runtime(job_id)
        if runtime and runtime.monitor_thread:
            runtime.monitor_thread.join(wait_timeout + 5)
        return self.status(job_id)

    def status(self, job_id: str) -> dict[str, object]:
        runtime = self._runtime(job_id)
        if runtime is not None:
            if runtime.process.poll() is None:
                def mark_running(job: dict[str, Any]) -> None:
                    if job.get("status") not in ACTIVE_STATUSES:
                        return
                    if (
                        job.get("status") == "running"
                        and job.get("pid") == runtime.process.pid
                    ):
                        return
                    job["status"] = "running"
                    job["pid"] = runtime.process.pid
                    job["last_error"] = None
                    job["updated_at"] = utc_now()

                job, _ = self.job_store.update(job_id, mark_running)
                return self.job_store.public_job(job)
            elif runtime.monitor_thread is not None and runtime.monitor_thread.is_alive():
                return self.job_store.public_job(self.job_store.read(job_id))
            else:
                self._remove_runtime(job_id, runtime)

        def mark_lost(job: dict[str, Any]) -> None:
            if job.get("status") not in ACTIVE_STATUSES:
                return
            job["status"] = "lost"
            job["last_error"] = "Broker process is no longer managing this job"
            job["updated_at"] = utc_now()

        job, _ = self.job_store.update(job_id, mark_lost)
        return self.job_store.public_job(job)

    def read(
        self,
        *,
        job_id: str,
        stream: str = "stdout",
        since_offset: int = 0,
        max_bytes: int | None = None,
    ) -> dict[str, object]:
        return self.artifact_manager.read(
            job_id=job_id,
            stream=stream,
            since_offset=since_offset,
            max_bytes=max_bytes,
        )

    def send_input(self, *, job_id: str, text: str) -> dict[str, object]:
        runtime = self._runtime(job_id)
        if runtime is None:
            raise RemoteAgentBrokerError("Job is not running or no longer accepts input")
        with runtime.input_lock:
            job = self.job_store.read(job_id)
            if job.get("status") not in ACTIVE_STATUSES or runtime.process.poll() is not None:
                raise RemoteAgentBrokerError("Job is not running or no longer accepts input")
            if not runtime.supports_interactive_input:
                raise RemoteAgentBrokerError(
                    f"Runner {job.get('agent')} does not support interactive input"
                )
            if runtime.process.stdin is None or runtime.process.stdin.closed:
                raise RemoteAgentBrokerError("Job stdin is not available")
            try:
                runtime.process.stdin.write(text)
                runtime.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise RemoteAgentBrokerError(f"Failed to send input: {exc}") from exc
            self.artifact_manager.append_transcript(
                job_id,
                "\n## Input\n\n" + redact_text(text) + "\n",
            )
        return self.status(job_id)

    def cancel(self, *, job_id: str, signal_name: str | None = None) -> dict[str, object]:
        runtime = self._runtime(job_id)
        if runtime is None:
            return self.status(job_id)
        cancellation_message = (
            f"Cancellation requested{f' with {signal_name}' if signal_name else ''}"
        )
        with runtime.input_lock:
            if runtime.process.poll() is not None:
                monitor_thread = runtime.monitor_thread
            else:
                def request_cancellation(job: dict[str, Any]) -> None:
                    if job.get("status") not in ACTIVE_STATUSES:
                        return
                    job["status"] = "cancelled"
                    job["updated_at"] = utc_now()
                    job["last_error"] = cancellation_message

                job, _ = self.job_store.update(job_id, request_cancellation)
                if job.get("status") != "cancelled":
                    return self.job_store.public_job(job)
                force = cancel_signal_is_force(signal_name)
                signal_error = self._terminate_process(runtime.process, force=force)
                if signal_error is not None:
                    return self.job_store.public_job(
                        self._record_cancellation_signal_result(
                            job_id,
                            cancellation_message=cancellation_message,
                            signal_error=signal_error,
                        )
                    )
                if not force:
                    deadline = time.monotonic() + DEFAULT_CANCEL_GRACE_SEC
                    while runtime.process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if runtime.process.poll() is None:
                        signal_error = self._terminate_process(runtime.process, force=True)
                        if signal_error is not None:
                            return self.job_store.public_job(
                                self._record_cancellation_signal_result(
                                    job_id,
                                    cancellation_message=cancellation_message,
                                    signal_error=signal_error,
                                )
                            )
                self._record_cancellation_signal_result(
                    job_id,
                    cancellation_message=cancellation_message,
                    signal_error=None,
                )
                monitor_thread = runtime.monitor_thread
        if monitor_thread:
            monitor_thread.join(6)
        return self.status(job_id)

    def diff(self, *, job_id: str, base_ref: str | None = None) -> dict[str, object]:
        return self.artifact_manager.diff(job_id=job_id, base_ref=base_ref)

    def artifacts(self, *, job_id: str) -> dict[str, object]:
        return self.artifact_manager.inventory(job_id=job_id)

    def cleanup(self, *, ttl_days: int | None = None) -> dict[str, object]:
        return self.cleanup_manager.cleanup(ttl_days=ttl_days)

    @property
    def jobs_root(self) -> Path:
        return self.job_store.jobs_root

    def ssh_tunnel_command(self) -> str | None:
        if not self.config.ssh_host:
            return None
        return (
            f"ssh -N -L {self.config.broker_port}:127.0.0.1:{self.config.broker_port} "
            f"{self.config.ssh_host}"
        )

    def _launch_job(
        self,
        job_id: str,
        job_dir: Path,
        spec: CommandSpec,
        cwd: Path,
        *,
        timeout_sec: int | None,
        idle_timeout_sec: int | None,
        poll_interval_sec: float,
        process_env: Mapping[str, str] | None = None,
        supports_interactive_input: bool,
    ) -> None:
        env = dict(self.env if process_env is None else process_env)
        try:
            process = subprocess.Popen(
                spec.argv,
                cwd=str(cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            def mark_start_failed(job: dict[str, Any]) -> None:
                now = utc_now()
                job["status"] = "failed"
                job["started_at"] = now
                job["updated_at"] = now
                job["completed_at"] = now
                job["last_error"] = f"Failed to start process: {exc}"

            with self.artifact_manager.lock(job_id):
                job, _ = self.job_store.update(job_id, mark_start_failed)
                self.artifact_manager.append_transcript(
                    job_id,
                    "\n## Completion\n\n- status: `failed`\n"
                    f"- error: {redact_text(str(exc))}\n",
                )
                self.artifact_manager.write_final(
                    job_id,
                    job,
                    stdout_tail="",
                    stderr_tail=str(exc),
                )
            self._log_job_event("job_completed", job_id=job_id, job=job)
            return

        stdout_thread = threading.Thread(
            target=self._pump_stream,
            args=(process.stdout, job_dir / "stdout.log", job_id),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._pump_stream,
            args=(process.stderr, job_dir / "stderr.log", job_id),
            daemon=True,
        )
        launched_monotonic = self._monitor_clock()
        runtime = JobRuntime(
            process=process,
            stdout_thread=stdout_thread,
            stderr_thread=stderr_thread,
            supports_interactive_input=supports_interactive_input,
            last_output_monotonic=launched_monotonic,
            last_output_state_write_monotonic=launched_monotonic,
        )
        with self._lock:
            self._runtimes[job_id] = runtime

        def mark_running(job: dict[str, Any]) -> None:
            now = utc_now()
            job["status"] = "running"
            job["pid"] = process.pid
            job["started_at"] = now
            job["updated_at"] = now

        self.job_store.update(job_id, mark_running)

        stdout_thread.start()
        stderr_thread.start()

        stdin_delivery_ready = threading.Event()
        if spec.stdin_text is not None or spec.close_stdin:
            def deliver_stdin() -> None:
                stdin_delivery_ready.wait()
                try:
                    assert process.stdin is not None
                    if spec.stdin_text is not None:
                        process.stdin.write(spec.stdin_text)
                        process.stdin.flush()
                except (OSError, UnicodeError, ValueError) as exc:
                    self._record_capture_error(job_id, stream_name="stdin", error=exc)
                finally:
                    if spec.close_stdin and process.stdin is not None:
                        try:
                            process.stdin.close()
                        except (OSError, UnicodeError, ValueError) as exc:
                            self._record_capture_error(job_id, stream_name="stdin", error=exc)

            runtime.stdin_thread = threading.Thread(target=deliver_stdin, daemon=True)
            runtime.stdin_thread.start()

        monitor_thread = threading.Thread(
            target=self._monitor_job,
            args=(job_id, timeout_sec, idle_timeout_sec, poll_interval_sec),
            daemon=True,
        )
        runtime.monitor_thread = monitor_thread
        monitor_thread.start()
        stdin_delivery_ready.set()

    def _monitor_job(
        self,
        job_id: str,
        timeout_sec: int | None,
        idle_timeout_sec: int | None,
        poll_interval_sec: float,
    ) -> None:
        runtime = self._runtime(job_id)
        if runtime is None:
            return
        process = runtime.process
        started = self._monitor_clock()
        timed_out = False
        idle_timed_out = False
        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            self._flush_output_state_if_due(job_id)
            if timeout_sec is not None and self._monitor_clock() - started > timeout_sec:
                timed_out = True
                self._terminate_process(process, force=False)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._terminate_process(process, force=True)
                break
            with self.job_store.lock(job_id):
                last_output_monotonic = runtime.last_output_monotonic
            if (
                idle_timeout_sec is not None
                and self._monitor_clock() - last_output_monotonic > idle_timeout_sec
            ):
                timed_out = True
                idle_timed_out = True
                self._terminate_process(process, force=False)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._terminate_process(process, force=True)
                break
            time.sleep(max(0.05, poll_interval_sec))
        try:
            returncode = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._terminate_process(process, force=True)
            returncode = process.wait(timeout=2)
        if runtime.stdin_thread is None and process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        self._drain_stream_capture_after_exit(job_id, runtime)

        with self.job_store.lock(job_id):
            capture_errors = dict(runtime.capture_errors)
        capture_error = "; ".join(
            f"{stream} capture failed: {error}"
            for stream, error in sorted(capture_errors.items())
        ) or None

        job_dir = self.job_store.job_dir(job_id)
        job_snapshot = self.job_store.read(job_id)
        completed_at = utc_now()
        completed_dt = parse_utc(completed_at)
        diff_result = self.artifact_manager.capture_job_diff(
            job_id=job_id,
            cwd=Path(str(job_snapshot.get("cwd") or "")),
        )

        def finalize(job: dict[str, Any]) -> None:
            previous_status = str(job.get("status") or "")
            if previous_status in TERMINAL_STATUSES:
                status = previous_status
                last_error = job.get("last_error")
            elif timed_out:
                status = "failed"
                if idle_timed_out:
                    last_error = f"Job idle timed out after {idle_timeout_sec}s without output"
                else:
                    last_error = f"Job timed out after {timeout_sec}s"
            elif capture_error is not None:
                status = "failed"
                last_error = capture_error
            elif returncode == 0:
                status = "completed"
                last_error = None
            else:
                status = "failed"
                last_error = f"Process exited with {returncode}"
            if capture_error is not None and capture_error not in str(last_error or ""):
                last_error = f"{last_error}; {capture_error}" if last_error else capture_error
            started_at = parse_utc(str(job.get("started_at") or ""))
            duration = None
            if started_at and completed_dt:
                duration = round((completed_dt - started_at).total_seconds(), 3)
            job.update(
                {
                    "status": status,
                    "updated_at": completed_at,
                    "completed_at": completed_at,
                    "exit_code": returncode,
                    "duration_sec": duration,
                    "last_error": last_error,
                    "diff": {key: value for key, value in diff_result.items() if key != "diff"},
                }
            )
            if runtime.last_output_at is not None:
                job["last_output_at"] = runtime.last_output_at
            runtime.output_state_dirty = False

        job, _ = self.job_store.update(job_id, finalize)
        status = str(job.get("status") or "failed")
        last_error = job.get("last_error")
        stdout_tail = read_file_tail(job_dir / "stdout.log")
        stderr_tail = read_file_tail(job_dir / "stderr.log")
        output_transcript = format_transcript_output(
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
        if output_transcript:
            self.artifact_manager.append_transcript(job_id, output_transcript)
        self.artifact_manager.append_transcript(
            job_id,
            f"\n## Completion\n\n- status: `{status}`\n- exit_code: `{returncode}`\n",
        )
        if last_error:
            self.artifact_manager.append_transcript(
                job_id,
                f"- error: {redact_text(str(last_error))}\n",
            )
        self.artifact_manager.write_final(
            job_id,
            job,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
        self._log_job_event("job_completed", job_id=job_id, job=job)
        self._remove_runtime(job_id, runtime)

    def _pump_stream(self, stream: Any, path: Path, job_id: str) -> None:
        if stream is None:
            return
        runtime = self._runtime(job_id)
        if runtime is None:
            return
        stream_name = path.stem
        try:
            with path.open("a", encoding="utf-8") as handle:
                binary_stream = getattr(stream, "buffer", None)
                decoder = None
                if binary_stream is not None:
                    decoder = codecs.getincrementaldecoder(stream.encoding or "utf-8")(
                        errors="replace"
                    )
                pending = ""
                suppress_overlong_line = False
                while True:
                    if binary_stream is None:
                        chunk = stream.read(STREAM_READ_CHARS)
                        reached_eof = chunk == ""
                    else:
                        raw_chunk = binary_stream.read1(STREAM_READ_CHARS)
                        reached_eof = raw_chunk == b""
                        assert decoder is not None
                        chunk = decoder.decode(raw_chunk, final=reached_eof)
                    if runtime.capture_abandoned.is_set():
                        return
                    if chunk:
                        self._record_output(job_id)
                    if reached_eof and not chunk:
                        break
                    pending += chunk
                    persisted: list[str] = []
                    while pending:
                        if suppress_overlong_line:
                            newline = pending.find("\n")
                            if newline < 0:
                                pending = ""
                                break
                            persisted.append("\n")
                            pending = pending[newline + 1 :]
                            suppress_overlong_line = False
                            continue
                        newline = pending.find("\n")
                        if 0 <= newline < MAX_STREAM_REDACTION_BUFFER_CHARS:
                            line = pending[: newline + 1]
                            persisted.append(redact_text(line))
                            pending = pending[newline + 1 :]
                            continue
                        if len(pending) < MAX_STREAM_REDACTION_BUFFER_CHARS:
                            break
                        persisted.append(
                            redact_text(pending[:MAX_STREAM_REDACTION_BUFFER_CHARS])
                        )
                        persisted.append("\n[output line truncated before persistence]\n")
                        pending = pending[MAX_STREAM_REDACTION_BUFFER_CHARS:]
                        suppress_overlong_line = True
                    if persisted:
                        if not self._persist_stream_text(
                            job_id,
                            runtime,
                            handle,
                            "".join(persisted),
                        ):
                            return
                    if reached_eof:
                        break
                if pending:
                    if not self._persist_stream_text(
                        job_id,
                        runtime,
                        handle,
                        redact_text(pending),
                    ):
                        return
        except Exception as exc:
            self._record_capture_error(job_id, stream_name=stream_name, error=exc)
        finally:
            try:
                stream.close()
            except Exception as exc:
                self._record_capture_error(job_id, stream_name=stream_name, error=exc)

    def _drain_stream_capture_after_exit(
        self,
        job_id: str,
        runtime: JobRuntime,
    ) -> None:
        alive = self._join_stream_readers(runtime, POST_EXIT_STREAM_DRAIN_SEC)
        if not alive:
            return

        cleanup_errors: list[str] = []
        signal_error = self._signal_process_group_after_exit(runtime.process, force=False)
        if signal_error is not None:
            cleanup_errors.append(f"SIGTERM: {signal_error}")
        alive = self._join_stream_readers(runtime, POST_EXIT_PROCESS_GROUP_TERM_SEC)
        if not alive:
            return

        signal_error = self._signal_process_group_after_exit(runtime.process, force=True)
        if signal_error is not None:
            cleanup_errors.append(f"SIGKILL: {signal_error}")
        alive = self._join_stream_readers(runtime, POST_EXIT_PROCESS_GROUP_KILL_SEC)
        if not alive:
            return

        runtime.capture_abandoned.set()
        with self.artifact_manager.lock(job_id):
            pass
        cleanup_detail = (
            f"; process-group cleanup errors: {'; '.join(cleanup_errors)}"
            if cleanup_errors
            else ""
        )
        for stream_name in alive:
            self._record_capture_error(
                job_id,
                stream_name=stream_name,
                error=RemoteAgentBrokerError(
                    "I/O worker did not finish after parent exit and bounded process-group cleanup"
                    f"{cleanup_detail}"
                ),
            )

    def _persist_stream_text(
        self,
        job_id: str,
        runtime: JobRuntime,
        handle: TextIO,
        text: str,
    ) -> bool:
        with self.artifact_manager.lock(job_id):
            if runtime.capture_abandoned.is_set():
                return False
            handle.write(text)
            handle.flush()
        return True

    def _join_stream_readers(
        self,
        runtime: JobRuntime,
        timeout_sec: float,
    ) -> list[str]:
        readers = [
            ("stdout", runtime.stdout_thread),
            ("stderr", runtime.stderr_thread),
        ]
        if runtime.stdin_thread is not None:
            readers.append(("stdin", runtime.stdin_thread))
        deadline = time.monotonic() + max(0.0, timeout_sec)
        for _stream_name, reader in readers:
            reader.join(max(0.0, deadline - time.monotonic()))
        return [stream_name for stream_name, reader in readers if reader.is_alive()]

    def _signal_process_group_after_exit(
        self,
        process: subprocess.Popen[str],
        *,
        force: bool,
    ) -> str | None:
        if not hasattr(os, "killpg"):
            return "process-group signalling is unavailable"
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            return None
        except OSError as exc:
            return f"{exc.__class__.__name__}: {exc}"
        return None

    def _record_output(self, job_id: str) -> None:
        now_monotonic = self._monitor_clock()
        with self.job_store.lock(job_id):
            runtime = self._runtime(job_id)
            if runtime is None:
                return
            runtime.last_output_monotonic = now_monotonic
            runtime.last_output_at = utc_now()
            runtime.output_state_dirty = True
        self._flush_output_state_if_due(job_id)

    def _flush_output_state_if_due(self, job_id: str) -> None:
        with self.job_store.lock(job_id):
            runtime = self._runtime(job_id)
            if runtime is None or not runtime.output_state_dirty:
                return
            current_monotonic = self._monitor_clock()
            elapsed = current_monotonic - runtime.last_output_state_write_monotonic
            if elapsed < OUTPUT_STATE_FLUSH_INTERVAL_SEC:
                return
            try:
                job = self.job_store.read_unlocked(job_id)
            except RemoteAgentBrokerError:
                return
            if job.get("status") not in ACTIVE_STATUSES or runtime.last_output_at is None:
                return
            job["last_output_at"] = runtime.last_output_at
            job["updated_at"] = runtime.last_output_at
            self.job_store.write_unlocked(self.job_store.job_dir(job_id), job)
            runtime.last_output_state_write_monotonic = self._monitor_clock()
            runtime.output_state_dirty = False

    def _process_env(self, additions: Mapping[str, Any] | None) -> dict[str, str]:
        env = dict(self.env)
        for key, value in (additions or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise RemoteAgentBrokerError("env names and values must be strings")
            env[key] = value
        return env

    def _record_capture_error(
        self,
        job_id: str,
        *,
        stream_name: str,
        error: Exception,
    ) -> None:
        error_text = redact_text(str(error)) or error.__class__.__name__
        with self.job_store.lock(job_id):
            runtime = self._runtime(job_id)
            if runtime is None:
                return
            previous = runtime.capture_errors.get(stream_name)
            if previous and error_text not in previous:
                runtime.capture_errors[stream_name] = f"{previous}; {error_text}"
            elif not previous:
                runtime.capture_errors[stream_name] = error_text

    def _record_cancellation_signal_result(
        self,
        job_id: str,
        *,
        cancellation_message: str,
        signal_error: str | None,
    ) -> dict[str, Any]:
        def record_result(job: dict[str, Any]) -> None:
            if job.get("status") != "cancelled":
                return
            if signal_error is not None:
                job["cancellation_signal_error"] = signal_error
                job["last_error"] = f"{cancellation_message}; signal delivery failed: {signal_error}"
                job["updated_at"] = utc_now()
            elif "cancellation_signal_error" in job:
                job.pop("cancellation_signal_error", None)
                job["last_error"] = cancellation_message
                job["updated_at"] = utc_now()

        job, _ = self.job_store.update(job_id, record_result)
        return job

    def _terminate_process(
        self,
        process: subprocess.Popen[str],
        *,
        force: bool,
    ) -> str | None:
        if process.poll() is not None:
            return None
        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            if hasattr(os, "killpg"):
                os.killpg(process.pid, sig)
            else:
                process.kill() if force else process.terminate()
        except ProcessLookupError:
            return None
        except OSError as exc:
            return f"{exc.__class__.__name__}: {exc}"
        return None

    def _log_job_event(
        self,
        message: str,
        *,
        job_id: str,
        context: RequestContext | None = None,
        job: Mapping[str, Any] | None = None,
    ) -> None:
        persisted_context = (
            job.get("request_context")
            if isinstance(job, Mapping)
            and isinstance(job.get("request_context"), Mapping)
            else {}
        )
        correlation_id = (
            context.correlation_id
            if context is not None
            else str(persisted_context.get("correlation_id") or job_id)
        )
        request_id = (
            context.request_id
            if context is not None
            else persisted_context.get("request_id")
        )
        log_runtime_event(
            configure_runtime_logger(),
            logging.INFO,
            message,
            correlation_id=correlation_id,
            request_id=request_id,
            status=job.get("status") if isinstance(job, Mapping) else None,
            job_id=job_id,
            runtime="broker",
        )

    def _runtime(self, job_id: str) -> JobRuntime | None:
        with self._lock:
            return self._runtimes.get(job_id)

    def _remove_runtime(self, job_id: str, runtime: JobRuntime) -> None:
        with self._lock:
            if self._runtimes.get(job_id) is runtime:
                self._runtimes.pop(job_id, None)

def normalize_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise RemoteAgentBrokerError("runner_options must be an object")
    return dict(options)


def format_transcript_output(*, stdout_tail: str, stderr_tail: str) -> str:
    sections: list[str] = []
    if stdout_tail:
        sections.append("### stdout (tail)\n\n" + redact_text(stdout_tail).rstrip())
    if stderr_tail:
        sections.append("### stderr (tail)\n\n" + redact_text(stderr_tail).rstrip())
    if not sections:
        return ""
    return "\n## Output\n\n" + "\n\n".join(sections) + "\n"


def cancel_signal_is_force(signal_name: str | None) -> bool:
    if signal_name is None:
        return False
    normalized = signal_name.strip().lower()
    return normalized in {"9", "kill", "sigkill", "signal.sigkill"}


def resolve_cwd(cwd: str) -> Path:
    if not isinstance(cwd, str):
        raise RemoteAgentBrokerError("cwd must be a string")
    text = cwd.strip()
    if not text:
        raise RemoteAgentBrokerError("cwd is required")
    path = Path(text).expanduser()
    if not path.exists():
        raise RemoteAgentBrokerError(f"cwd does not exist: {path}")
    if not path.is_dir():
        raise RemoteAgentBrokerError(f"cwd is not a directory: {path}")
    return path.resolve()


def positive_int_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def port_argument(value: str) -> int:
    try:
        return require_port(int(value), name="port")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer from 1 through 65535") from exc
    except RemoteAgentBrokerError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def capture_git_diff(cwd: Path, *, base_ref: str | None = None) -> dict[str, object]:
    if not cwd.exists() or not cwd.is_dir():
        return {"available": False, "error": f"cwd is unavailable: {cwd}"}
    if not shutil.which("git"):
        return {"available": False, "error": "git command not found"}
    try:
        check = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"available": False, "error": f"git work-tree check timed out after {exc.timeout}s"}
    except OSError as exc:
        return {"available": False, "error": f"git work-tree check failed: {exc}"}
    if check.returncode != 0:
        return {"available": False, "error": "cwd is not inside a git work tree"}

    def run_diff(*args: str) -> dict[str, object]:
        try:
            diff = subprocess.run(
                ["git", "-C", str(cwd), "diff", "--no-ext-diff", "--binary", *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {"available": False, "error": f"git diff timed out after {exc.timeout}s"}
        except OSError as exc:
            return {"available": False, "error": f"git diff failed: {exc}"}
        if diff.returncode != 0:
            return {
                "available": False,
                "error": (diff.stderr or diff.stdout or "").strip()
                or f"git diff exited {diff.returncode}",
            }
        return {"available": True, "diff": diff.stdout or ""}

    if base_ref:
        return run_diff(base_ref)

    try:
        has_head = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"available": False, "error": f"git HEAD check timed out after {exc.timeout}s"}
    except OSError as exc:
        return {"available": False, "error": f"git HEAD check failed: {exc}"}
    if has_head.returncode == 0:
        return run_diff("HEAD")

    staged = run_diff("--cached", "--root")
    if not staged["available"]:
        return staged
    unstaged = run_diff()
    if not unstaged["available"]:
        return unstaged
    return {"available": True, "diff": f"{staged['diff']}{unstaged['diff']}"}


class MCPServer:
    def __init__(self, broker: RemoteAgentBroker | None = None) -> None:
        self.broker = broker or RemoteAgentBroker()
        self.dispatcher = ToolCallDispatcher(self.broker)

    def handle(
        self,
        request: Mapping[str, Any],
        *,
        request_context: RequestContext | None = None,
    ) -> dict[str, Any] | None:
        context = request_context or RequestContext.from_request(
            request_id=request.get("id"), correlation_id=None, transport="stdio-mcp",
        )
        return handle_request(
            request, server_name=SERVER_NAME, server_version=SERVER_VERSION,
            protocol_version=SUPPORTED_PROTOCOL_VERSION, list_tools=mcp_tools,
            call_tool=lambda params: self.dispatcher.dispatch(params, request_context=context),
        )


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_handler(server: MCPServer) -> type[BaseHTTPRequestHandler]:
    class RemoteAgentMCPHandler(BaseHTTPRequestHandler):
        server_version = "TokimonRemoteAgentBroker/0.1"

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(HTTP_READ_TIMEOUT_SEC)

        def _local_request_allowed(self) -> bool:
            hosts = self.headers.get_all("Host", [])
            try:
                authority = urllib.parse.urlsplit("//" + hosts[0]) if len(hosts) == 1 else None
                host = authority.hostname if authority is not None else None
                if authority is not None:
                    authority.port  # Validate malformed ports before trusting the host.
                valid_authority = bool(
                    authority and authority.netloc == hosts[0]
                    and authority.username is None and authority.password is None
                    and not authority.path and not authority.query and not authority.fragment
                )
            except ValueError:
                host, valid_authority = None, False
            if (
                not valid_authority or not is_loopback_host(host or "")
                or not is_loopback_host(self.client_address[0])
                or not is_loopback_host(str(self.server.server_address[0]))
                or self.headers.get_all("Origin", [])
            ):
                self.send_error(403, "Agent MCP requires a local non-browser client")
                return False
            return True

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
                self._send_json(server.broker.health())
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
            )
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
            lengths = self.headers.get_all("Content-Length", [])
            if (
                self.headers.get_all("Transfer-Encoding", []) or len(lengths) != 1
                or not lengths[0].isascii() or not lengths[0].isdecimal()
                or len(lengths[0]) > 10 or int(lengths[0]) <= 0
            ):
                self.send_error(400, "One positive decimal Content-Length is required")
                return
            length = int(lengths[0])
            if length > MAX_HTTP_REQUEST_BYTES:
                self.send_error(413, "JSON-RPC body exceeds 1 MiB")
                return
            content_types = self.headers.get_all("Content-Type", [])
            if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
                self.send_error(415, "Content-Type must be application/json")
                return
            try:
                raw = self.rfile.read(length)
            except OSError:
                self.send_error(400, "Could not read the JSON-RPC body")
                return
            if len(raw) != length:
                self.send_error(400, "Incomplete JSON-RPC body")
                return
            try:
                request = json.loads(raw.decode("utf-8"))
            except (ValueError, RecursionError) as exc:
                session_key = self._session_key(None)
                log_runtime_event(
                    configure_runtime_logger(),
                    logging.WARNING,
                    "invalid_jsonrpc_request",
                    correlation_id=session_key,
                    method="POST",
                    path=parsed_path.path,
                    client=self.client_address[0],
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
            )
            result = server.handle(
                request,
                request_context=RequestContext.from_request(
                    request_id=request.get("id"),
                    correlation_id=session_key,
                    transport="http-mcp",
                ),
            )
            if result is None:
                self.send_response(202)
                self.send_header("mcp-session-id", session_key)
                self.end_headers()
                return
            self._send_sse(result, mcp_session_key=session_key)

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
                    time.sleep(15)
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _send_json(self, payload: Mapping[str, Any]) -> None:
            body = json_wire_text(payload, indent=2).encode("utf-8")
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
            )

    return RemoteAgentMCPHandler


def serve(
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    server: MCPServer | None = None,
) -> int:
    resolved_server = server or MCPServer()
    logger = configure_runtime_logger()
    log_runtime_event(
        logger,
        logging.INFO,
        "server_start",
        correlation_id="server-start",
        method="serve",
        runtime="stdio-mcp",
    )
    for line in input_stream:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except (ValueError, RecursionError) as exc:
            log_runtime_event(
                logger, logging.WARNING, "invalid_jsonrpc_request",
                correlation_id="parse-error", runtime="stdio-mcp",
            )
            result = error_response(None, -32700, str(exc))
        else:
            if not isinstance(request, dict):
                result = error_response(None, -32600, "Request must be an object")
            else:
                log_runtime_event(
                    logger,
                    logging.INFO,
                    "request_received",
                    correlation_id=str(request.get("id") or "notification"),
                    request_id=request.get("id"),
                    method=request.get("method"),
                    runtime="stdio-mcp",
                )
                result = resolved_server.handle(
                    request,
                    request_context=RequestContext.from_request(
                        request_id=request.get("id"),
                        correlation_id=str(request.get("id") or "notification"),
                        transport="stdio-mcp",
                    ),
                )
        if result is not None:
            print(json_wire_text(result), file=output_stream, flush=True)
    return 0


def serve_http(*, config: BrokerConfig | None = None, broker: RemoteAgentBroker | None = None) -> int:
    resolved_config = config or BrokerConfig.from_env()
    if not is_loopback_host(resolved_config.bind_host):
        raise RemoteAgentBrokerError("HTTP MCP requires a loopback bind; use stdio or an SSH tunnel for remote access")
    resolved_broker = broker or RemoteAgentBroker(config=resolved_config)
    server = MCPServer(resolved_broker)
    httpd = ThreadingHTTPServer(
        (resolved_config.bind_host, resolved_config.broker_port),
        make_handler(server),
    )
    log_runtime_event(
        configure_runtime_logger(),
        logging.INFO,
        "server_start",
        correlation_id="server-start",
        method="serve-http",
        path=f"http://{resolved_config.bind_host}:{resolved_config.broker_port}/mcp",
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        httpd.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the remote agent broker MCP server.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Run the stdio MCP broker")

    serve_parser = subparsers.add_parser("serve-http", help="Run the HTTP MCP broker")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=port_argument, default=None)

    health_parser = subparsers.add_parser("health", help="Show broker health")
    health_parser.add_argument("--json", action="store_true")

    cleanup_parser = subparsers.add_parser("cleanup", help="Remove expired completed job artifacts")
    cleanup_parser.add_argument("--ttl-days", type=positive_int_argument, default=None)
    cleanup_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BrokerConfig.from_env()
    if args.command == "serve":
        return serve()
    if args.command == "serve-http":
        if args.host is not None:
            config = replace(config, bind_host=args.host)
        if args.port is not None:
            config = replace(config, broker_port=args.port)
        return serve_http(config=config)
    if args.command == "health":
        payload = RemoteAgentBroker(config=config).health()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"{SERVER_NAME}: {'ok' if payload['ok'] else 'unavailable'}")
            print(f"Broker URL: {payload['broker_url']}")
            print(f"Runtime root: {payload['runtime_root']}")
            for agent in payload["agents"]:
                print(f"{agent['agent']}: {agent['path'] or 'missing'}")
        return 0 if payload["ok"] else 1
    if args.command == "cleanup":
        payload = RemoteAgentBroker(config=config).cleanup(ttl_days=args.ttl_days)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            if payload.get("ok") is False:
                print(payload.get("error", "cleanup failed"))
            else:
                print(f"Removed {payload['removed_count']} expired job artifact directories")
        return 0 if payload.get("ok") is not False else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
