#!/usr/bin/env python3
"""Structured AI debate MCP server and direct CLI."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from tokimon.tools.mcp._protocol import response, error_response, json_text, tool_result, handle_request, serve_stdio
from tokimon.tools._runtime import configure_logger, format_log_record, resolve_executable
from tokimon.runtime import resolve_runtime_path, runtime_root as repository_runtime_root, source_root
from tokimon.settings import (
    TokimonSettingsError,
    load_settings,
    load_settings_section,
    resolve_agent_execution_options,
    resolve_agent_options,
)
from tokimon.tools.capability_readiness import require_ai_debate_readiness_configuration

WORKSPACE_ROOT = source_root()
WORKSPACE_STATE_ROOT = repository_runtime_root(WORKSPACE_ROOT)
DEBATE_STATE_ROOT = WORKSPACE_STATE_ROOT / "tools" / "mcp" / "ai-debate"
DEFAULT_RUNS_ROOT = DEBATE_STATE_ROOT / "runs"

DEFAULT_MAX_ROUNDS = 24
DEFAULT_RETRY_AFTER_SEC = 3600
SUPPORTED_ACCESS = {"local-cli"}
PERSISTED_ACCESS = SUPPORTED_ACCESS | {"mcp.remote-agent-broker"}
SUPPORTED_SIDE_B_PROVIDERS = {"claude", "codex"}
RUN_STATUS_VALUES = {"in_progress", "waiting_retry", "failed", "completed"}
RUN_NEXT_STEP_VALUES = {"agent_a", "agent_b", "final_judgment", "completed"}
RUN_ROLE_NAMES = ("agent_a", "agent_b", "judge")
TURN_SIGNAL_VALUES = {"continue", "concede", "tie"}
TERMINATION_VALUES = {"concession", "mutual_tie", "round_limit"}
WINNER_VALUES = {"agent_a", "agent_b", "tie"}
FAILURE_REASON_VALUES = {
    "tool_failure",
    "provider_failure",
    "rate_limit",
    "quota_exhausted",
    "token_exhausted",
}
SERVER_NAME = "tokimon-ai-debate"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSION = "2024-11-05"
LOGGER_NAME = "tokimon.tools.mcp.ai_debate"

WINNER_LABELS = {
    "agent_a": "Agent A",
    "agent_b": "Agent B",
    "tie": "Tie",
}


class DebateToolError(RuntimeError):
    """Raised when an AI debate operation cannot continue."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class DebateRunFailedError(DebateToolError):
    """Raised when a created run failed but has a persisted run summary."""

    def __init__(
        self,
        message: str,
        *,
        summary: dict[str, Any],
        exit_code: int = 1,
    ) -> None:
        super().__init__(message, exit_code=exit_code)
        self.summary = summary


class ProviderCallError(DebateToolError):
    """Raised when a provider task fails after artifacts can be captured."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        task_id: str,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        cli_exit_code: int | None = None,
        retry_reason: str | None = None,
        retry_after_sec: int | None = None,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message, exit_code=exit_code)
        self.provider = provider
        self.task_id = task_id
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.cli_exit_code = cli_exit_code
        self.retry_reason = retry_reason
        self.retry_after_sec = retry_after_sec

    @property
    def retryable(self) -> bool:
        return bool(self.retry_reason and self.retry_after_sec)


@dataclass(frozen=True)
class RunOptions:
    topic: str
    side_a: str
    side_b: str
    side_b_provider: str | None = None
    access: str | None = None
    max_rounds: int = DEFAULT_MAX_ROUNDS
    agent_a_model: str | None = None
    agent_b_model: str | None = None
    agent_b_effort: str | None = None
    judge_model: str | None = None
    turn_idle_timeout_sec: int | None = None
    turn_timeout_sec: int | None = None
    output_root: Path | None = None
    model: str | None = None
    effort: str | None = None


@dataclass
class RunContext:
    run_dir: Path
    config: dict[str, Any]
    state: dict[str, Any]

    @property
    def debate_log(self) -> Path:
        return self.run_dir / "debate_log.md"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def prompts_dir(self) -> Path:
        return self.run_dir / "prompts"

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"


@dataclass(frozen=True)
class ProviderCallResult:
    payload: dict[str, Any]
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    cli_exit_code: int = 0


class ProviderRunner:
    """Runs installed provider CLIs and extracts JSON payloads."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        self.env = dict(os.environ if env is None else env)
        self._monotonic = monotonic
        self._sleep = sleeper

    def resolve_executable(self, provider: str) -> str | None:
        if provider == "codex":
            return resolve_executable("codex", "CODEX_CLI", self.env)
        if provider == "claude":
            return resolve_executable("claude", "CLAUDE_CODE_CLI", self.env)
        raise DebateToolError(f"Unsupported provider: {provider}")

    def capture_runtime_versions(self) -> dict[str, str]:
        return {
            "python": sys.version.split()[0],
            "codex": self._capture_version("codex"),
            "claude": self._capture_version("claude"),
        }

    def _capture_version(self, provider: str) -> str:
        executable = self.resolve_executable(provider)
        if not executable:
            return "unavailable"
        try:
            process = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"unavailable: {exc}"
        output = (process.stdout or process.stderr).strip()
        return output.splitlines()[0] if output else f"exit:{process.returncode}"

    def run_json(
        self,
        *,
        provider: str,
        model: str,
        effort: str | None = None,
        prompt: str,
        timeout_sec: int,
        workspace_dir: Path,
        task_id: str,
        results_dir: Path,
        expected_keys: Iterable[str],
        poll_interval_sec: float = 1.0,
        max_wall_time_sec: int | None = None,
    ) -> ProviderCallResult:
        validate_provider_timing("timeout_sec", timeout_sec)
        validate_provider_timing("poll_interval_sec", poll_interval_sec)
        if max_wall_time_sec is not None:
            validate_provider_timing("max_wall_time_sec", max_wall_time_sec)
        executable = self.resolve_executable(provider)
        if not executable:
            raise ProviderCallError(
                f"Provider CLI not found for {provider}",
                provider=provider,
                task_id=task_id,
            )

        stdout_path = results_dir / f"{task_id}.stdout"
        stderr_path = results_dir / f"{task_id}.stderr"
        try:
            require_run_write_path(workspace_dir, results_dir, kind="directory")
            results_dir.mkdir(parents=True, exist_ok=True)
            require_run_write_path(workspace_dir, results_dir, kind="directory")
        except (DebateToolError, OSError) as exc:
            raise ProviderCallError(
                f"Could not create provider result directory {results_dir}: {exc}",
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ) from exc
        schema = schema_for_expected_keys(tuple(expected_keys))
        schema_path = results_dir / f"{task_id}.schema.json"
        try:
            for artifact_path in (schema_path, stdout_path, stderr_path):
                require_run_write_path(workspace_dir, artifact_path, kind="file")
            write_text_artifact(
                schema_path,
                json.dumps(schema, indent=2),
                run_dir=workspace_dir,
            )
        except DebateToolError as exc:
            raise ProviderCallError(
                f"Could not write provider schema artifact {schema_path}: {exc}",
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ) from exc
        cmd, stdin_text = self._build_command(
            executable=executable,
            provider=provider,
            model=model,
            effort=effort,
            prompt=prompt,
            workspace_dir=workspace_dir,
            schema=schema,
            schema_path=schema_path,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stream_errors: list[tuple[Path, Exception]] = []
        provider_started_at = self._monotonic()
        last_output = {"monotonic": provider_started_at}
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(workspace_dir),
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise ProviderCallError(
                f"Failed to run {provider}: {exc}",
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ) from exc

        stdout_thread = threading.Thread(
            target=pump_provider_stream,
            args=(
                process.stdout,
                stdout_path,
                stdout_chunks,
                last_output,
                self._monotonic,
                stream_errors,
            ),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=pump_provider_stream,
            args=(
                process.stderr,
                stderr_path,
                stderr_chunks,
                last_output,
                self._monotonic,
                stream_errors,
            ),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        stdin_thread: threading.Thread | None = None
        if stdin_text is not None:
            stdin_thread = threading.Thread(
                target=pump_provider_stdin,
                args=(process.stdin, stdin_text),
                daemon=True,
            )
            stdin_thread.start()

        timeout_message = self._wait_for_provider_process(
            process,
            provider=provider,
            task_id=task_id,
            last_output=last_output,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            max_wall_time_sec=max_wall_time_sec,
            started_at=provider_started_at,
        )
        if timeout_message is not None:
            terminate_provider_process(process, force=False)

        if timeout_message is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                terminate_provider_process(process, force=True)
                process.wait(timeout=5)
            # The session leader may exit on SIGTERM while a descendant that
            # inherited provider pipes ignores it.  Force-clean the original
            # process group even after the parent has a return code.
            terminate_exited_provider_group(process, force=True)
            stdout_thread.join(2)
            stderr_thread.join(2)
            if stdin_thread is not None:
                stdin_thread.join(2)
            raise ProviderCallError(
                timeout_message,
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cli_exit_code=process.returncode,
            )

        process.wait()
        drain_message = self._wait_for_provider_streams(
            (stdout_thread, stderr_thread),
            provider=provider,
            task_id=task_id,
            last_output=last_output,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
            max_wall_time_sec=max_wall_time_sec,
            started_at=provider_started_at,
        )
        if drain_message is not None:
            terminate_exited_provider_group(process, force=True)
            stdout_thread.join(2)
            stderr_thread.join(2)
            if stdin_thread is not None:
                stdin_thread.join(2)
            raise ProviderCallError(
                drain_message,
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cli_exit_code=process.returncode,
            )
        stdin_delivery_incomplete = False
        if stdin_thread is not None:
            stdin_grace = 0.05
            if max_wall_time_sec is not None:
                remaining_wall = max(
                    0.0,
                    max_wall_time_sec - (self._monotonic() - provider_started_at),
                )
                stdin_grace = min(stdin_grace, remaining_wall)
            stdin_thread.join(stdin_grace)
            if stdin_thread.is_alive():
                stdin_delivery_incomplete = True
                terminate_exited_provider_group(process, force=True)
                stdin_thread.join(2)
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

        if stream_errors:
            failed_path, stream_error = stream_errors[0]
            raise ProviderCallError(
                f"Could not capture provider stream artifact {failed_path}: {stream_error}",
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cli_exit_code=process.returncode,
            ) from stream_error

        if process.returncode != 0:
            detail = compact_failure_message(stderr or stdout or "")
            retry = classify_provider_retry_failure(
                "\n".join([stdout, stderr])
            )
            raise ProviderCallError(
                f"{provider} task {task_id} exited with {process.returncode}: {detail}",
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cli_exit_code=process.returncode,
                retry_reason=retry["reason"] if retry else None,
                retry_after_sec=retry["retry_after_sec"] if retry else None,
            )

        if stdin_delivery_incomplete:
            raise ProviderCallError(
                f"{provider} task {task_id} exited before stdin delivery completed",
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cli_exit_code=process.returncode,
            )

        try:
            payload = extract_provider_json(
                stdout,
                expected_keys=tuple(expected_keys),
            )
        except DebateToolError as exc:
            retry = classify_provider_retry_failure(
                "\n".join([stdout, stderr])
            )
            if retry:
                detail = compact_failure_message(stderr or stdout or str(exc))
                raise ProviderCallError(
                    f"{provider} task {task_id} failed with retryable provider output: {detail}",
                    provider=provider,
                    task_id=task_id,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    cli_exit_code=process.returncode,
                    retry_reason=retry["reason"],
                    retry_after_sec=retry["retry_after_sec"],
                ) from exc
            detail = compact_failure_message(stderr or stdout or str(exc))
            raise ProviderCallError(
                f"{provider} task {task_id} returned invalid structured output: {detail}",
                provider=provider,
                task_id=task_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cli_exit_code=process.returncode,
            ) from exc
        return ProviderCallResult(
            payload=payload,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cli_exit_code=process.returncode,
        )

    def _wait_for_provider_process(
        self,
        process: subprocess.Popen[str],
        *,
        provider: str,
        task_id: str,
        last_output: Mapping[str, float],
        timeout_sec: float,
        poll_interval_sec: float,
        max_wall_time_sec: float | None,
        started_at: float | None = None,
    ) -> str | None:
        started = self._monotonic() if started_at is None else started_at
        while process.poll() is None:
            now = self._monotonic()
            wall_elapsed = now - started
            idle_elapsed = now - last_output["monotonic"]
            if max_wall_time_sec is not None and wall_elapsed >= max_wall_time_sec:
                return f"{provider} task {task_id} timed out after {max_wall_time_sec}s"
            if idle_elapsed >= timeout_sec:
                return (
                    f"{provider} task {task_id} idle timed out after "
                    f"{timeout_sec}s without output"
                )
            sleep_for = min(poll_interval_sec, timeout_sec - idle_elapsed)
            if max_wall_time_sec is not None:
                sleep_for = min(sleep_for, max_wall_time_sec - wall_elapsed)
            if sleep_for > 0:
                self._sleep(sleep_for)
        return None

    def _wait_for_provider_streams(
        self,
        threads: tuple[threading.Thread, threading.Thread],
        *,
        provider: str,
        task_id: str,
        last_output: Mapping[str, float],
        timeout_sec: float,
        poll_interval_sec: float,
        max_wall_time_sec: float | None,
        started_at: float,
    ) -> str | None:
        while True:
            for thread in threads:
                thread.join(0)
            if not any(thread.is_alive() for thread in threads):
                return None
            now = self._monotonic()
            wall_elapsed = now - started_at
            idle_elapsed = now - last_output["monotonic"]
            if max_wall_time_sec is not None and wall_elapsed >= max_wall_time_sec:
                return (
                    f"{provider} task {task_id} output drain timed out after "
                    f"{max_wall_time_sec}s total wall time"
                )
            if idle_elapsed >= timeout_sec:
                return (
                    f"{provider} task {task_id} output drain idle timed out after "
                    f"{timeout_sec}s without output"
                )
            sleep_for = min(0.05, poll_interval_sec, timeout_sec - idle_elapsed)
            if max_wall_time_sec is not None:
                sleep_for = min(sleep_for, max_wall_time_sec - wall_elapsed)
            if sleep_for > 0:
                self._sleep(sleep_for)

    def _build_command(
        self,
        *,
        executable: str,
        provider: str,
        model: str,
        effort: str | None,
        prompt: str,
        workspace_dir: Path,
        schema: dict[str, Any],
        schema_path: Path,
    ) -> tuple[list[str], str | None]:
        if provider == "codex":
            cmd = [
                executable,
                "--ask-for-approval",
                "never",
                "--sandbox",
                "read-only",
                "exec",
                "--cd",
                str(workspace_dir),
            ]
            if model and model != "default":
                cmd.extend(["--model", model])
            if effort and effort != "default":
                cmd.extend(["--config", f'model_reasoning_effort="{effort}"'])
            cmd.extend(
                [
                    "--json",
                    "--output-schema",
                    str(schema_path),
                    "-",
                ]
            )
            return cmd, prompt

        if provider == "claude":
            cmd = [
                executable,
                "--print",
                "--input-format",
                "text",
                "--output-format",
                "json",
                "--safe-mode",
                "--permission-mode",
                "dontAsk",
                "--tools",
                "",
                "--no-session-persistence",
                "--json-schema",
                json.dumps(schema, ensure_ascii=False),
            ]
            if model and model != "default":
                cmd.extend(["--model", model])
            if effort and effort != "default":
                cmd.extend(["--effort", effort])
            return cmd, prompt

        raise DebateToolError(f"Unsupported provider: {provider}")


def pump_provider_stream(
    stream: Any,
    path: Path,
    chunks: list[str],
    last_output: dict[str, float],
    monotonic: Any = time.monotonic,
    errors: list[tuple[Path, Exception]] | None = None,
) -> None:
    if stream is None:
        return
    handle: TextIO | None = None
    try:
        try:
            handle = path.open("w", encoding="utf-8")
        except OSError as exc:
            if errors is not None:
                errors.append((path, exc))
        while True:
            try:
                chunk = stream.read(1)
            except (OSError, UnicodeError, ValueError) as exc:
                if errors is not None:
                    errors.append((path, exc))
                break
            if chunk == "":
                break
            chunks.append(chunk)
            if handle is not None:
                try:
                    handle.write(chunk)
                    handle.flush()
                except (OSError, UnicodeError, ValueError) as exc:
                    if errors is not None:
                        errors.append((path, exc))
                    try:
                        handle.close()
                    except (OSError, UnicodeError, ValueError) as close_exc:
                        if errors is not None:
                            errors.append((path, close_exc))
                    handle = None
            last_output["monotonic"] = monotonic()
    finally:
        if handle is not None:
            try:
                handle.close()
            except (OSError, UnicodeError, ValueError) as exc:
                if errors is not None:
                    errors.append((path, exc))
        try:
            stream.close()
        except (OSError, UnicodeError, ValueError) as exc:
            if errors is not None:
                errors.append((path, exc))


def pump_provider_stdin(stream: Any, text: str) -> None:
    """Feed a provider without letting pipe backpressure block timeout enforcement."""

    if stream is None:
        return
    try:
        stream.write(text)
        stream.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def validate_provider_timing(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DebateToolError(f"{name} must be a number greater than zero", exit_code=2)
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise DebateToolError(
            f"{name} must be finite and greater than zero",
            exit_code=2,
        )
    return resolved


def terminate_provider_process(process: subprocess.Popen[str], *, force: bool) -> None:
    if process.poll() is not None:
        return
    try:
        sig = signal.SIGKILL if force else signal.SIGTERM
        if hasattr(os, "killpg"):
            os.killpg(process.pid, sig)
        else:
            process.kill() if force else process.terminate()
    except OSError:
        return


def terminate_exited_provider_group(
    process: subprocess.Popen[str],
    *,
    force: bool,
) -> None:
    """Terminate descendants that may still hold inherited provider pipes."""

    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif process.poll() is None:
            process.kill() if force else process.terminate()
    except OSError:
        return


class StructuredLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return format_log_record(
            record, timestamp=datetime.now(timezone.utc).isoformat(),
            service=SERVER_NAME, runtime="stdio-mcp",
            fields=('request_id', 'method', 'run_id', 'task_id'),
            format_exception=self.formatException,
        )


def configure_runtime_logger(*, stream: TextIO | None = None) -> logging.Logger:
    return configure_logger(LOGGER_NAME, StructuredLogFormatter, stream=stream)


def log_runtime_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    request_id: Any = None,
    method: Any = None,
    correlation_id: str | None = None,
) -> None:
    logger.log(
        level,
        message,
        extra={
            "correlation_id": correlation_id
            or (str(request_id) if request_id is not None else "server"),
            "request_id": request_id,
            "method": method,
        },
    )


class DebateOrchestrator:
    """Owns deterministic debate state and artifact writing."""

    def __init__(
        self,
        runner: ProviderRunner | None = None,
    ) -> None:
        self._injected_runner = runner
        self.runner = runner or ProviderRunner()

    def health(self) -> dict[str, Any]:
        try:
            settings = load_settings(workspace_root=WORKSPACE_ROOT, required=True)
            configuration = require_ai_debate_readiness_configuration(settings)
            runner = self._runner_for_access(configuration.access)
        except (DebateToolError, TokimonSettingsError) as exc:
            return {
                "ok": False,
                "workspace_root": str(WORKSPACE_ROOT),
                "state_root": str(DEBATE_STATE_ROOT),
                "default_runs_root": str(DEFAULT_RUNS_ROOT),
                "settings_error": str(exc),
            }
        provider_paths = {
            provider: runner.resolve_executable(provider)
            for provider in ("codex", "claude")
        }
        return {
            "ok": all(
                provider_paths[provider]
                for provider in configuration.required_providers
            ),
            "workspace_root": str(WORKSPACE_ROOT),
            "state_root": str(DEBATE_STATE_ROOT),
            "default_runs_root": str(DEFAULT_RUNS_ROOT),
            "access": configuration.access,
            "peer_b": configuration.peer_b,
            "required_providers": list(configuration.required_providers),
            "readiness_scope": "executable-discovery",
            "authentication_checked": False,
            "providers": {
                "codex": {
                    "available": bool(provider_paths["codex"]),
                    "executable": provider_paths["codex"],
                    "version": runner._capture_version("codex"),
                    "authentication": "not_checked",
                },
                "claude": {
                    "available": bool(provider_paths["claude"]),
                    "executable": provider_paths["claude"],
                    "version": runner._capture_version("claude"),
                    "authentication": "not_checked",
                },
            },
        }

    def _runner_for_access(self, access: str) -> ProviderRunner:
        if access not in SUPPORTED_ACCESS:
            raise DebateToolError(f"access must be one of {', '.join(sorted(SUPPORTED_ACCESS))}")
        if self._injected_runner is not None:
            return self._injected_runner
        return self.runner

    def run_new(self, options: RunOptions) -> dict[str, Any]:
        validate_options(options)
        context = self._create_run(options)
        try:
            self._execute(context)
        except DebateToolError as exc:
            raise DebateRunFailedError(
                str(exc),
                summary=self._run_summary(context),
                exit_code=exc.exit_code,
            ) from exc
        return self._run_summary(context)

    def start_new(self, options: RunOptions) -> dict[str, Any]:
        validate_options(options)
        context = self._create_run(options)
        return self._run_summary(context)

    def resume(self, run_dir: Path | str) -> dict[str, Any]:
        context = self._load_run(resolve_run_dir(run_dir))
        if context.state.get("status") == "completed":
            return self._run_summary(context)
        if isinstance(context.state.get("active_task"), dict):
            raise DebateRunFailedError(
                "Cannot resume while historical state.active_task is present; "
                "repair or clear the historical active task after confirming no provider job is still running, "
                "then call ai_debate_resume again from a local agent context",
                summary=self._run_summary(context),
                exit_code=2,
            )
        if context.config.get("access") != "local-cli":
            raise DebateRunFailedError(
                "Cannot resume a historical mcp.remote-agent-broker debate with the "
                "local-only AI Debate runtime; inspect it with status or read-log and "
                "start a new local-cli debate if execution must continue",
                summary=self._run_summary(context),
                exit_code=2,
            )
        try:
            self._execute(context)
        except DebateToolError as exc:
            raise DebateRunFailedError(
                str(exc),
                summary=self._run_summary(context),
                exit_code=exc.exit_code,
            ) from exc
        return self._run_summary(context)

    def resume_start(self, run_dir: Path | str) -> dict[str, Any]:
        context = self._load_run(resolve_run_dir(run_dir))
        return self._run_summary(context)

    def status(self, run_dir: Path | str) -> dict[str, Any]:
        context = self._load_run(resolve_run_dir(run_dir))
        return self._run_summary(context)

    def turn_start(self, run_dir: Path | str) -> dict[str, Any]:
        context = self._load_run(resolve_run_dir(run_dir))
        self._require_async_access(context, "ai_debate_turn_start")

    def turn_status(self, run_dir: Path | str) -> dict[str, Any]:
        context = self._load_run(resolve_run_dir(run_dir))
        self._require_async_access(context, "ai_debate_turn_status")

    def turn_finalize(self, run_dir: Path | str) -> dict[str, Any]:
        context = self._load_run(resolve_run_dir(run_dir))
        self._require_async_access(context, "ai_debate_turn_finalize")

    def read_log(self, run_dir: Path | str, *, max_chars: int | None = None) -> dict[str, Any]:
        context = self._load_run(resolve_run_dir(run_dir))
        text = read_text_artifact(context.debate_log)
        truncated = False
        if max_chars is not None and max_chars > 0 and len(text) > max_chars:
            text = text[-max_chars:]
            truncated = True
        return {
            "run_dir": str(context.run_dir),
            "debate_log": str(context.debate_log),
            "truncated": truncated,
            "text": text,
        }

    def _create_run(self, options: RunOptions) -> RunContext:
        try:
            return self._create_run_unchecked(options)
        except DebateToolError:
            raise
        except OSError as exc:
            raise DebateToolError(f"Could not create debate run artifacts: {exc}") from exc

    def _create_run_unchecked(self, options: RunOptions) -> RunContext:
        resolved = resolve_run_configuration(options)
        output_root = resolve_output_root(options.output_root)
        run_dir = allocate_run_dir(output_root)
        prompts_dir = run_dir / "prompts"
        results_dir = run_dir / "results"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        created_at = utc_now()
        config = {
            "run_id": run_dir.name,
            "created_at": created_at,
            "access": resolved["access"],
            "topic": options.topic.strip(),
            "side_a": options.side_a.strip(),
            "side_b": options.side_b.strip(),
            "side_b_provider": resolved["side_b_provider"],
            "max_rounds": options.max_rounds,
            "turn_timeout_sec": resolved["agent_execution"]["idle_timeout_sec"],
            "turn_idle_timeout_sec": resolved["agent_execution"]["idle_timeout_sec"],
            "agent_execution": resolved["agent_execution"],
            "roles": resolved["roles"],
        }
        state = {
            "status": "in_progress",
            "current_round": 1,
            "next_step": "agent_a",
            "rounds": [],
            "termination": None,
            "winner": None,
            "final_result": None,
            "last_error": None,
            "last_failure": None,
            "retry": None,
            "active_task": None,
            "task_attempts": {},
        }
        context = RunContext(run_dir=run_dir, config=config, state=state)
        write_json_atomic(run_dir / "run_config.json", config, run_dir=run_dir)
        write_json_atomic(context.state_path, state, run_dir=run_dir)
        write_json_atomic(
            run_dir / "runtime_versions.json",
            {
                "captured_at": created_at,
                "server_version": SERVER_VERSION,
                **self.runner.capture_runtime_versions(),
            },
            run_dir=run_dir,
        )
        write_text_artifact(
            context.debate_log,
            render_debate_log(config, state),
            run_dir=run_dir,
        )
        return context

    def _load_run(self, run_dir: Path) -> RunContext:
        try:
            return self._load_run_unchecked(run_dir)
        except DebateToolError:
            raise
        except (OSError, RuntimeError) as exc:
            raise DebateToolError(f"Could not load debate run {run_dir}: {exc}") from exc

    def _load_run_unchecked(self, run_dir: Path) -> RunContext:
        if not run_dir.is_dir():
            raise DebateToolError(f"Debate run not found: {run_dir}")
        config_path = require_run_artifact(run_dir, "run_config.json", kind="file")
        state_path = require_run_artifact(run_dir, "state.json", kind="file")
        debate_log = require_run_artifact(run_dir, "debate_log.md", kind="file")
        prompts_dir = require_run_artifact(run_dir, "prompts", kind="directory")
        results_dir = require_run_artifact(run_dir, "results", kind="directory")
        runtime_versions_path = require_run_artifact(
            run_dir, "runtime_versions.json", kind="file"
        )
        config = validate_persisted_run_config(read_json(config_path), config_path)
        state = validate_persisted_run_state(
            read_json(state_path),
            state_path,
            config=config,
            run_dir=run_dir,
        )
        validate_runtime_versions(read_json(runtime_versions_path), runtime_versions_path)
        del debate_log, prompts_dir, results_dir
        return RunContext(run_dir=run_dir, config=config, state=state)

    def _execute(self, context: RunContext) -> None:
        if isinstance(context.state.get("active_task"), dict):
            raise DebateToolError(
                "Cannot execute synchronously while an async active_task exists",
                exit_code=2,
            )
        self._reconcile_debate_log(context)
        context.state["status"] = "in_progress"
        context.state["last_error"] = None
        context.state["last_failure"] = None
        context.state["retry"] = None
        self._persist_state(context)
        try:
            while context.state.get("next_step") != "completed":
                next_step = context.state.get("next_step")
                if next_step == "agent_a":
                    self._run_agent_turn(context, "agent_a")
                elif next_step == "agent_b":
                    self._run_agent_turn(context, "agent_b")
                elif next_step == "final_judgment":
                    self._run_final_judgment(context)
                else:
                    raise DebateToolError(f"Unknown next_step: {next_step}", exit_code=2)
        except ProviderCallError as exc:
            if exc.retryable:
                try:
                    self._record_waiting_retry(context, exc)
                except DebateToolError as persist_exc:
                    raise DebateToolError(
                        f"{exc}; could not persist retry state: {persist_exc}",
                        exit_code=exc.exit_code,
                    ) from persist_exc
                return
            try:
                self._record_failure(context, exc)
            except DebateToolError as persist_exc:
                raise DebateToolError(
                    f"{exc}; could not persist failure state: {persist_exc}",
                    exit_code=exc.exit_code,
                ) from persist_exc
            raise
        except DebateToolError as exc:
            try:
                self._record_failure(context, exc)
            except DebateToolError as persist_exc:
                raise DebateToolError(
                    f"{exc}; could not persist failure state: {persist_exc}",
                    exit_code=exc.exit_code,
                ) from persist_exc
            raise
        except OSError as exc:
            wrapped = DebateToolError(f"Debate artifact I/O failed: {exc}")
            try:
                self._record_failure(context, wrapped)
            except DebateToolError as persist_exc:
                raise DebateToolError(
                    f"{wrapped}; could not persist failure state: {persist_exc}"
                ) from persist_exc
            raise wrapped from exc

    def _require_async_access(self, context: RunContext, tool_name: str) -> None:
        del context
        raise DebateToolError(
            f"{tool_name} is unavailable because AI Debate always uses local-cli; "
            "use ai_debate_resume from a local agent context",
            exit_code=2,
        )

    def _run_agent_turn(self, context: RunContext, role: str) -> None:
        round_number = int(context.state["current_round"])
        task_id = f"round-{round_number:02d}-{'agent-a' if role == 'agent_a' else 'agent-b'}"
        prompt = render_agent_prompt(context, role)
        prompt_path = context.prompts_dir / f"{task_id}.md"
        require_task_write_targets(context, task_id)
        write_text_artifact(prompt_path, prompt, run_dir=context.run_dir)

        role_config = context.config["roles"][role]
        execution = context.config["agent_execution"]
        provider_runner = self._runner_for_access(str(context.config["access"]))
        call_result = provider_runner.run_json(
            provider=role_config["provider"],
            model=role_config["model"],
            effort=role_config.get("effort"),
            prompt=prompt,
            timeout_sec=int(execution["idle_timeout_sec"]),
            poll_interval_sec=float(execution["poll_interval_sec"]),
            max_wall_time_sec=(
                int(execution["max_wall_time_sec"])
                if execution.get("max_wall_time_sec") is not None
                else None
            ),
            workspace_dir=context.run_dir,
            task_id=task_id,
            results_dir=context.results_dir,
            expected_keys=("response_md", "signal", "signal_reason"),
        )
        turn_record = validate_turn_payload(call_result.payload)
        result_path = self._write_task_result(
            context=context,
            task_id=task_id,
            role=role,
            provider=role_config["provider"],
            model=role_config["model"],
            payload=turn_record,
            call_result=call_result,
        )
        turn_record.update(
            {
                "task_id": task_id,
                "result_path": relative_to(context.run_dir, result_path),
            }
        )
        self._append_turn_to_log(context, role, turn_record)
        self._advance_after_turn(context, role, turn_record)
        self._persist_state(context)

    def _run_final_judgment(self, context: RunContext) -> None:
        task_id = "final-judgment"
        prompt = render_final_prompt(context)
        prompt_path = context.prompts_dir / f"{task_id}.md"
        require_task_write_targets(context, task_id)
        write_text_artifact(prompt_path, prompt, run_dir=context.run_dir)

        role_config = context.config["roles"]["judge"]
        execution = context.config["agent_execution"]
        provider_runner = self._runner_for_access(str(context.config["access"]))
        call_result = provider_runner.run_json(
            provider=role_config["provider"],
            model=role_config["model"],
            effort=role_config.get("effort"),
            prompt=prompt,
            timeout_sec=int(execution["idle_timeout_sec"]),
            poll_interval_sec=float(execution["poll_interval_sec"]),
            max_wall_time_sec=(
                int(execution["max_wall_time_sec"])
                if execution.get("max_wall_time_sec") is not None
                else None
            ),
            workspace_dir=context.run_dir,
            task_id=task_id,
            results_dir=context.results_dir,
            expected_keys=("winner", "reason", "termination", "decisive_round"),
        )
        final_result = validate_final_payload(call_result.payload, context.state)
        result_path = self._write_task_result(
            context=context,
            task_id=task_id,
            role="judge",
            provider=role_config["provider"],
            model=role_config["model"],
            payload=final_result,
            call_result=call_result,
        )
        final_result["task_id"] = task_id
        final_result["result_path"] = relative_to(context.run_dir, result_path)
        self._append_final_to_log(context, final_result)
        context.state["final_result"] = final_result
        context.state["status"] = "completed"
        context.state["next_step"] = "completed"
        context.state["last_error"] = None
        context.state["last_failure"] = None
        context.state["retry"] = None
        self._persist_state(context)

    def _write_task_result(
        self,
        *,
        context: RunContext,
        task_id: str,
        role: str,
        provider: str,
        model: str,
        payload: dict[str, Any],
        call_result: ProviderCallResult,
    ) -> Path:
        result_path = context.results_dir / f"{task_id}.json"
        write_json_atomic(
            result_path,
            {
                "id": task_id,
                "role": role,
                "provider": provider,
                "model": model,
                "status": "success",
                "response_json": payload,
                "stdout_path": (
                    relative_to(context.run_dir, call_result.stdout_path)
                    if call_result.stdout_path
                    else None
                ),
                "stderr_path": (
                    relative_to(context.run_dir, call_result.stderr_path)
                    if call_result.stderr_path
                    else None
                ),
                "cli_exit_code": call_result.cli_exit_code,
                "finished_at": utc_now(),
            },
            run_dir=context.run_dir,
        )
        return result_path

    def _append_turn_to_log(
        self,
        context: RunContext,
        role: str,
        turn_record: dict[str, Any],
    ) -> None:
        task_id = str(turn_record.get("task_id") or "")
        marker = debate_log_task_marker(task_id) if task_id else None
        if marker and marker in read_text_artifact(context.debate_log):
            return
        append_text_artifact(
            context.debate_log,
            render_turn_log_entry(
                round_number=int(context.state["current_round"]),
                role=role,
                turn_record=turn_record,
            ),
            run_dir=context.run_dir,
        )

    def _append_final_to_log(self, context: RunContext, final_result: dict[str, Any]) -> None:
        task_id = str(final_result.get("task_id") or "final-judgment")
        marker = debate_log_task_marker(task_id)
        if marker in read_text_artifact(context.debate_log):
            return
        append_text_artifact(
            context.debate_log,
            render_final_log_entry(final_result),
            run_dir=context.run_dir,
        )

    def _reconcile_debate_log(self, context: RunContext) -> None:
        expected = render_debate_log(context.config, context.state)
        if read_text_artifact(context.debate_log) == expected:
            return
        write_text_artifact_atomic(
            context.debate_log,
            expected,
            run_dir=context.run_dir,
        )

    def _advance_after_turn(
        self,
        context: RunContext,
        role: str,
        turn_record: dict[str, Any],
    ) -> None:
        round_record = ensure_round(context.state, int(context.state["current_round"]))
        round_record[role] = turn_record

        if role == "agent_a":
            if turn_record["signal"] == "concede":
                context.state["termination"] = "concession"
                context.state["winner"] = "agent_b"
                context.state["next_step"] = "final_judgment"
            else:
                context.state["next_step"] = "agent_b"
            return

        if turn_record["signal"] == "concede":
            context.state["termination"] = "concession"
            context.state["winner"] = "agent_a"
            context.state["next_step"] = "final_judgment"
            return

        agent_a = round_record.get("agent_a")
        if agent_a and agent_a.get("signal") == "tie" and turn_record["signal"] == "tie":
            context.state["termination"] = "mutual_tie"
            context.state["winner"] = "tie"
            context.state["next_step"] = "final_judgment"
            return

        if int(context.state["current_round"]) >= int(context.config["max_rounds"]):
            context.state["termination"] = "round_limit"
            context.state["winner"] = None
            context.state["next_step"] = "final_judgment"
            return

        context.state["current_round"] = int(context.state["current_round"]) + 1
        context.state["next_step"] = "agent_a"

    def _record_waiting_retry(self, context: RunContext, exc: ProviderCallError) -> None:
        failure = self._failure_metadata(context, exc)
        context.state["status"] = "waiting_retry"
        context.state["last_error"] = str(exc)
        context.state["last_failure"] = failure
        context.state["retry"] = failure
        context.state["active_task"] = None
        self._write_task_failure(context, failure)
        self._persist_state(context)

    def _record_failure(self, context: RunContext, exc: DebateToolError) -> None:
        failure = self._failure_metadata(context, exc)
        context.state["status"] = "failed"
        context.state["last_error"] = str(exc)
        context.state["last_failure"] = failure
        context.state["retry"] = None
        context.state["active_task"] = None
        if isinstance(exc, ProviderCallError):
            self._write_task_failure(context, failure)
        self._persist_state(context)

    def _failure_metadata(
        self,
        context: RunContext,
        exc: DebateToolError,
    ) -> dict[str, Any]:
        detected_at = datetime.now(timezone.utc)
        failure: dict[str, Any] = {
            "retryable": False,
            "reason": "tool_failure",
            "message": compact_failure_message(str(exc)),
            "detected_at": detected_at.isoformat(),
            "next_step": context.state.get("next_step"),
        }
        if isinstance(exc, ProviderCallError):
            failure.update(
                {
                    "reason": exc.retry_reason or "provider_failure",
                    "provider": exc.provider,
                    "task_id": exc.task_id,
                    "role": role_for_next_step(context.state.get("next_step")),
                    "stdout_path": (
                        relative_to(context.run_dir, exc.stdout_path)
                        if exc.stdout_path is not None and exc.stdout_path.is_file()
                        else None
                    ),
                    "stderr_path": (
                        relative_to(context.run_dir, exc.stderr_path)
                        if exc.stderr_path is not None and exc.stderr_path.is_file()
                        else None
                    ),
                    "cli_exit_code": exc.cli_exit_code,
                }
            )
            if exc.retryable:
                retry_after_sec = int(exc.retry_after_sec or DEFAULT_RETRY_AFTER_SEC)
                failure.update(
                    {
                        "retryable": True,
                        "retry_after_sec": retry_after_sec,
                        "retry_after_at": (
                            detected_at + timedelta(seconds=retry_after_sec)
                        ).isoformat(),
                    }
                )
        return failure

    def _write_task_failure(self, context: RunContext, failure: Mapping[str, Any]) -> None:
        task_id = failure.get("task_id")
        if not task_id:
            return
        role = failure.get("role")
        if not isinstance(role, str) or role not in context.config["roles"]:
            raise DebateToolError("Cannot persist provider failure without a valid role")
        result_path = context.results_dir / f"{task_id}.json"
        write_json_atomic(
            result_path,
            {
                "id": task_id,
                "role": role,
                "provider": failure.get("provider"),
                "model": context.config["roles"][role]["model"],
                "status": "waiting_retry" if failure.get("retryable") else "failed",
                "failure": dict(failure),
                "stdout_path": failure.get("stdout_path"),
                "stderr_path": failure.get("stderr_path"),
                "cli_exit_code": failure.get("cli_exit_code"),
                "finished_at": failure.get("detected_at") or utc_now(),
            },
            run_dir=context.run_dir,
        )

    def _persist_state(self, context: RunContext) -> None:
        write_json_atomic(
            context.state_path,
            context.state,
            run_dir=context.run_dir,
        )

    def _run_summary(self, context: RunContext) -> dict[str, Any]:
        return {
            "run_dir": str(context.run_dir),
            "debate_log": str(context.debate_log),
            "state_path": str(context.state_path),
            "status": context.state.get("status"),
            "next_step": context.state.get("next_step"),
            "current_round": context.state.get("current_round"),
            "termination": context.state.get("termination"),
            "winner": context.state.get("winner"),
            "final_result": context.state.get("final_result"),
            "last_error": context.state.get("last_error"),
            "last_failure": context.state.get("last_failure"),
            "retry": context.state.get("retry"),
            "active_task": context.state.get("active_task"),
        }


def validate_options(options: RunOptions) -> None:
    for field, value in (
        ("topic", options.topic),
        ("side_a", options.side_a),
        ("side_b", options.side_b),
    ):
        if not isinstance(value, str) or not value.strip():
            raise DebateToolError(f"{field} is required")
    if (
        options.side_b_provider is not None
        and options.side_b_provider not in SUPPORTED_SIDE_B_PROVIDERS
    ):
        raise DebateToolError("side_b_provider must be claude or codex")
    if options.access is not None and options.access not in SUPPORTED_ACCESS:
        raise DebateToolError("access must be local-cli")
    if (
        isinstance(options.max_rounds, bool)
        or not isinstance(options.max_rounds, int)
        or options.max_rounds <= 0
    ):
        raise DebateToolError("max_rounds must be a positive integer")
    for field, value in (
        ("turn_idle_timeout_sec", options.turn_idle_timeout_sec),
        ("turn_timeout_sec", options.turn_timeout_sec),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise DebateToolError(f"{field} must be a positive integer")
    for field, value in (
        ("model", options.model),
        ("effort", options.effort),
        ("agent_a_model", options.agent_a_model),
        ("agent_b_model", options.agent_b_model),
        ("agent_b_effort", options.agent_b_effort),
        ("judge_model", options.judge_model),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise DebateToolError(f"{field} must be a non-empty string")
    if options.output_root is not None and not isinstance(options.output_root, Path):
        raise DebateToolError("output_root must be a pathlib.Path")


def ensure_round(state: dict[str, Any], round_number: int) -> dict[str, Any]:
    rounds = state.setdefault("rounds", [])
    for record in rounds:
        if int(record.get("round_number", 0)) == round_number:
            return record
    record = {
        "round_number": round_number,
        "agent_a": None,
        "agent_b": None,
    }
    rounds.append(record)
    rounds.sort(key=lambda item: int(item["round_number"]))
    return record


def render_debate_log(config: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    parts = [
        "# Debate Topic\n"
        f"{config['topic']}\n\n"
        "# Position A\n"
        f"{config['side_a']}\n\n"
        "# Position B\n"
        f"{config['side_b']}\n"
    ]
    for round_record in state.get("rounds", []):
        round_number = int(round_record["round_number"])
        for role in ("agent_a", "agent_b"):
            turn_record = round_record.get(role)
            if isinstance(turn_record, Mapping):
                parts.append(
                    render_turn_log_entry(
                        round_number=round_number,
                        role=role,
                        turn_record=turn_record,
                    )
                )
    final_result = state.get("final_result")
    if isinstance(final_result, Mapping):
        parts.append(render_final_log_entry(final_result))
    return "".join(parts)


def render_turn_log_entry(
    *,
    round_number: int,
    role: str,
    turn_record: Mapping[str, Any],
) -> str:
    task_id = str(turn_record.get("task_id") or "")
    lines: list[str] = []
    if task_id:
        lines.append(debate_log_task_marker(task_id))
    if role == "agent_a":
        lines.extend(["", f"## Round {round_number}", "", "### Agent A"])
    else:
        lines.append("### Agent B")
    lines.extend(
        [
            str(turn_record["response_md"]).strip(),
            "",
            f"> Debate signal: {turn_record['signal']}",
            f"> Signal reason: {turn_record['signal_reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def render_final_log_entry(final_result: Mapping[str, Any]) -> str:
    task_id = str(final_result.get("task_id") or "final-judgment")
    return "\n".join(
        [
            "",
            debate_log_task_marker(task_id),
            "# Final Result",
            f"Winner: {WINNER_LABELS[str(final_result['winner'])]}",
            f"Reason: {final_result['reason']}",
            f"Termination: {final_result['termination']}",
            "",
        ]
    )


def render_agent_prompt(context: RunContext, role: str) -> str:
    config = context.config
    role_label = "Agent A" if role == "agent_a" else "Agent B"
    position = config["side_a"] if role == "agent_a" else config["side_b"]
    opponent_label = "Agent B" if role == "agent_a" else "Agent A"
    transcript = read_text_artifact(context.debate_log)
    latest_opponent = latest_opponent_response(context.state, role)
    return (
        f"You are {role_label} in a structured AI debate.\n\n"
        f"Topic:\n{config['topic']}\n\n"
        f"Your assigned position:\n{position}\n\n"
        f"Current round: {context.state['current_round']}\n"
        f"Shared transcript path: {context.debate_log}\n"
        f"Opponent: {opponent_label}\n\n"
        "Rules:\n"
        "- Strengthen your assigned position and directly engage the latest opposing argument.\n"
        "- Push the reasoning forward; do not repeat slogans.\n"
        "- The host, not prose, controls termination through your JSON signal.\n"
        "- Return JSON only, with no Markdown fence and no commentary outside JSON.\n\n"
        "Signal meanings:\n"
        "- continue: keep debating.\n"
        "- concede: you explicitly lose and the opponent wins immediately.\n"
        "- tie: you judge the debate inconclusive; a tie only ends the debate if both agents emit tie in the same round.\n\n"
        f"Latest opponent response:\n{latest_opponent}\n\n"
        f"Transcript so far:\n{transcript}\n\n"
        "Return this JSON object:\n"
        "{\n"
        '  "response_md": "Markdown body for the debate log",\n'
        '  "signal": "continue",\n'
        '  "signal_reason": "One short sentence",\n'
        '  "citations": [],\n'
        '  "files_created": []\n'
        "}\n"
    )


def render_final_prompt(context: RunContext) -> str:
    transcript = read_text_artifact(context.debate_log)
    termination = context.state.get("termination") or "round_limit"
    fixed = ""
    if termination in {"concession", "mutual_tie"}:
        fixed = (
            "The host has already fixed the outcome. "
            f"Return winner={context.state.get('winner')}, termination={termination}, "
            f"decisive_round={context.state.get('current_round')}. "
            "Only provide a concise reason consistent with that outcome."
        )
    else:
        fixed = "Choose winner=agent_a, agent_b, or tie based on the transcript. Set termination=round_limit."
    return (
        "You are the final judge for a structured AI debate.\n\n"
        f"Topic:\n{context.config['topic']}\n\n"
        f"Host termination: {termination}\n"
        f"Host constraints: {fixed}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return JSON only:\n"
        "{\n"
        '  "winner": "agent_a",\n'
        '  "reason": "Short justification",\n'
        f'  "termination": "{termination}",\n'
        f'  "decisive_round": {context.state.get("current_round")}\n'
        "}\n"
    )


def latest_opponent_response(state: dict[str, Any], role: str) -> str:
    rounds = state.get("rounds", [])
    if role == "agent_b":
        current = ensure_round(state, int(state["current_round"]))
        agent_a = current.get("agent_a")
        if agent_a:
            return str(agent_a.get("response_md", ""))
        return "No Agent A response exists for this round yet."
    for record in reversed(rounds):
        agent_b = record.get("agent_b")
        if agent_b:
            return str(agent_b.get("response_md", ""))
    return "No Agent B response exists yet."


def validate_turn_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    response_md = require_payload_text(payload.get("response_md"), "response_md", "turn")
    signal = require_payload_text(payload.get("signal"), "signal", "turn")
    signal_reason = require_payload_text(
        payload.get("signal_reason"), "signal_reason", "turn"
    )
    if signal not in {"continue", "concede", "tie"}:
        raise DebateToolError(f"turn response has invalid signal: {signal}", exit_code=2)
    return {
        "response_md": response_md,
        "signal": signal,
        "signal_reason": signal_reason,
        "citations": optional_payload_string_list(payload.get("citations"), "citations"),
        "files_created": optional_payload_string_list(
            payload.get("files_created"), "files_created"
        ),
    }


def validate_final_payload(
    payload: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    reason = require_payload_text(payload.get("reason"), "reason", "final judgment")
    termination = require_payload_text(
        payload.get("termination"), "termination", "final judgment"
    )
    winner = require_payload_text(payload.get("winner"), "winner", "final judgment")
    decisive_round_raw = payload.get("decisive_round")
    if decisive_round_raw is not None and (
        isinstance(decisive_round_raw, bool) or not isinstance(decisive_round_raw, int)
    ):
        raise DebateToolError(
            f"final judgment has invalid decisive_round: {decisive_round_raw}",
            exit_code=2,
        )
    decisive_round = decisive_round_raw

    host_termination = state.get("termination")
    if host_termination in {"concession", "mutual_tie"}:
        winner = str(state.get("winner"))
        termination = str(host_termination)
        decisive_round = int(state.get("current_round", 0))
    else:
        if winner not in {"agent_a", "agent_b", "tie"}:
            raise DebateToolError(f"final judgment has invalid winner: {winner}", exit_code=2)
        if termination != "round_limit":
            raise DebateToolError(
                f"round-limit judgment has invalid termination: {termination}",
                exit_code=2,
            )
        if decisive_round is None:
            decisive_round = int(state.get("current_round", 0))
        if decisive_round != int(state.get("current_round", 0)):
            raise DebateToolError(
                "round-limit judgment decisive_round must match current_round",
                exit_code=2,
            )

    return {
        "winner": winner,
        "reason": reason,
        "termination": termination,
        "decisive_round": decisive_round,
    }


def require_payload_text(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DebateToolError(f"{context} missing or invalid {field}", exit_code=2)
    return value.strip()


def optional_payload_string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DebateToolError(f"turn response {field} must be an array of strings", exit_code=2)
    return [item.strip() for item in value if item.strip()]


def role_for_next_step(next_step: Any) -> str | None:
    if next_step in {"agent_a", "agent_b"}:
        return str(next_step)
    if next_step == "final_judgment":
        return "judge"
    return None


def classify_provider_retry_failure(text: str) -> dict[str, Any] | None:
    normalized = text.lower()
    if not normalized.strip():
        return None

    reason: str | None = None
    if matches_any(
        normalized,
        (
            r"\btoken quota\b",
            r"\btokens? exhausted\b",
            r"\bout of tokens\b",
            r"\btokens? limit(?: reached| exceeded)?\b",
        ),
    ):
        reason = "token_exhausted"
    elif matches_any(
        normalized,
        (
            r"\binsufficient[_\s-]?quota",
            r"\bquota\b",
            r"\busage limit(?: reached| exceeded)?\b",
            r"\bcredit balance\b",
            r"\bcredits? exhausted\b",
            r"\bbilling hard limit\b",
            r"\bmonthly limit\b",
        ),
    ):
        reason = "quota_exhausted"
    elif matches_any(
        normalized,
        (
            r"\brate[-_\s]?limit",
            r"\b429\b",
            r"\btoo many requests\b",
            r"\btry again later\b",
            r"\bretry later\b",
        ),
    ):
        reason = "rate_limit"

    if reason is None:
        return None

    return {
        "reason": reason,
        "retry_after_sec": parse_retry_after_seconds(text) or DEFAULT_RETRY_AFTER_SEC,
    }


def matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def parse_retry_after_seconds(text: str) -> int | None:
    patterns = (
        r"\bretry[-\s]?after\s*[:=]?\s*(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?\b",
        r"\btry again in\s+(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b",
        r"\bretry in\s+(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        amount = int(match.group(1))
        unit = (match.group(2) or "seconds").lower()
        if unit.startswith(("hour", "hr", "h")):
            return amount * 60 * 60
        if unit.startswith(("minute", "min", "m")):
            return amount * 60
        return amount
    return None


def compact_failure_message(text: str, *, max_chars: int = 600) -> str:
    message = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    message = " ".join(message.strip().split())
    if not message:
        return "Provider call failed"
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3].rstrip() + "..."


def schema_for_expected_keys(expected_keys: tuple[str, ...]) -> dict[str, Any]:
    if "response_md" in expected_keys:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "response_md": {"type": "string", "minLength": 1},
                "signal": {
                    "type": "string",
                    "enum": ["continue", "concede", "tie"],
                },
                "signal_reason": {"type": "string", "minLength": 1},
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "files_created": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "response_md",
                "signal",
                "signal_reason",
                "citations",
                "files_created",
            ],
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["agent_a", "agent_b", "tie"],
            },
            "reason": {"type": "string", "minLength": 1},
            "termination": {
                "type": "string",
                "enum": ["concession", "mutual_tie", "round_limit"],
            },
            "decisive_round": {
                "type": ["integer", "null"],
            },
        },
        "required": ["winner", "reason", "termination", "decisive_round"],
    }


def extract_provider_json(text: str, *, expected_keys: tuple[str, ...]) -> dict[str, Any]:
    direct = parse_json_text(text, expected_keys=expected_keys)
    if direct is not None:
        return direct

    event_texts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            direct_event = unwrap_json_candidate(event, expected_keys=expected_keys)
            if direct_event is not None:
                return direct_event
            event_text = extract_event_text(event)
            if event_text:
                event_texts.append(event_text)

    if event_texts:
        parsed = parse_json_text("\n".join(event_texts), expected_keys=expected_keys)
        if parsed is not None:
            return parsed

    raise DebateToolError("provider output did not contain the required JSON object", exit_code=2)


def parse_json_text(text: str, *, expected_keys: tuple[str, ...]) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        unwrapped = unwrap_json_candidate(loaded, expected_keys=expected_keys)
        if unwrapped is not None:
            return unwrapped

    decoder = json.JSONDecoder()
    matches: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and all(key in value for key in expected_keys):
            matches.append(value)
    return matches[-1] if matches else None


def unwrap_json_candidate(
    value: Mapping[str, Any],
    *,
    expected_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    if all(key in value for key in expected_keys):
        return dict(value)
    for key in ("response_json", "result", "response", "content", "text", "message"):
        nested = value.get(key)
        if isinstance(nested, dict) and all(item in nested for item in expected_keys):
            return dict(nested)
        if isinstance(nested, str):
            parsed = parse_json_text(nested, expected_keys=expected_keys)
            if parsed is not None:
                return parsed
    return None


def extract_event_text(event: Mapping[str, Any]) -> str | None:
    if event.get("type") == "assistant.message":
        data = event.get("data")
        if isinstance(data, dict):
            content = data.get("content")
            return str(content) if content else None
    item = event.get("item")
    if isinstance(item, dict):
        if item.get("type") == "agent_message":
            text = item.get("text") or item.get("message")
            return str(text) if text else None
        content = item.get("content")
        if isinstance(content, list):
            parts = [
                str(part.get("text"))
                for part in content
                if isinstance(part, dict) and part.get("text")
            ]
            if parts:
                return "\n".join(parts)
    msg = event.get("msg")
    if isinstance(msg, dict) and msg.get("type") == "agent_message":
        text = msg.get("text") or msg.get("message")
        return str(text) if text else None
    return None


def resolve_output_root(output_root: Path | None) -> Path:
    if output_root is None:
        root = DEFAULT_RUNS_ROOT
    else:
        root = output_root.expanduser()
        if not root.is_absolute():
            root = resolve_runtime_path(root, workspace_root=WORKSPACE_ROOT)
    root = root.resolve()
    ensure_allowed_artifact_path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_run_dir(run_dir: Path | str) -> Path:
    path = Path(run_dir).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        workspace_relative = resolve_runtime_path(path, workspace_root=WORKSPACE_ROOT).resolve()
        default_relative = (DEFAULT_RUNS_ROOT / path).resolve()
        resolved = default_relative if default_relative.exists() else workspace_relative
    ensure_allowed_artifact_path(resolved)
    return resolved


def ensure_allowed_artifact_path(path: Path) -> None:
    resolved = path.resolve()
    allowed_roots = allowed_artifact_roots()
    if any(is_relative_to(resolved, root) for root in allowed_roots):
        return
    roots = ", ".join(str(root) for root in allowed_roots)
    raise DebateToolError(
        f"Debate artifacts must stay under the Tokimon runtime or repository root ({roots}): {path}",
        exit_code=1,
    )


def allowed_artifact_roots() -> tuple[Path, ...]:
    roots = {
        WORKSPACE_STATE_ROOT.resolve(),
        WORKSPACE_ROOT.resolve(),
        WORKSPACE_STATE_ROOT.resolve().parent,
    }
    return tuple(sorted(roots, key=lambda root: root.as_posix()))


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def allocate_run_dir(output_root: Path) -> Path:
    base = output_root / f"debate_{timestamp_slug()}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def require_run_write_path(run_dir: Path, path: Path, *, kind: str) -> Path:
    """Reject non-canonical or symlink-backed write targets inside a run."""

    if kind not in {"file", "directory"}:
        raise DebateToolError(f"Unsupported debate artifact kind: {kind}")
    try:
        lexical_run = Path(os.path.abspath(run_dir))
        lexical_path = Path(os.path.abspath(path))
        relative_path = lexical_path.relative_to(lexical_run)
        resolved_run = run_dir.resolve()
        resolved_parent = path.parent.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise DebateToolError(
            f"Debate write target is outside or cannot be resolved from {run_dir}: {path}"
        ) from exc
    if not is_relative_to(resolved_parent, resolved_run):
        raise DebateToolError(
            f"Debate artifact escapes run boundary {resolved_run}: {path}"
        )

    cursor = lexical_run
    for part in relative_path.parts:
        cursor /= part
        if cursor.is_symlink():
            raise DebateToolError(
                f"Debate write target cannot use a symbolic link: {path}"
            )

    if path.exists():
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            raise DebateToolError(
                f"Debate write target must be a {kind}: {path}"
            )
    return path


def require_task_write_targets(context: RunContext, task_id: str) -> None:
    """Validate every canonical target that one provider task may overwrite."""

    targets = (
        context.prompts_dir / f"{task_id}.md",
        context.results_dir / f"{task_id}.schema.json",
        context.results_dir / f"{task_id}.stdout",
        context.results_dir / f"{task_id}.stderr",
        context.results_dir / f"{task_id}.json",
        context.results_dir / f"{task_id}.json.tmp",
    )
    for target in targets:
        require_run_write_path(context.run_dir, target, kind="file")


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    run_dir: Path | None = None,
) -> None:
    try:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if run_dir is not None:
            require_run_write_path(run_dir, path, kind="file")
            require_run_write_path(run_dir, tmp_path, kind="file")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        raise DebateToolError(f"Could not write debate artifact {path}: {exc}") from exc


def write_text_artifact(
    path: Path,
    text: str,
    *,
    run_dir: Path | None = None,
) -> None:
    try:
        if run_dir is not None:
            require_run_write_path(run_dir, path, kind="file")
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise DebateToolError(f"Could not write debate artifact {path}: {exc}") from exc


def write_text_artifact_atomic(
    path: Path,
    text: str,
    *,
    run_dir: Path | None = None,
) -> None:
    try:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        if run_dir is not None:
            require_run_write_path(run_dir, path, kind="file")
            require_run_write_path(run_dir, tmp_path, kind="file")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except OSError as exc:
        raise DebateToolError(f"Could not write debate artifact {path}: {exc}") from exc


def append_text_artifact(
    path: Path,
    text: str,
    *,
    run_dir: Path | None = None,
) -> None:
    try:
        if run_dir is not None:
            require_run_write_path(run_dir, path, kind="file")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError as exc:
        raise DebateToolError(f"Could not append debate artifact {path}: {exc}") from exc


def read_text_artifact(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DebateToolError(f"Could not read debate artifact {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DebateToolError(f"Could not read debate artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DebateToolError(
            f"Malformed JSON in debate artifact {path}: "
            f"{exc.msg} at line {exc.lineno} column {exc.colno}"
        ) from exc


def require_run_artifact(run_dir: Path, relative_path: str, *, kind: str) -> Path:
    artifact_path = run_dir / relative_path
    try:
        resolved_run = run_dir.resolve()
        resolved_artifact = artifact_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise DebateToolError(
            f"Malformed debate run {run_dir}: cannot resolve {relative_path}: {exc}"
        ) from exc
    if not is_relative_to(resolved_artifact, resolved_run):
        raise DebateToolError(
            f"Malformed debate run {run_dir}: artifact escapes the run boundary: "
            f"{relative_path}"
        )
    require_run_write_path(run_dir, artifact_path, kind=kind)
    valid = resolved_artifact.is_file() if kind == "file" else resolved_artifact.is_dir()
    if not valid:
        raise DebateToolError(
            f"Malformed debate run {run_dir}: required {kind} is missing: {relative_path}"
        )
    return artifact_path


def validate_runtime_versions(value: Any, path: Path) -> dict[str, Any]:
    payload = require_persisted_object(
        value,
        path,
        required_fields=("captured_at", "server_version", "python", "codex", "claude"),
    )
    require_persisted_timestamp(payload["captured_at"], path, "captured_at")
    for key in ("server_version", "python", "codex", "claude"):
        require_persisted_text(payload[key], path, key)
    return payload


def validate_persisted_run_config(value: Any, path: Path) -> dict[str, Any]:
    config = require_persisted_object(
        value,
        path,
        required_fields=(
            "run_id",
            "created_at",
            "access",
            "topic",
            "side_a",
            "side_b",
            "side_b_provider",
            "max_rounds",
            "agent_execution",
            "roles",
        ),
    )
    run_id = require_persisted_text(config["run_id"], path, "run_id")
    if run_id != path.parent.name:
        raise malformed_artifact(path, "run_id does not match the run directory")
    require_persisted_timestamp(config["created_at"], path, "created_at")
    for key in ("topic", "side_a", "side_b"):
        require_persisted_text(config[key], path, key)
    access = config["access"]
    if not isinstance(access, str) or access not in PERSISTED_ACCESS:
        raise malformed_artifact(
            path, "access must be local-cli or historical mcp.remote-agent-broker"
        )
    side_b_provider = config["side_b_provider"]
    if (
        not isinstance(side_b_provider, str)
        or side_b_provider not in SUPPORTED_SIDE_B_PROVIDERS
    ):
        raise malformed_artifact(path, "side_b_provider must be claude or codex")
    require_persisted_positive_int(config["max_rounds"], path, "max_rounds")

    execution = require_persisted_object(
        config["agent_execution"],
        path,
        field="agent_execution",
        required_fields=("idle_timeout_sec", "poll_interval_sec", "max_wall_time_sec"),
    )
    require_persisted_positive_int(
        execution["idle_timeout_sec"], path, "agent_execution.idle_timeout_sec"
    )
    poll_interval = execution["poll_interval_sec"]
    if isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float)):
        raise malformed_artifact(path, "agent_execution.poll_interval_sec must be a number")
    if not math.isfinite(float(poll_interval)) or poll_interval <= 0:
        raise malformed_artifact(
            path,
            "agent_execution.poll_interval_sec must be finite and greater than zero",
        )
    max_wall_time = execution["max_wall_time_sec"]
    if max_wall_time is not None:
        require_persisted_positive_int(
            max_wall_time, path, "agent_execution.max_wall_time_sec"
        )

    roles = require_persisted_object(config["roles"], path, field="roles")
    expected_providers = {
        "agent_a": "codex",
        "agent_b": config["side_b_provider"],
        "judge": "codex",
    }
    for role in RUN_ROLE_NAMES:
        role_config = require_persisted_object(
            roles.get(role),
            path,
            field=f"roles.{role}",
            required_fields=("provider", "model", "effort"),
        )
        provider = require_persisted_text(
            role_config["provider"], path, f"roles.{role}.provider"
        )
        if provider != expected_providers[role]:
            raise malformed_artifact(
                path,
                f"roles.{role}.provider must be {expected_providers[role]}",
            )
        require_persisted_text(role_config["model"], path, f"roles.{role}.model")
        require_persisted_text(role_config["effort"], path, f"roles.{role}.effort")
    return config


def validate_persisted_run_state(
    value: Any,
    path: Path,
    *,
    config: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    state = require_persisted_object(
        value,
        path,
        required_fields=(
            "status",
            "current_round",
            "next_step",
            "rounds",
            "termination",
            "winner",
            "final_result",
            "last_error",
            "last_failure",
            "retry",
        ),
    )
    # These keys were introduced by the historical async implementation. Older
    # local-only runs remain valid without them, so normalize in memory rather
    # than mutating persisted artifacts during read-only operations.
    state.setdefault("active_task", None)
    state.setdefault("task_attempts", {})
    status = state["status"]
    next_step = state["next_step"]
    if not isinstance(status, str) or status not in RUN_STATUS_VALUES:
        raise malformed_artifact(path, "status is unsupported")
    if not isinstance(next_step, str) or next_step not in RUN_NEXT_STEP_VALUES:
        raise malformed_artifact(path, "next_step is unsupported")
    current_round = require_persisted_positive_int(
        state["current_round"], path, "current_round"
    )
    if current_round > int(config["max_rounds"]):
        raise malformed_artifact(path, "current_round exceeds run_config.max_rounds")

    run_created_at = require_persisted_timestamp(
        config["created_at"], path, "run_config.created_at"
    )
    historical_remote = config["access"] == "mcp.remote-agent-broker"
    rounds_by_number = validate_persisted_rounds(
        state["rounds"],
        path,
        config=config,
        run_dir=run_dir,
        run_created_at=run_created_at,
        historical_remote=historical_remote,
    )
    expected_last_round = current_round - 1 if next_step == "agent_a" else current_round
    expected_round_numbers = set(range(1, expected_last_round + 1))
    if set(rounds_by_number) != expected_round_numbers:
        raise malformed_artifact(
            path,
            "rounds must contain one contiguous record for each reached round",
        )
    for round_number, record in rounds_by_number.items():
        if round_number < current_round and (
            record["agent_a"] is None or record["agent_b"] is None
        ):
            raise malformed_artifact(path, "an earlier round is incomplete")
        if round_number < current_round and (
            record["agent_a"]["signal"] == "concede"
            or record["agent_b"]["signal"] == "concede"
            or (
                record["agent_a"]["signal"] == "tie"
                and record["agent_b"]["signal"] == "tie"
            )
        ):
            raise malformed_artifact(path, "an earlier round has a terminal signal")

    current = rounds_by_number.get(current_round)
    if next_step == "agent_b" and (
        current is None
        or current["agent_a"] is None
        or current["agent_b"] is not None
        or current["agent_a"]["signal"] == "concede"
    ):
        raise malformed_artifact(
            path,
            "next_step agent_b requires only Agent A's current-round turn",
        )

    termination = state["termination"]
    winner = state["winner"]
    if termination is not None and (
        not isinstance(termination, str) or termination not in TERMINATION_VALUES
    ):
        raise malformed_artifact(path, "termination is unsupported")
    if winner is not None and (
        not isinstance(winner, str) or winner not in WINNER_VALUES
    ):
        raise malformed_artifact(path, "winner is unsupported")
    validate_persisted_outcome(
        state,
        path,
        config=config,
        current=current,
        run_dir=run_dir,
    )

    last_error = state["last_error"]
    if last_error is not None and (
        not isinstance(last_error, str) or not last_error.strip()
    ):
        raise malformed_artifact(path, "last_error must be null or non-empty text")
    last_failure = optional_persisted_object(
        state["last_failure"], path, "last_failure"
    )
    retry = optional_persisted_object(state["retry"], path, "retry")
    active_task = optional_persisted_object(
        state["active_task"], path, "active_task"
    )
    task_attempts = require_persisted_object(
        state["task_attempts"], path, field="task_attempts"
    )
    for task_id, attempt in task_attempts.items():
        if not isinstance(task_id, str) or not task_id.strip():
            raise malformed_artifact(path, "task_attempts keys must be non-empty strings")
        require_persisted_positive_int(attempt, path, f"task_attempts.{task_id}")
    validate_historical_lifecycle_state(
        active_task=active_task,
        task_attempts=task_attempts,
        state=state,
        config=config,
        path=path,
        run_dir=run_dir,
    )

    if status in {"failed", "waiting_retry"}:
        if last_error is None or last_failure is None:
            raise malformed_artifact(
                path,
                f"status {status} requires last_error and last_failure",
            )
        validate_persisted_failure(
            last_failure,
            path,
            "last_failure",
            run_status=status,
            next_step=next_step,
            current_round=current_round,
            config=config,
            run_dir=run_dir,
        )
    if status == "waiting_retry":
        if retry is None:
            raise malformed_artifact(
                path,
                "status waiting_retry requires retryable retry metadata",
            )
        validate_persisted_failure(
            retry,
            path,
            "retry",
            run_status=status,
            next_step=next_step,
            current_round=current_round,
            config=config,
            run_dir=run_dir,
            require_retryable=True,
        )
        if retry != last_failure:
            raise malformed_artifact(path, "retry must match last_failure")
    elif last_failure is not None and last_failure.get("retryable") is True:
        raise malformed_artifact(path, "retryable failure requires status waiting_retry")
    elif retry is not None:
        raise malformed_artifact(path, "retry metadata requires status waiting_retry")
    if status in {"in_progress", "completed"} and (
        last_error is not None or last_failure is not None or retry is not None
    ):
        raise malformed_artifact(
            path,
            f"status {status} cannot retain failure or retry metadata",
        )
    if status == "completed" and active_task is not None:
        raise malformed_artifact(path, "a completed run cannot have an active_task")
    return state


def validate_persisted_rounds(
    value: Any,
    path: Path,
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    run_created_at: datetime,
    historical_remote: bool,
) -> dict[int, dict[str, Any]]:
    if not isinstance(value, list):
        raise malformed_artifact(path, "rounds must be an array")
    rounds_by_number: dict[int, dict[str, Any]] = {}
    for index, value_record in enumerate(value):
        field = f"rounds[{index}]"
        record = require_persisted_object(
            value_record,
            path,
            field=field,
            required_fields=("round_number", "agent_a", "agent_b"),
        )
        round_number = require_persisted_positive_int(
            record["round_number"], path, f"{field}.round_number"
        )
        if round_number != index + 1:
            raise malformed_artifact(
                path, "rounds must be stored in ascending contiguous order"
            )
        if round_number in rounds_by_number:
            raise malformed_artifact(path, f"duplicate round_number: {round_number}")
        record["agent_a"] = validate_persisted_turn(
            record["agent_a"],
            path,
            f"{field}.agent_a",
            expected_task_id=f"round-{round_number:02d}-agent-a",
            expected_role="agent_a",
            role_config=config["roles"]["agent_a"],
            run_dir=run_dir,
            run_created_at=run_created_at,
            historical_remote=historical_remote,
        )
        record["agent_b"] = validate_persisted_turn(
            record["agent_b"],
            path,
            f"{field}.agent_b",
            expected_task_id=f"round-{round_number:02d}-agent-b",
            expected_role="agent_b",
            role_config=config["roles"]["agent_b"],
            run_dir=run_dir,
            run_created_at=run_created_at,
            historical_remote=historical_remote,
        )
        if record["agent_b"] is not None and record["agent_a"] is None:
            raise malformed_artifact(path, f"{field}.agent_b cannot precede agent_a")
        rounds_by_number[round_number] = record
    return rounds_by_number


def validate_persisted_turn(
    value: Any,
    path: Path,
    field: str,
    *,
    expected_task_id: str,
    expected_role: str,
    role_config: Mapping[str, Any],
    run_dir: Path,
    run_created_at: datetime,
    historical_remote: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    turn = require_persisted_object(
        value,
        path,
        field=field,
        required_fields=(
            "response_md",
            "signal",
            "signal_reason",
            "task_id",
            "result_path",
        ),
    )
    require_persisted_text(turn["response_md"], path, f"{field}.response_md")
    signal_value = turn["signal"]
    if not isinstance(signal_value, str) or signal_value not in TURN_SIGNAL_VALUES:
        raise malformed_artifact(path, f"{field}.signal is unsupported")
    require_persisted_text(turn["signal_reason"], path, f"{field}.signal_reason")
    for key in ("citations", "files_created"):
        if key in turn and (
            not isinstance(turn[key], list)
            or any(not isinstance(item, str) for item in turn[key])
        ):
            raise malformed_artifact(path, f"{field}.{key} must be an array of strings")
    task_id = require_persisted_text(turn["task_id"], path, f"{field}.task_id")
    if task_id != expected_task_id:
        raise malformed_artifact(path, f"{field}.task_id does not match its round and role")
    result_path = require_persisted_text(
        turn["result_path"], path, f"{field}.result_path"
    )
    expected_result_path = f"results/{expected_task_id}.json"
    if result_path != expected_result_path:
        raise malformed_artifact(path, f"{field}.result_path is not canonical")
    require_run_artifact(run_dir, expected_result_path, kind="file")
    require_run_artifact(run_dir, f"prompts/{expected_task_id}.md", kind="file")
    validate_task_result_artifact(
        read_json(run_dir / expected_result_path),
        run_dir / expected_result_path,
        expected_task_id=expected_task_id,
        expected_role=expected_role,
        expected_provider=str(role_config["provider"]),
        expected_model=str(role_config["model"]),
        expected_statuses={"success"},
        expected_response={
            "response_md": turn["response_md"],
            "signal": turn["signal"],
            "signal_reason": turn["signal_reason"],
            "citations": turn.get("citations", []),
            "files_created": turn.get("files_created", []),
        },
        run_dir=run_dir,
        run_created_at=run_created_at,
        historical_remote=historical_remote,
    )
    return turn


def validate_persisted_outcome(
    state: Mapping[str, Any],
    path: Path,
    *,
    config: Mapping[str, Any],
    current: Mapping[str, Any] | None,
    run_dir: Path,
) -> None:
    status = state["status"]
    next_step = state["next_step"]
    termination = state["termination"]
    winner = state["winner"]
    final_result_value = state["final_result"]

    if (status == "completed") != (next_step == "completed"):
        raise malformed_artifact(
            path,
            "status completed and next_step completed must occur together",
        )
    final_phase = next_step in {"final_judgment", "completed"}
    if final_phase:
        if termination is None or current is None or current["agent_a"] is None:
            raise malformed_artifact(
                path,
                "the final phase requires an outcome and an Agent A turn",
            )
        validate_persisted_termination(
            termination,
            winner,
            current,
            path,
            current_round=int(state["current_round"]),
            max_rounds=int(config["max_rounds"]),
        )
    elif termination is not None or winner is not None:
        raise malformed_artifact(path, "an unfinished turn cannot have an outcome")

    if status == "completed":
        final_result = require_persisted_object(
            final_result_value,
            path,
            field="final_result",
            required_fields=(
                "winner",
                "reason",
                "termination",
                "decisive_round",
                "result_path",
            ),
        )
        final_winner = final_result["winner"]
        if not isinstance(final_winner, str) or final_winner not in WINNER_VALUES:
            raise malformed_artifact(path, "final_result.winner is unsupported")
        require_persisted_text(final_result["reason"], path, "final_result.reason")
        if final_result["termination"] != termination:
            raise malformed_artifact(
                path,
                "final_result.termination does not match termination",
            )
        decisive_round = require_persisted_positive_int(
            final_result["decisive_round"], path, "final_result.decisive_round"
        )
        if decisive_round != state["current_round"]:
            raise malformed_artifact(
                path,
                "final_result.decisive_round does not match current_round",
            )
        if termination != "round_limit" and final_winner != winner:
            raise malformed_artifact(path, "final_result.winner does not match winner")
        task_id = final_result.get("task_id", "final-judgment")
        if task_id != "final-judgment":
            raise malformed_artifact(path, "final_result.task_id must be final-judgment")
        result_path = require_persisted_text(
            final_result["result_path"], path, "final_result.result_path"
        )
        if result_path != "results/final-judgment.json":
            raise malformed_artifact(path, "final_result.result_path is not canonical")
        require_run_artifact(run_dir, result_path, kind="file")
        require_run_artifact(run_dir, "prompts/final-judgment.md", kind="file")
        validate_task_result_artifact(
            read_json(run_dir / result_path),
            run_dir / result_path,
            expected_task_id="final-judgment",
            expected_role="judge",
            expected_provider=str(config["roles"]["judge"]["provider"]),
            expected_model=str(config["roles"]["judge"]["model"]),
            expected_statuses={"success"},
            expected_response={
                "winner": final_result["winner"],
                "reason": final_result["reason"],
                "termination": final_result["termination"],
                "decisive_round": final_result["decisive_round"],
            },
            run_dir=run_dir,
            run_created_at=require_persisted_timestamp(
                config["created_at"], path, "run_config.created_at"
            ),
            historical_remote=config["access"] == "mcp.remote-agent-broker",
        )
    elif final_result_value is not None:
        raise malformed_artifact(path, "final_result requires status completed")


def validate_persisted_termination(
    termination: str,
    winner: Any,
    current: Mapping[str, Any],
    path: Path,
    *,
    current_round: int,
    max_rounds: int,
) -> None:
    agent_a = current["agent_a"]
    agent_b = current["agent_b"]
    if termination == "concession":
        valid = (
            winner == "agent_b"
            and agent_a["signal"] == "concede"
            and agent_b is None
        ) or (
            winner == "agent_a"
            and agent_b is not None
            and agent_a["signal"] != "concede"
            and agent_b["signal"] == "concede"
        )
        if not valid:
            raise malformed_artifact(path, "concession outcome is inconsistent")
        return
    if termination == "mutual_tie":
        if (
            winner != "tie"
            or agent_b is None
            or agent_a["signal"] != "tie"
            or agent_b["signal"] != "tie"
        ):
            raise malformed_artifact(path, "mutual_tie outcome is inconsistent")
        return
    if (
        termination != "round_limit"
        or winner is not None
        or current_round != max_rounds
        or agent_b is None
        or agent_a["signal"] == "concede"
        or agent_b["signal"] == "concede"
        or (agent_a["signal"] == "tie" and agent_b["signal"] == "tie")
    ):
        raise malformed_artifact(path, "round_limit outcome is inconsistent")


def validate_persisted_failure(
    failure: Mapping[str, Any],
    path: Path,
    field: str,
    *,
    run_status: str,
    next_step: str,
    current_round: int,
    config: Mapping[str, Any],
    run_dir: Path,
    require_retryable: bool = False,
) -> None:
    required = ("retryable", "reason", "message", "detected_at", "next_step")
    missing = sorted(key for key in required if key not in failure)
    if missing:
        raise malformed_artifact(
            path,
            f"{field} is missing required field(s): {', '.join(missing)}",
        )
    retryable = failure["retryable"]
    if not isinstance(retryable, bool):
        raise malformed_artifact(path, f"{field}.retryable must be a boolean")
    if require_retryable and not retryable:
        raise malformed_artifact(path, f"{field}.retryable must be true")
    reason = failure["reason"]
    if not isinstance(reason, str) or reason not in FAILURE_REASON_VALUES:
        raise malformed_artifact(path, f"{field}.reason is unsupported")
    require_persisted_text(failure["message"], path, f"{field}.message")
    detected_at = require_persisted_timestamp(
        failure["detected_at"], path, f"{field}.detected_at"
    )
    if failure["next_step"] != next_step:
        raise malformed_artifact(path, f"{field}.next_step does not match next_step")
    retry_reasons = {"rate_limit", "quota_exhausted", "token_exhausted"}
    if reason in retry_reasons and not retryable:
        raise malformed_artifact(path, f"{field}.reason requires retryable=true")
    if reason == "tool_failure":
        if retryable:
            raise malformed_artifact(path, f"{field}.tool_failure cannot be retryable")
        return
    if retryable and reason not in retry_reasons:
        raise malformed_artifact(path, f"{field}.reason is not retryable")
    for key in ("provider", "task_id", "role"):
        require_persisted_text(failure.get(key), path, f"{field}.{key}")
    expected_role = role_for_next_step(next_step)
    if failure["role"] != expected_role:
        raise malformed_artifact(path, f"{field}.role does not match next_step")
    if failure["provider"] != config["roles"][expected_role]["provider"]:
        raise malformed_artifact(path, f"{field}.provider does not match run_config")
    expected_task_id = task_id_for_step(next_step, current_round)
    if failure["task_id"] != expected_task_id:
        raise malformed_artifact(path, f"{field}.task_id does not match next_step")
    for key in ("stdout_path", "stderr_path"):
        artifact_reference = failure.get(key)
        if artifact_reference is None:
            continue
        reference = require_persisted_text(
            artifact_reference, path, f"{field}.{key}"
        )
        expected_reference = f"results/{expected_task_id}.{key.removesuffix('_path')}"
        if reference != expected_reference:
            raise malformed_artifact(path, f"{field}.{key} is not canonical")
        require_run_artifact(run_dir, reference, kind="file")
    cli_exit_code = failure.get("cli_exit_code")
    if cli_exit_code is not None and (
        isinstance(cli_exit_code, bool) or not isinstance(cli_exit_code, int)
    ):
        raise malformed_artifact(path, f"{field}.cli_exit_code must be an integer or null")
    result_path = f"results/{expected_task_id}.json"
    require_run_artifact(run_dir, result_path, kind="file")
    validate_task_result_artifact(
        read_json(run_dir / result_path),
        run_dir / result_path,
        expected_task_id=expected_task_id,
        expected_role=str(expected_role),
        expected_provider=str(config["roles"][expected_role]["provider"]),
        expected_model=str(config["roles"][expected_role]["model"]),
        expected_statuses={run_status},
        expected_failure=failure,
        run_dir=run_dir,
        run_created_at=require_persisted_timestamp(
            config["created_at"], path, "run_config.created_at"
        ),
        historical_remote=config["access"] == "mcp.remote-agent-broker",
    )
    if not retryable:
        return
    retry_after_at = require_persisted_timestamp(
        failure.get("retry_after_at"), path, f"{field}.retry_after_at"
    )
    retry_after_sec = require_persisted_positive_int(
        failure.get("retry_after_sec"), path, f"{field}.retry_after_sec"
    )
    if retry_after_at <= detected_at:
        raise malformed_artifact(path, f"{field}.retry_after_at must follow detected_at")
    expected_retry_at = detected_at + timedelta(seconds=retry_after_sec)
    if retry_after_at != expected_retry_at:
        raise malformed_artifact(
            path,
            f"{field}.retry_after_at must equal detected_at plus retry_after_sec",
        )


def task_id_for_step(next_step: str, current_round: int) -> str:
    if next_step == "agent_a":
        return f"round-{current_round:02d}-agent-a"
    if next_step == "agent_b":
        return f"round-{current_round:02d}-agent-b"
    if next_step == "final_judgment":
        return "final-judgment"
    raise DebateToolError(f"Cannot derive task id for next_step={next_step}")


def validate_historical_lifecycle_state(
    *,
    active_task: Mapping[str, Any] | None,
    task_attempts: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    path: Path,
    run_dir: Path,
) -> None:
    historical_remote = config["access"] == "mcp.remote-agent-broker"
    if not historical_remote:
        if active_task is not None:
            raise malformed_artifact(
                path,
                "active_task is allowed only for historical remote-broker runs",
            )
        if task_attempts:
            raise malformed_artifact(
                path,
                "task_attempts are allowed only for historical remote-broker runs",
            )
        return

    allowed_task_ids: set[str] = set()
    for round_record in state["rounds"]:
        for role in ("agent_a", "agent_b"):
            turn = round_record.get(role)
            if isinstance(turn, dict) and isinstance(turn.get("task_id"), str):
                allowed_task_ids.add(turn["task_id"])
    final_result = state.get("final_result")
    if isinstance(final_result, dict):
        allowed_task_ids.add(str(final_result.get("task_id", "final-judgment")))
    next_step = str(state["next_step"])
    current_round = int(state["current_round"])
    if next_step != "completed":
        allowed_task_ids.add(task_id_for_step(next_step, current_round))

    unexpected_attempts = sorted(set(task_attempts) - allowed_task_ids)
    if unexpected_attempts:
        raise malformed_artifact(
            path,
            "task_attempts contains task ids outside the reached lifecycle: "
            + ", ".join(unexpected_attempts),
        )
    if active_task is None:
        return

    required_fields = (
        "task_id",
        "role",
        "phase",
        "next_step",
        "round_number",
        "attempt",
        "backend",
        "access",
        "provider",
        "model",
        "effort",
        "job_id",
        "status",
        "terminal",
        "finalized",
        "started_at",
        "updated_at",
        "prompt_path",
        "stdout_path",
        "stderr_path",
        "expected_keys",
        "broker_exit_code",
        "broker_last_error",
    )
    missing = sorted(key for key in required_fields if key not in active_task)
    if missing:
        raise malformed_artifact(
            path,
            f"active_task is missing required field(s): {', '.join(missing)}",
        )

    expected_role = role_for_next_step(next_step)
    if expected_role is None:
        raise malformed_artifact(path, "active_task requires an executable next_step")
    expected_task_id = task_id_for_step(next_step, current_round)
    if active_task["task_id"] != expected_task_id:
        raise malformed_artifact(path, "active_task.task_id does not match next_step")
    if active_task["role"] != expected_role:
        raise malformed_artifact(path, "active_task.role does not match next_step")
    if active_task["next_step"] != next_step:
        raise malformed_artifact(path, "active_task.next_step does not match state")
    if active_task["phase"] != (
        "final_judgment" if expected_role == "judge" else "turn"
    ):
        raise malformed_artifact(path, "active_task.phase does not match role")
    if active_task["round_number"] != current_round:
        raise malformed_artifact(path, "active_task.round_number does not match state")
    attempt = require_persisted_positive_int(
        active_task["attempt"], path, "active_task.attempt"
    )
    if task_attempts.get(expected_task_id) != attempt:
        raise malformed_artifact(path, "active_task.attempt does not match task_attempts")
    for field in ("backend", "access"):
        if active_task[field] != "mcp.remote-agent-broker":
            raise malformed_artifact(
                path,
                f"active_task.{field} must be mcp.remote-agent-broker",
            )

    role_config = config["roles"][expected_role]
    for field in ("provider", "model", "effort"):
        value = require_persisted_text(
            active_task[field], path, f"active_task.{field}"
        )
        if value != role_config[field]:
            raise malformed_artifact(
                path,
                f"active_task.{field} does not match run_config",
            )
    job_id = require_persisted_text(active_task["job_id"], path, "active_task.job_id")
    if not re.fullmatch(r"job_[0-9]{8}T[0-9]{6}Z_[a-f0-9]{12}", job_id):
        raise malformed_artifact(path, "active_task.job_id is not canonical")

    broker_statuses = {
        "queued",
        "running",
        "waiting_input",
        "completed",
        "failed",
        "cancelled",
        "lost",
        "expired",
    }
    broker_status = active_task["status"]
    if not isinstance(broker_status, str) or broker_status not in broker_statuses:
        raise malformed_artifact(path, "active_task.status is unsupported")
    terminal_statuses = {"completed", "failed", "cancelled", "lost", "expired"}
    if not isinstance(active_task["terminal"], bool) or active_task["terminal"] != (
        broker_status in terminal_statuses
    ):
        raise malformed_artifact(path, "active_task.terminal does not match status")
    if active_task["finalized"] is not False:
        raise malformed_artifact(path, "active_task.finalized must be false")

    started_at = require_persisted_timestamp(
        active_task["started_at"], path, "active_task.started_at"
    )
    updated_at = require_persisted_timestamp(
        active_task["updated_at"], path, "active_task.updated_at"
    )
    if updated_at < started_at:
        raise malformed_artifact(path, "active_task.updated_at cannot precede started_at")
    broker_completed_at = active_task.get("broker_completed_at")
    if broker_completed_at is not None and require_persisted_timestamp(
        broker_completed_at, path, "active_task.broker_completed_at"
    ) < started_at:
        raise malformed_artifact(
            path,
            "active_task.broker_completed_at cannot precede started_at",
        )

    expected_references = {
        "prompt_path": f"prompts/{expected_task_id}.md",
        "stdout_path": f"results/{expected_task_id}.stdout",
        "stderr_path": f"results/{expected_task_id}.stderr",
    }
    for field, expected_reference in expected_references.items():
        reference = require_persisted_text(
            active_task[field], path, f"active_task.{field}"
        )
        if reference != expected_reference:
            raise malformed_artifact(path, f"active_task.{field} is not canonical")
    require_run_artifact(run_dir, expected_references["prompt_path"], kind="file")

    expected_keys = (
        ["winner", "reason", "termination", "decisive_round"]
        if expected_role == "judge"
        else ["response_md", "signal", "signal_reason"]
    )
    if active_task["expected_keys"] != expected_keys:
        raise malformed_artifact(path, "active_task.expected_keys does not match role")
    broker_exit_code = active_task["broker_exit_code"]
    if broker_exit_code is not None and (
        isinstance(broker_exit_code, bool) or not isinstance(broker_exit_code, int)
    ):
        raise malformed_artifact(
            path,
            "active_task.broker_exit_code must be an integer or null",
        )
    broker_last_error = active_task["broker_last_error"]
    if broker_last_error is not None:
        require_persisted_text(
            broker_last_error, path, "active_task.broker_last_error"
        )


def validate_task_result_artifact(
    value: Any,
    path: Path,
    *,
    expected_task_id: str,
    expected_role: str,
    expected_provider: str,
    expected_model: str,
    expected_statuses: set[str],
    run_dir: Path,
    run_created_at: datetime,
    historical_remote: bool,
    expected_response: Mapping[str, Any] | None = None,
    expected_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = require_persisted_object(
        value,
        path,
        required_fields=(
            "id",
            "role",
            "provider",
            "status",
            "stdout_path",
            "stderr_path",
            "cli_exit_code",
            "finished_at",
        ),
    )
    if payload["id"] != expected_task_id:
        raise malformed_artifact(path, "id does not match the referenced task")
    if payload["role"] != expected_role:
        raise malformed_artifact(path, "role does not match the referenced task")
    provider = require_persisted_text(payload["provider"], path, "provider")
    if provider != expected_provider:
        raise malformed_artifact(path, "provider does not match run_config")
    status = payload["status"]
    if not isinstance(status, str) or status not in expected_statuses:
        allowed = ", ".join(sorted(expected_statuses))
        raise malformed_artifact(path, f"status must be one of: {allowed}")
    finished_at = require_persisted_timestamp(
        payload["finished_at"], path, "finished_at"
    )
    if finished_at < run_created_at:
        raise malformed_artifact(path, "finished_at cannot precede run creation")

    for key in ("stdout_path", "stderr_path"):
        reference_value = payload[key]
        if reference_value is None:
            reference = None
            if status == "success":
                raise malformed_artifact(path, f"successful task requires {key}")
        else:
            reference = require_persisted_text(reference_value, path, key)
            expected_reference = (
                f"results/{expected_task_id}.{key.removesuffix('_path')}"
            )
            if reference != expected_reference:
                raise malformed_artifact(path, f"{key} is not canonical")
            require_run_artifact(run_dir, reference, kind="file")
        if expected_failure is not None and reference != expected_failure.get(key):
            raise malformed_artifact(path, f"{key} does not match failure metadata")

    cli_exit_code = payload["cli_exit_code"]
    if cli_exit_code is not None and (
        isinstance(cli_exit_code, bool) or not isinstance(cli_exit_code, int)
    ):
        raise malformed_artifact(path, "cli_exit_code must be an integer or null")

    if status == "success":
        for field in ("model", "response_json"):
            if field not in payload:
                raise malformed_artifact(path, f"missing required field: {field}")
        model = require_persisted_text(payload["model"], path, "model")
        if model != expected_model:
            raise malformed_artifact(path, "model does not match run_config")
        response = normalize_persisted_task_response(
            payload["response_json"],
            path,
            role=expected_role,
            allow_historical_objects=historical_remote,
        )
        normalized_expected = (
            normalize_persisted_task_response(
                expected_response,
                path,
                role=expected_role,
                allow_historical_objects=historical_remote,
            )
            if expected_response is not None
            else None
        )
        if normalized_expected is None or response != normalized_expected:
            raise malformed_artifact(path, "response_json does not match state")
        if cli_exit_code != 0:
            raise malformed_artifact(path, "successful task cli_exit_code must be zero")
        if payload.get("failure") is not None:
            raise malformed_artifact(path, "successful task cannot contain failure metadata")
        schema_path = run_dir / "results" / f"{expected_task_id}.schema.json"
        if not historical_remote or schema_path.exists() or schema_path.is_symlink():
            require_run_artifact(
                run_dir,
                f"results/{expected_task_id}.schema.json",
                kind="file",
            )
            expected_keys = (
                ("winner", "reason", "termination", "decisive_round")
                if expected_role == "judge"
                else ("response_md", "signal", "signal_reason")
            )
            if read_json(schema_path) != schema_for_expected_keys(expected_keys):
                raise malformed_artifact(path, "task schema does not match its role")
        return payload

    if "model" not in payload:
        # Failure artifacts written before model binding was introduced inherit
        # the authoritative role model from run_config during read validation.
        payload["model"] = expected_model
    else:
        model_value = payload["model"]
        model = require_persisted_text(model_value, path, "model")
        if model != expected_model:
            raise malformed_artifact(path, "model does not match run_config")
    failure = require_persisted_object(
        payload.get("failure"), path, field="failure"
    )
    if expected_failure is None or failure != dict(expected_failure):
        raise malformed_artifact(path, "failure does not match state")
    if finished_at != require_persisted_timestamp(
        expected_failure.get("detected_at"), path, "failure.detected_at"
    ):
        raise malformed_artifact(path, "finished_at does not match failure.detected_at")
    if cli_exit_code != expected_failure.get("cli_exit_code"):
        raise malformed_artifact(path, "cli_exit_code does not match failure metadata")
    if payload.get("response_json") is not None:
        raise malformed_artifact(path, "failed task cannot contain response_json")
    return payload


def normalize_persisted_task_response(
    value: Any,
    path: Path,
    *,
    role: str,
    allow_historical_objects: bool,
) -> dict[str, Any]:
    response = require_persisted_object(value, path, field="response_json")
    if role in {"agent_a", "agent_b"}:
        missing = sorted(
            key
            for key in ("response_md", "signal", "signal_reason")
            if key not in response
        )
        if missing:
            raise malformed_artifact(
                path,
                f"response_json is missing required field(s): {', '.join(missing)}",
            )
        signal = require_persisted_text(response["signal"], path, "response_json.signal")
        if signal not in TURN_SIGNAL_VALUES:
            raise malformed_artifact(path, "response_json.signal is unsupported")
        return {
            "response_md": require_persisted_text(
                response["response_md"], path, "response_json.response_md"
            ),
            "signal": signal,
            "signal_reason": require_persisted_text(
                response["signal_reason"], path, "response_json.signal_reason"
            ),
            "citations": normalize_historical_string_list(
                response.get("citations", []),
                path,
                "response_json.citations",
                allow_objects=allow_historical_objects,
            ),
            "files_created": normalize_historical_string_list(
                response.get("files_created", []),
                path,
                "response_json.files_created",
                allow_objects=False,
            ),
        }

    missing = sorted(
        key
        for key in ("winner", "reason", "termination", "decisive_round")
        if key not in response
    )
    if missing:
        raise malformed_artifact(
            path,
            f"response_json is missing required field(s): {', '.join(missing)}",
        )
    winner = require_persisted_text(response["winner"], path, "response_json.winner")
    termination = require_persisted_text(
        response["termination"], path, "response_json.termination"
    )
    if winner not in WINNER_VALUES:
        raise malformed_artifact(path, "response_json.winner is unsupported")
    if termination not in TERMINATION_VALUES:
        raise malformed_artifact(path, "response_json.termination is unsupported")
    return {
        "winner": winner,
        "reason": require_persisted_text(
            response["reason"], path, "response_json.reason"
        ),
        "termination": termination,
        "decisive_round": require_persisted_positive_int(
            response["decisive_round"], path, "response_json.decisive_round"
        ),
    }


def normalize_historical_string_list(
    value: Any,
    path: Path,
    field: str,
    *,
    allow_objects: bool,
) -> list[str]:
    if not isinstance(value, list):
        raise malformed_artifact(path, f"{field} must be an array")
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized_item = item.strip()
        elif allow_objects and isinstance(item, dict):
            # Historical remote-broker runs persisted provider citation objects
            # in task results while their state records stored the same objects'
            # string representation.  Preserve read-only compatibility while
            # still binding each result item to the corresponding state value.
            normalized_item = str(item).strip()
        else:
            raise malformed_artifact(
                path,
                f"{field} must contain strings or historical citation objects",
            )
        if normalized_item:
            normalized.append(normalized_item)
    return normalized


def optional_persisted_object(
    value: Any,
    path: Path,
    field: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return require_persisted_object(value, path, field=field)


def require_persisted_object(
    value: Any,
    path: Path,
    *,
    field: str | None = None,
    required_fields: Iterable[str] = (),
) -> dict[str, Any]:
    label = field or path.name
    if not isinstance(value, dict):
        raise malformed_artifact(path, f"{label} must be an object")
    missing = sorted(key for key in required_fields if key not in value)
    if missing:
        raise malformed_artifact(
            path,
            f"{label} is missing required field(s): {', '.join(missing)}",
        )
    return value


def require_persisted_text(value: Any, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise malformed_artifact(path, f"{field} must be a non-empty string")
    return value.strip()


def require_persisted_timestamp(value: Any, path: Path, field: str) -> datetime:
    text = require_persisted_text(value, path, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise malformed_artifact(path, f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise malformed_artifact(path, f"{field} must include a timezone")
    return parsed


def require_persisted_positive_int(value: Any, path: Path, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise malformed_artifact(path, f"{field} must be a positive integer")
    return value


def malformed_artifact(path: Path, detail: str) -> DebateToolError:
    return DebateToolError(f"Malformed {path.name} in {path.parent}: {detail}")


def relative_to(base: Path, target: Path | None) -> str | None:
    if target is None:
        return None
    resolved_base = base.resolve()
    resolved_target = target.resolve()
    if not is_relative_to(resolved_target, resolved_base):
        raise DebateToolError(
            f"Debate artifact escapes run boundary {resolved_base}: {target}"
        )
    return resolved_target.relative_to(resolved_base).as_posix()


def debate_log_task_marker(task_id: str) -> str:
    return f"<!-- ai-debate-task:{task_id} -->"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def run_options_from_mapping(arguments: Mapping[str, Any]) -> RunOptions:
    settings = ai_debate_settings()
    access = setting_or_argument(arguments, "access", settings, "access")
    side_b_provider = setting_or_argument(
        arguments,
        "side_b_provider",
        settings,
        "peer_b",
    )
    return RunOptions(
        topic=str(arguments.get("topic", "")).strip(),
        side_a=str(arguments.get("side_a", "")).strip(),
        side_b=str(arguments.get("side_b", "")).strip(),
        side_b_provider=side_b_provider,
        access=access,
        max_rounds=int_argument(arguments, "max_rounds", DEFAULT_MAX_ROUNDS),
        model=str(arguments["model"]).strip() if arguments.get("model") is not None else None,
        effort=str(arguments["effort"]).strip() if arguments.get("effort") is not None else None,
        agent_a_model=(
            str(arguments["agent_a_model"]).strip()
            if arguments.get("agent_a_model") is not None
            else None
        ),
        agent_b_model=(
            str(arguments["agent_b_model"]).strip()
            if arguments.get("agent_b_model") is not None
            else None
        ),
        agent_b_effort=(
            str(arguments["agent_b_effort"]).strip()
            if arguments.get("agent_b_effort") is not None
            else None
        ),
        judge_model=(
            str(arguments["judge_model"]).strip()
            if arguments.get("judge_model") is not None
            else None
        ),
        turn_idle_timeout_sec=optional_int_argument(arguments, "turn_idle_timeout_sec"),
        turn_timeout_sec=optional_int_argument(arguments, "turn_timeout_sec"),
        output_root=(
            Path(str(arguments["output_root"]))
            if arguments.get("output_root") is not None
            else None
        ),
    )


def ai_debate_settings() -> dict[str, Any]:
    try:
        return load_settings_section("ai_debate", required=True)
    except TokimonSettingsError as exc:
        raise DebateToolError(str(exc), exit_code=2) from exc


def setting_or_argument(
    arguments: Mapping[str, Any],
    argument_key: str,
    settings: Mapping[str, Any],
    settings_key: str,
) -> str:
    if arguments.get(argument_key) is not None:
        return str(arguments[argument_key]).strip()
    if settings.get(settings_key) is not None:
        return str(settings[settings_key]).strip()
    raise DebateToolError(f"ai_debate.{settings_key} is required", exit_code=2)


def required_setting(settings: Mapping[str, Any], key: str, *, context: str) -> str:
    value = settings.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DebateToolError(f"{context}.{key} is required", exit_code=2)
    return value.strip()


def resolve_run_configuration(options: RunOptions) -> dict[str, Any]:
    settings = ai_debate_settings()
    access = options.access or required_setting(settings, "access", context="ai_debate")
    side_b_provider = options.side_b_provider or required_setting(
        settings,
        "peer_b",
        context="ai_debate",
    )
    if access not in SUPPORTED_ACCESS:
        raise DebateToolError("access must be local-cli", exit_code=2)
    if side_b_provider not in SUPPORTED_SIDE_B_PROVIDERS:
        raise DebateToolError("side_b_provider must be claude or codex", exit_code=2)

    agent_a_options = resolve_provider_options(
        "codex", model=options.agent_a_model or options.model, effort=options.effort,
    )
    agent_b_options = resolve_provider_options(
        side_b_provider,
        model=options.agent_b_model or (options.model if side_b_provider == "codex" else None),
        effort=options.agent_b_effort or (options.effort if side_b_provider == "codex" else None),
    )
    judge_options = resolve_provider_options(
        "codex", model=options.judge_model or options.model, effort=options.effort,
    )
    execution_overrides: dict[str, Any] = {}
    if options.turn_timeout_sec is not None:
        execution_overrides["idle_timeout_sec"] = options.turn_timeout_sec
    if options.turn_idle_timeout_sec is not None:
        execution_overrides["idle_timeout_sec"] = options.turn_idle_timeout_sec
    try:
        agent_execution = resolve_agent_execution_options(
            execution_overrides,
            workspace_root=WORKSPACE_ROOT,
        )
    except TokimonSettingsError as exc:
        raise DebateToolError(str(exc), exit_code=2) from exc
    poll_interval = float(agent_execution["poll_interval_sec"])
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise DebateToolError(
            "agent_execution.poll_interval_sec must be finite and greater than zero",
            exit_code=2,
        )
    return {
        "access": access,
        "side_b_provider": side_b_provider,
        "agent_execution": agent_execution,
        "roles": {
            "agent_a": {"provider": "codex", **agent_a_options},
            "agent_b": {"provider": side_b_provider, **agent_b_options},
            "judge": {"provider": "codex", **judge_options},
        },
    }


def resolve_provider_options(
    provider: str,
    *,
    model: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if effort:
        overrides["effort"] = effort
    try:
        resolved = resolve_agent_options(provider, overrides, workspace_root=WORKSPACE_ROOT)
        resolved.setdefault("model", "default")
        resolved.setdefault("effort", "default")
        return resolved
    except TokimonSettingsError as exc:
        raise DebateToolError(str(exc), exit_code=2) from exc


def int_argument(arguments: Mapping[str, Any], name: str, default: int) -> int:
    try:
        value = int(arguments.get(name, default))
    except (TypeError, ValueError) as exc:
        raise DebateToolError(f"{name} must be an integer") from exc
    if value <= 0:
        raise DebateToolError(f"{name} must be greater than zero")
    return value


def optional_int_argument(arguments: Mapping[str, Any], name: str) -> int | None:
    if arguments.get(name) is None:
        return None
    try:
        value = int(arguments[name])
    except (TypeError, ValueError) as exc:
        raise DebateToolError(f"{name} must be an integer") from exc
    if value <= 0:
        raise DebateToolError(f"{name} must be greater than zero")
    return value


def run_options_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["topic", "side_a", "side_b"],
        "properties": {
            "topic": {"type": "string"},
            "side_a": {"type": "string"},
            "side_b": {"type": "string"},
            "side_b_provider": {
                "type": "string",
                "enum": ["claude", "codex"],
            },
            "access": {
                "type": "string",
                "enum": ["local-cli"],
            },
            "max_rounds": {"type": "integer", "minimum": 1},
            "model": {
                "type": "string",
                "description": "Read the active main-thread model from session metadata and pass it for all roles.",
            },
            "effort": {
                "type": "string",
                "description": "Read the active main-thread effort from session metadata and pass it for all roles.",
            },
            "agent_a_model": {"type": "string"},
            "agent_b_model": {"type": "string"},
            "agent_b_effort": {"type": "string"},
            "judge_model": {"type": "string"},
            "turn_idle_timeout_sec": {"type": "integer", "minimum": 1},
            "turn_timeout_sec": {
                "type": "integer",
                "minimum": 1,
                "description": "Compatibility alias for turn_idle_timeout_sec.",
            },
            "output_root": {"type": "string"},
        },
    }


def mcp_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "ai_debate_health",
            "description": "Report AI debate MCP server health and provider CLI discovery.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
        {
            "name": "ai_debate_run",
            "description": "Run a fresh structured AI debate.",
            "inputSchema": run_options_schema(),
        },
        {
            "name": "ai_debate_start",
            "description": "Create a fresh debate run and return its run_dir without executing turns.",
            "inputSchema": run_options_schema(),
        },
        {
            "name": "ai_debate_resume",
            "description": "Resume an incomplete debate run.",
            "inputSchema": run_dir_schema(),
        },
        {
            "name": "ai_debate_resume_start",
            "description": "Validate an existing debate run_dir and return its summary without executing turns.",
            "inputSchema": run_dir_schema(),
        },
        {
            "name": "ai_debate_turn_start",
            "description": "Local-only compatibility endpoint retained for old clients; always fails fast.",
            "inputSchema": run_dir_schema(),
        },
        {
            "name": "ai_debate_turn_status",
            "description": "Local-only compatibility endpoint retained for old clients; always fails fast.",
            "inputSchema": run_dir_schema(),
        },
        {
            "name": "ai_debate_turn_finalize",
            "description": "Local-only compatibility endpoint retained for old clients; always fails fast.",
            "inputSchema": run_dir_schema(),
        },
        {
            "name": "ai_debate_status",
            "description": "Read state for a debate run.",
            "inputSchema": run_dir_schema(),
        },
        {
            "name": "ai_debate_read_log",
            "description": "Read the debate transcript.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["run_dir"],
                "properties": {
                    "run_dir": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1},
                },
            },
        },
    ]


def run_dir_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["run_dir"],
        "properties": {
            "run_dir": {"type": "string"},
        },
    }


class MCPServer:
    def __init__(self, orchestrator: DebateOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or DebateOrchestrator()

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        return handle_request(
            request, server_name=SERVER_NAME, server_version=SERVER_VERSION,
            protocol_version=SUPPORTED_PROTOCOL_VERSION,
            list_tools=mcp_tools, call_tool=self._call_tool,
        )


    def _call_tool(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name_value = params.get("name")
        if not isinstance(name_value, str):
            return tool_result({"error": "tool name must be a string"}, is_error=True)
        name = name_value
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return tool_result({"error": "arguments must be an object"}, is_error=True)
        try:
            schema = next(
                (
                    tool["inputSchema"]
                    for tool in mcp_tools()
                    if tool["name"] == name
                ),
                None,
            )
            if schema is None:
                return tool_result({"error": f"Unknown tool: {name}"}, is_error=True)
            validate_mcp_arguments(arguments, schema, path="arguments")
            if name == "ai_debate_health":
                return tool_result(self.orchestrator.health())
            if name == "ai_debate_run":
                return tool_result(self.orchestrator.run_new(run_options_from_mapping(arguments)))
            if name == "ai_debate_start":
                return tool_result(self.orchestrator.start_new(run_options_from_mapping(arguments)))
            if name == "ai_debate_resume":
                return tool_result(self.orchestrator.resume(required_run_dir(arguments)))
            if name == "ai_debate_resume_start":
                return tool_result(self.orchestrator.resume_start(required_run_dir(arguments)))
            if name == "ai_debate_turn_start":
                return tool_result(self.orchestrator.turn_start(required_run_dir(arguments)))
            if name == "ai_debate_turn_status":
                return tool_result(self.orchestrator.turn_status(required_run_dir(arguments)))
            if name == "ai_debate_turn_finalize":
                return tool_result(self.orchestrator.turn_finalize(required_run_dir(arguments)))
            if name == "ai_debate_status":
                return tool_result(self.orchestrator.status(required_run_dir(arguments)))
            if name == "ai_debate_read_log":
                max_chars = arguments.get("max_chars")
                return tool_result(
                    self.orchestrator.read_log(
                        required_run_dir(arguments),
                        max_chars=(
                            int_argument(arguments, "max_chars", 1)
                            if max_chars is not None
                            else None
                        ),
                    )
                )
            raise DebateToolError(f"Tool dispatch is missing for declared tool: {name}")
        except DebateRunFailedError as exc:
            return tool_result(
                {
                    "error": str(exc),
                    "run_dir": exc.summary.get("run_dir"),
                    "run": exc.summary,
                },
                is_error=True,
            )
        except DebateToolError as exc:
            return tool_result({"error": str(exc)}, is_error=True)


def validate_mcp_arguments(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
) -> None:
    """Validate the JSON Schema subset used by AI Debate tool declarations."""

    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise DebateToolError(f"{path} must be an object")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise DebateToolError(f"Invalid tool schema at {path}")
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise DebateToolError(f"Invalid tool schema at {path}")
        missing = [name for name in required if name not in value]
        if missing:
            raise DebateToolError(f"{path}.{missing[0]} is required")
        extras = sorted(set(value) - set(properties))
        if extras and schema.get("additionalProperties", True) is False:
            raise DebateToolError(
                f"{path} contains unsupported field(s): {', '.join(extras)}"
            )
        for name, item in value.items():
            property_schema = properties.get(name)
            if isinstance(property_schema, Mapping):
                validate_mcp_arguments(item, property_schema, path=f"{path}.{name}")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise DebateToolError(f"{path} must be a string")
    elif expected_type == "integer":
        if type(value) is not int:
            raise DebateToolError(f"{path} must be an integer")

    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        choices = ", ".join(str(item) for item in allowed)
        raise DebateToolError(f"{path} must be one of: {choices}")
    minimum = schema.get("minimum")
    if type(value) is int and type(minimum) is int and value < minimum:
        raise DebateToolError(f"{path} must be greater than or equal to {minimum}")


def required_run_dir(arguments: Mapping[str, Any]) -> str:
    run_dir = str(arguments.get("run_dir", "")).strip()
    if not run_dir:
        raise DebateToolError("run_dir is required")
    return run_dir


def serve(
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    server: MCPServer | None = None,
) -> int:
    return serve_stdio(
        input_stream=input_stream, output_stream=output_stream,
        server=server or MCPServer(), logger=configure_runtime_logger(), log_event=log_runtime_event,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AI Debate MCP server or direct CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Run the stdio MCP server")

    health_parser = subparsers.add_parser("health", help="Show provider and server health")
    health_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run a fresh debate")
    add_run_arguments(run_parser)
    run_parser.add_argument("--json", action="store_true")

    resume_parser = subparsers.add_parser("resume", help="Resume an incomplete debate")
    resume_parser.add_argument("run_dir")
    resume_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser("status", help="Show debate run status")
    status_parser.add_argument("run_dir")
    status_parser.add_argument("--json", action="store_true")

    turn_start_parser = subparsers.add_parser(
        "turn-start",
        help="Compatibility endpoint that fails fast in local-only mode",
    )
    turn_start_parser.add_argument("run_dir")
    turn_start_parser.add_argument("--json", action="store_true")

    turn_status_parser = subparsers.add_parser(
        "turn-status",
        help="Compatibility endpoint that fails fast in local-only mode",
    )
    turn_status_parser.add_argument("run_dir")
    turn_status_parser.add_argument("--json", action="store_true")

    turn_finalize_parser = subparsers.add_parser(
        "turn-finalize",
        help="Compatibility endpoint that fails fast in local-only mode",
    )
    turn_finalize_parser.add_argument("run_dir")
    turn_finalize_parser.add_argument("--json", action="store_true")

    log_parser = subparsers.add_parser("read-log", help="Read debate_log.md")
    log_parser.add_argument("run_dir")
    log_parser.add_argument("--max-chars", type=positive_int)
    log_parser.add_argument("--json", action="store_true")

    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", required=True)
    parser.add_argument("--side-a", required=True)
    parser.add_argument("--side-b", required=True)
    parser.add_argument("--side-b-provider", choices=("claude", "codex"))
    parser.add_argument("--access", choices=("local-cli",))
    parser.add_argument("--max-rounds", type=positive_int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--model", help="Active main-thread model from session metadata")
    parser.add_argument("--effort", help="Active main-thread effort from session metadata")
    parser.add_argument("--agent-a-model")
    parser.add_argument("--agent-b-model")
    parser.add_argument("--agent-b-effort")
    parser.add_argument("--judge-model")
    parser.add_argument("--turn-idle-timeout-sec", type=positive_int)
    parser.add_argument("--turn-timeout-sec", type=positive_int)
    parser.add_argument("--output-root", type=Path)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def options_from_args(args: argparse.Namespace) -> RunOptions:
    settings = ai_debate_settings()
    return RunOptions(
        topic=args.topic,
        side_a=args.side_a,
        side_b=args.side_b,
        side_b_provider=args.side_b_provider
        or required_setting(settings, "peer_b", context="ai_debate"),
        access=args.access or required_setting(settings, "access", context="ai_debate"),
        max_rounds=args.max_rounds,
        model=args.model,
        effort=args.effort,
        agent_a_model=args.agent_a_model,
        agent_b_model=args.agent_b_model,
        agent_b_effort=args.agent_b_effort,
        judge_model=args.judge_model,
        turn_idle_timeout_sec=args.turn_idle_timeout_sec,
        turn_timeout_sec=args.turn_timeout_sec,
        output_root=args.output_root,
    )


def print_payload(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json_text(payload))
    else:
        if isinstance(payload, dict) and payload.get("text"):
            print(payload["text"])
        else:
            print(json_text(payload))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrator = DebateOrchestrator()

    try:
        if args.command == "serve":
            return serve()
        if args.command == "health":
            payload = orchestrator.health()
            print_payload(payload, as_json=args.json)
            return 0 if payload.get("ok") else 1
        if args.command == "run":
            print_payload(orchestrator.run_new(options_from_args(args)), as_json=args.json)
            return 0
        if args.command == "resume":
            print_payload(orchestrator.resume(args.run_dir), as_json=args.json)
            return 0
        if args.command == "status":
            print_payload(orchestrator.status(args.run_dir), as_json=args.json)
            return 0
        if args.command == "turn-start":
            print_payload(orchestrator.turn_start(args.run_dir), as_json=args.json)
            return 0
        if args.command == "turn-status":
            print_payload(orchestrator.turn_status(args.run_dir), as_json=args.json)
            return 0
        if args.command == "turn-finalize":
            print_payload(orchestrator.turn_finalize(args.run_dir), as_json=args.json)
            return 0
        if args.command == "read-log":
            print_payload(
                orchestrator.read_log(args.run_dir, max_chars=args.max_chars),
                as_json=args.json,
            )
            return 0
    except DebateRunFailedError as exc:
        if getattr(args, "json", False):
            print_payload(
                {
                    "error": str(exc),
                    "run_dir": exc.summary.get("run_dir"),
                    "run": exc.summary,
                },
                as_json=True,
            )
        else:
            print(str(exc), file=sys.stderr)
            run_dir = exc.summary.get("run_dir") if isinstance(exc.summary, dict) else None
            if run_dir:
                print(f"Run directory: {run_dir}", file=sys.stderr)
        return exc.exit_code
    except DebateToolError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
