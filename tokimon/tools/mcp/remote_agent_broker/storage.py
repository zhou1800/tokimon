"""Persisted job state, artifacts, and cleanup collaborators."""

from __future__ import annotations

import codecs
import fcntl
import json
import os
import secrets
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Mapping

from tokimon.tools.mcp.remote_agent_broker.model import (
    JOB_ID_RE,
    JOB_STATUSES,
    TERMINAL_STATUSES,
    RemoteAgentBrokerError,
    RequestContext,
    cleanup_boundary_failure,
    is_relative_to,
    parse_utc,
    redact_text,
    require_positive_int,
    safe_options,
    summarize_prompt,
    timestamp_for_id,
    utc_now,
    validate_job_id,
)


JobUpdater = Callable[[dict[str, Any]], None]
DiffCapture = Callable[[Path, str | None], dict[str, object]]
RuntimeLookup = Callable[[str], object | None]


class RuntimeFileLocks:
    """Provide reentrant thread locks backed by process-visible lock files."""

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.locks_root = runtime_root / "locks"
        self._lock = threading.RLock()
        self._thread_locks: dict[str, threading.RLock] = {}
        self._thread_state = threading.local()

    @contextmanager
    def acquire(self, name: str, *, blocking: bool = True) -> Iterator[None]:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        if not name or any(character not in allowed for character in name):
            raise RemoteAgentBrokerError("Invalid runtime lock name")
        with self._lock:
            thread_lock = self._thread_locks.setdefault(name, threading.RLock())
        with thread_lock:
            depths = getattr(self._thread_state, "depths", None)
            if depths is None:
                depths = {}
                self._thread_state.depths = depths
            depth = int(depths.get(name, 0))
            if depth:
                depths[name] = depth + 1
                try:
                    yield
                finally:
                    depths[name] -= 1
                return

            runtime_root = self.runtime_root.resolve(strict=False)
            locks_root = self.locks_root.resolve(strict=False)
            if not is_relative_to(locks_root, runtime_root):
                raise RemoteAgentBrokerError(
                    f"Runtime locks path is outside the remote-agent runtime root: {locks_root}"
                )
            try:
                self.locks_root.mkdir(parents=True, exist_ok=True)
                handle = (self.locks_root / f"{name}.lock").open(
                    "a+",
                    encoding="utf-8",
                )
            except OSError as exc:
                raise RemoteAgentBrokerError(
                    f"Failed to open runtime lock {name}: {exc}"
                ) from exc
            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(handle.fileno(), operation)
            except BlockingIOError as exc:
                handle.close()
                raise RemoteAgentBrokerError(
                    f"Runtime lock is already held: {name}"
                ) from exc
            except OSError as exc:
                handle.close()
                raise RemoteAgentBrokerError(
                    f"Failed to acquire runtime lock {name}: {exc}"
                ) from exc
            depths[name] = 1
            try:
                yield
            finally:
                depths.pop(name, None)
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()


class JobStateStore:
    """Own atomic job.json serialization and per-job state locks."""

    def __init__(
        self,
        *,
        jobs_root: Path,
        default_ttl_days: int,
        default_capture_limit_bytes: int,
    ) -> None:
        self.jobs_root = jobs_root
        self.default_ttl_days = require_positive_int(
            default_ttl_days,
            name="default_ttl_days",
        )
        self.default_capture_limit_bytes = default_capture_limit_bytes
        self._lock = threading.RLock()
        self._job_locks: dict[str, threading.RLock] = {}

    def initial_job(
        self,
        *,
        job_id: str,
        kind: str,
        agent: str,
        cwd: Path,
        command: list[str],
        runner_options: Mapping[str, Any],
        prompt: str | None,
        ttl_days: int | None,
        capture_limit_bytes: int | None,
        timeout_sec: int | None,
        idle_timeout_sec: int | None,
        max_wall_time_sec: int | None,
        artifacts: Mapping[str, str],
        request_context: RequestContext,
    ) -> dict[str, Any]:
        now = utc_now()
        return {
            "job_id": job_id,
            "kind": kind,
            "agent": agent,
            "cwd": str(cwd),
            "command": [redact_text(item) for item in command],
            "runner_options": safe_options(runner_options),
            "prompt_summary": summarize_prompt(prompt or "") if prompt else None,
            "request_context": request_context.to_dict(),
            "status": "queued",
            "pid": None,
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "completed_at": None,
            "exit_code": None,
            "duration_sec": None,
            "timeout_sec": timeout_sec,
            "idle_timeout_sec": idle_timeout_sec,
            "max_wall_time_sec": max_wall_time_sec,
            "last_output_at": now,
            "ttl_days": require_positive_int(
                ttl_days if ttl_days is not None else self.default_ttl_days,
                name="ttl_days",
            ),
            "capture_limit_bytes": (
                capture_limit_bytes or self.default_capture_limit_bytes
            ),
            "artifacts": dict(artifacts),
            "last_error": None,
        }

    def public_job(self, job: Mapping[str, Any]) -> dict[str, object]:
        return {"job": dict(job)}

    def job_dir(self, job_id: str) -> Path:
        validate_job_id(job_id)
        return self.jobs_root / job_id

    def read(self, job_id: str) -> dict[str, Any]:
        with self.lock(job_id):
            return self.read_unlocked(job_id)

    def read_unlocked(self, job_id: str) -> dict[str, Any]:
        job_path = self.job_dir(job_id) / "job.json"
        if not job_path.exists():
            raise RemoteAgentBrokerError(f"Unknown job_id: {job_id}")
        try:
            payload = json.loads(job_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RemoteAgentBrokerError(
                f"Job state is not valid JSON: {job_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RemoteAgentBrokerError(f"Job state is not an object: {job_path}")
        return dict(payload)

    def write(self, job_dir: Path, job: Mapping[str, Any]) -> None:
        job_id = str(job.get("job_id") or job_dir.name)
        validate_job_id(job_id)
        if job_dir.name != job_id:
            raise RemoteAgentBrokerError(
                "Job state id does not match its artifact directory"
            )
        with self.lock(job_id):
            self.write_unlocked(job_dir, job)

    def write_unlocked(self, job_dir: Path, job: Mapping[str, Any]) -> None:
        job_dir.mkdir(parents=True, exist_ok=True)
        tmp = job_dir / (
            f"job.json.{os.getpid()}.{threading.get_ident()}."
            f"{secrets.token_hex(4)}.tmp"
        )
        tmp.write_text(
            json.dumps(job, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(job_dir / "job.json")

    def update(
        self,
        job_id: str,
        update: JobUpdater,
    ) -> tuple[dict[str, Any], bool]:
        with self.lock(job_id):
            current = self.read_unlocked(job_id)
            candidate = dict(current)
            update(candidate)
            current_status = str(current.get("status") or "")
            if current_status in TERMINAL_STATUSES:
                candidate["status"] = current_status
            candidate_status = str(candidate.get("status") or "")
            if candidate_status not in JOB_STATUSES:
                raise RemoteAgentBrokerError(
                    f"Invalid job status: {candidate_status or '<empty>'}"
                )
            changed = candidate != current
            if changed:
                self.write_unlocked(self.job_dir(job_id), candidate)
            return candidate, changed

    def lock(self, job_id: str) -> threading.RLock:
        validate_job_id(job_id)
        with self._lock:
            return self._job_locks.setdefault(job_id, threading.RLock())


class JobArtifactManager:
    """Own job directories, logs, transcripts, final payloads, and diffs."""

    def __init__(
        self,
        *,
        state_store: JobStateStore,
        max_read_bytes: int,
        capture_diff: DiffCapture,
    ) -> None:
        self.state_store = state_store
        self.max_read_bytes = max_read_bytes
        self.capture_diff = capture_diff
        self._file_locks = RuntimeFileLocks(runtime_root=self.jobs_root.parent)

    @property
    def jobs_root(self) -> Path:
        return self.state_store.jobs_root

    def allocate_job(self) -> tuple[str, Path]:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        for _ in range(20):
            job_id = f"job_{timestamp_for_id()}_{secrets.token_hex(6)}"
            job_dir = self.jobs_root / job_id
            try:
                job_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            for filename in ("stdout.log", "stderr.log", "transcript.md"):
                (job_dir / filename).touch()
            return job_id, job_dir
        raise RemoteAgentBrokerError("Failed to allocate unique job id")

    def artifact_paths(self, job_id: str) -> dict[str, str]:
        job_dir = self.state_store.job_dir(job_id)
        return {
            "job": str(job_dir / "job.json"),
            "stdout": str(job_dir / "stdout.log"),
            "stderr": str(job_dir / "stderr.log"),
            "transcript": str(job_dir / "transcript.md"),
            "final": str(job_dir / "final.json"),
            "diff": str(job_dir / "diff.patch"),
        }

    def append_transcript(self, job_id: str, text: str) -> None:
        with self.lock(job_id):
            job_dir = self.state_store.job_dir(job_id)
            with (job_dir / "transcript.md").open("a", encoding="utf-8") as handle:
                handle.write(text)

    def write_final(
        self,
        job_id: str,
        job: Mapping[str, Any],
        *,
        stdout_tail: str,
        stderr_tail: str,
    ) -> None:
        with self.lock(job_id):
            final = {
                "job_id": job_id,
                "status": job.get("status"),
                "exit_code": job.get("exit_code"),
                "completed_at": job.get("completed_at"),
                "duration_sec": job.get("duration_sec"),
                "last_error": job.get("last_error"),
                "stdout_tail": redact_text(stdout_tail),
                "stderr_tail": redact_text(stderr_tail),
            }
            (self.state_store.job_dir(job_id) / "final.json").write_text(
                json.dumps(final, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def read(
        self,
        *,
        job_id: str,
        stream: str,
        since_offset: int,
        max_bytes: int | None,
    ) -> dict[str, object]:
        if since_offset < 0:
            raise RemoteAgentBrokerError("since_offset must be >= 0")
        if max_bytes is not None and max_bytes <= 0:
            raise RemoteAgentBrokerError("max_bytes must be greater than zero")
        with self.lock(job_id):
            job_dir = self.state_store.job_dir(job_id)
            if not job_dir.exists():
                raise RemoteAgentBrokerError(f"Unknown job_id: {job_id}")
            job = self.state_store.read(job_id)
            filename = {
                "stdout": "stdout.log",
                "stderr": "stderr.log",
                "transcript": "transcript.md",
                "final": "final.json",
            }.get(stream)
            if filename is None:
                raise RemoteAgentBrokerError(
                    "stream must be stdout, stderr, transcript, or final"
                )
            path = job_dir / filename
            capture_limit = _positive_int_or_default(
                job.get("capture_limit_bytes"), self.max_read_bytes
            )
            limit_ceiling = min(capture_limit, self.max_read_bytes)
            limit = min(max_bytes or limit_ceiling, limit_ceiling)
            if not path.exists():
                return {
                    "job_id": job_id,
                    "stream": stream,
                    "path": str(path),
                    "text": "",
                    "since_offset": since_offset,
                    "next_offset": since_offset,
                    "size": 0,
                    "eof": True,
                }
            text, next_offset, size = self.read_page(
                path,
                since_offset=since_offset,
                limit=limit,
            )
            return {
                "job_id": job_id,
                "stream": stream,
                "path": str(path),
                "text": text,
                "since_offset": since_offset,
                "next_offset": next_offset,
                "size": size,
                "eof": next_offset >= size,
            }

    def read_page(
        self,
        path: Path,
        *,
        since_offset: int,
        limit: int,
    ) -> tuple[str, int, int]:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(since_offset)
            window = handle.read(limit + 4)
        data = window[:limit]
        if not data:
            return "", since_offset, size
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        try:
            text = decoder.decode(data, final=False)
        except UnicodeDecodeError as exc:
            raise RemoteAgentBrokerError(
                f"Artifact is not valid UTF-8 at since_offset={since_offset}; "
                "since_offset must align with a character boundary"
            ) from exc
        buffered, _ = decoder.getstate()
        consumed = len(data) - len(buffered)
        if consumed == 0:
            if len(window) > limit:
                raise RemoteAgentBrokerError(
                    "max_bytes is too small to return the next complete UTF-8 character"
                )
            return "", since_offset, size
        return text, since_offset + consumed, size

    def diff(self, *, job_id: str, base_ref: str | None) -> dict[str, object]:
        with self.lock(job_id):
            job = self.state_store.read(job_id)
            job_dir = self.state_store.job_dir(job_id)
            diff_path = job_dir / "diff.patch"
            if diff_path.exists() and not base_ref:
                return {
                    "job_id": job_id,
                    "available": True,
                    "path": str(diff_path),
                    "diff": diff_path.read_text(encoding="utf-8", errors="replace"),
                }
            return self.capture_job_diff(
                job_id=job_id,
                cwd=Path(str(job.get("cwd") or "")),
                base_ref=base_ref,
            )

    def capture_job_diff(
        self,
        *,
        job_id: str,
        cwd: Path,
        base_ref: str | None = None,
    ) -> dict[str, object]:
        with self.lock(job_id):
            result = self.capture_diff(cwd, base_ref)
            diff_path = self.state_store.job_dir(job_id) / "diff.patch"
            if result.get("available"):
                diff_path.write_text(str(result.get("diff") or ""), encoding="utf-8")
                result["path"] = str(diff_path)
            result["job_id"] = job_id
            return result

    def inventory(self, *, job_id: str) -> dict[str, object]:
        with self.lock(job_id):
            job_dir = self.state_store.job_dir(job_id)
            if not job_dir.exists():
                raise RemoteAgentBrokerError(f"Unknown job_id: {job_id}")
            files = [
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(job_dir.iterdir(), key=lambda item: item.name)
                if path.is_file()
            ]
            return {"job_id": job_id, "job_dir": str(job_dir), "artifacts": files}

    def lock(self, job_id: str) -> ContextManager[None]:
        validate_job_id(job_id)
        return self._file_locks.acquire(f"artifact-{job_id}")


class JobCleanupManager:
    """Own retention cleanup without traversing outside the runtime root."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        default_ttl_days: int,
        state_store: JobStateStore,
        artifact_manager: JobArtifactManager,
        runtime_lookup: RuntimeLookup,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.runtime_root = runtime_root
        self.default_ttl_days = default_ttl_days
        self.state_store = state_store
        self.artifact_manager = artifact_manager
        self.runtime_lookup = runtime_lookup
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._file_locks = RuntimeFileLocks(runtime_root=runtime_root)

    def cleanup(self, *, ttl_days: int | None = None) -> dict[str, object]:
        ttl = require_positive_int(
            ttl_days if ttl_days is not None else self.default_ttl_days,
            name="ttl_days",
        )
        runtime_root = self.runtime_root.resolve(strict=False)
        jobs_root = self.state_store.jobs_root
        resolved_jobs_root = jobs_root.resolve(strict=False)
        if not is_relative_to(resolved_jobs_root, runtime_root):
            return cleanup_boundary_failure(
                ttl_days=ttl,
                path=resolved_jobs_root,
                runtime_root=runtime_root,
            )
        acquired = False
        try:
            with self._file_locks.acquire("cleanup", blocking=False):
                acquired = True
                return self._cleanup_locked(ttl_days=ttl_days, reported_ttl=ttl)
        except (OSError, RemoteAgentBrokerError) as exc:
            failure = "cleanup failure" if acquired else "cleanup lock failure"
            return {
                "ok": False,
                "ttl_days": ttl,
                "removed": [],
                "removed_count": 0,
                "error": f"{failure}: {exc}",
            }

    def _cleanup_locked(
        self,
        *,
        ttl_days: int | None,
        reported_ttl: int,
    ) -> dict[str, object]:
        now = self.now()
        expired_dirs: list[tuple[str, Path, int]] = []
        removed: list[str] = []
        runtime_root = self.runtime_root.resolve(strict=False)
        jobs_root = self.state_store.jobs_root
        jobs_root.mkdir(parents=True, exist_ok=True)
        for job_dir in sorted(jobs_root.iterdir(), key=lambda item: item.name):
            if not JOB_ID_RE.fullmatch(job_dir.name):
                continue
            if job_dir.is_symlink():
                resolved_job_dir = job_dir.resolve(strict=False)
                if not is_relative_to(resolved_job_dir, runtime_root):
                    return cleanup_boundary_failure(
                        ttl_days=reported_ttl,
                        path=resolved_job_dir,
                        runtime_root=runtime_root,
                    )
                return cleanup_safety_failure(
                    ttl_days=reported_ttl,
                    path=job_dir,
                    detail="job directory is a symbolic link",
                )
            if not job_dir.is_dir():
                continue
            resolved_job_dir = job_dir.resolve(strict=False)
            if not is_relative_to(resolved_job_dir, runtime_root):
                return cleanup_boundary_failure(
                    ttl_days=reported_ttl,
                    path=resolved_job_dir,
                    runtime_root=runtime_root,
                )
            with self.state_store.lock(job_dir.name):
                if self.runtime_lookup(job_dir.name) is not None:
                    continue
                try:
                    job = self.state_store.read_unlocked(job_dir.name)
                except RemoteAgentBrokerError:
                    continue
                status = str(job.get("status") or "")
                if status not in TERMINAL_STATUSES:
                    continue
                stamp = parse_utc(
                    str(job.get("completed_at") or job.get("updated_at") or "")
                )
                effective_ttl = require_positive_int(
                    ttl_days
                    if ttl_days is not None
                    else job.get("ttl_days", self.default_ttl_days),
                    name=f"{job_dir.name}.ttl_days",
                )
                if stamp and stamp < now - timedelta(days=effective_ttl):
                    expired_dirs.append((job_dir.name, job_dir, effective_ttl))
        for job_id, job_dir, effective_ttl in expired_dirs:
            with self.artifact_manager.lock(job_id):
                with self.state_store.lock(job_id):
                    if self.runtime_lookup(job_id) is not None:
                        continue
                    if job_dir.is_symlink():
                        resolved_job_dir = job_dir.resolve(strict=False)
                        if not is_relative_to(resolved_job_dir, runtime_root):
                            return cleanup_boundary_failure(
                                ttl_days=reported_ttl,
                                path=resolved_job_dir,
                                runtime_root=runtime_root,
                            )
                        return cleanup_safety_failure(
                            ttl_days=reported_ttl,
                            path=job_dir,
                            detail="job directory became a symbolic link during cleanup",
                        )
                    if not job_dir.exists():
                        continue
                    resolved_job_dir = job_dir.resolve(strict=False)
                    if not is_relative_to(resolved_job_dir, runtime_root):
                        return cleanup_boundary_failure(
                            ttl_days=reported_ttl,
                            path=resolved_job_dir,
                            runtime_root=runtime_root,
                        )
                    if not job_dir.is_dir():
                        return cleanup_safety_failure(
                            ttl_days=reported_ttl,
                            path=job_dir,
                            detail="job directory is no longer a directory",
                        )
                    try:
                        job = self.state_store.read_unlocked(job_id)
                    except RemoteAgentBrokerError:
                        continue
                    status = str(job.get("status") or "")
                    stamp = parse_utc(
                        str(job.get("completed_at") or job.get("updated_at") or "")
                    )
                    if (
                        status not in TERMINAL_STATUSES
                        or stamp is None
                        or stamp >= now - timedelta(days=effective_ttl)
                    ):
                        continue
                    shutil.rmtree(job_dir)
                    removed.append(job_id)
        return {
            "ok": True,
            "ttl_days": reported_ttl,
            "retention_policy": "override" if ttl_days is not None else "per_job",
            "removed": removed,
            "removed_count": len(removed),
        }


def cleanup_safety_failure(
    *,
    ttl_days: int,
    path: Path,
    detail: str,
) -> dict[str, object]:
    return {
        "ok": False,
        "ttl_days": ttl_days,
        "removed": [],
        "removed_count": 0,
        "error": f"cleanup safety failure: {path}: {detail}",
    }


def _positive_int_or_default(value: object, default: int) -> int:
    if type(value) is not int:
        return default
    return value if value > 0 else default
