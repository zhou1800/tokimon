"""Persistent bounded jobs that outlive individual MCP connections."""

from __future__ import annotations

import json
import logging
import os
import selectors
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .files import FileBoundary


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


class JobRuntime:
    def __init__(self, state: Path, limits: dict, files: FileBoundary, logger: logging.Logger):
        self.state, self.limits, self.files, self.logger = state, limits, files, logger
        self.lock = threading.RLock()
        self.records: dict[str, dict] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.cancelled: set[str] = set()
        self.paused = False
        state.mkdir(parents=True, exist_ok=True)
        self.artifacts = FileBoundary({str(state): False})
        with self.artifacts.directory(str(state)) as directory, os.scandir(directory) as entries:
            for entry in entries:
                job_id = entry.name
                if len(self.records) >= limits["max_jobs"] or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
                    raise RuntimeError("corrupt or excessive persisted job storage")
                with self.artifacts.open_regular(str(state / job_id / "job.json"), os.O_RDONLY) as fd:
                    if os.fstat(fd).st_size > 65536:
                        raise RuntimeError("persisted job metadata exceeds bounds")
                    record = json.loads(os.read(fd, 65537))
                if record["job_id"] != job_id or record["state"] not in {"running", "succeeded", "failed", "cancelled", "timed_out", "interrupted"}:
                    raise RuntimeError("corrupt persisted job identity/state")
                if record["state"] == "running":
                    record.update(state="interrupted", finished_at=now(), exit_code=None)
                    self._persist(record)
                self.records[job_id] = record

    def _persist(self, record: dict) -> None:
        self.artifacts.atomic_write(str(self.state / record["job_id"] / "job.json"), json.dumps(record).encode() + b"\n")

    def start(self, command: str, cwd: str, timeout: int, *, correlation_id: str) -> dict:
        with self.lock:
            if self.paused:
                raise ValueError("deployment maintenance temporarily pauses new jobs")
            if len(self.records) >= self.limits["max_jobs"]:
                raise ValueError("job-count limit reached; manage retained storage explicitly")
            if len(self.processes) >= self.limits["concurrency"]:
                raise ValueError("concurrency limit reached")
            job_id = uuid.uuid4().hex
            directory = self.state / job_id
            with self.artifacts.directory(str(self.state)) as parent:
                os.mkdir(job_id, mode=0o700, dir_fd=parent)
            record = {"job_id": job_id, "state": "running", "started_at": now(), "finished_at": None,
                      "exit_code": None, "stdout_bytes": 0, "stderr_bytes": 0,
                      "stdout_truncated": False, "stderr_truncated": False, "request_id": correlation_id}
            self._persist(record)
            try:
                with self.files.directory(cwd) as cwd_fd:
                    process = subprocess.Popen(
                        ["/bin/bash", "--noprofile", "--norc", "-c", command],
                        cwd=f"/proc/self/fd/{cwd_fd}", pass_fds=(cwd_fd,), start_new_session=True,
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/workspace", "LANG": "C.UTF-8"},
                    )
            except Exception:
                record.update(state="failed", finished_at=now())
                self._persist(record)
                self.records[job_id] = record
                raise
            self.records[job_id] = record
            self.processes[job_id] = process
            try:
                threading.Thread(target=self._monitor, args=(job_id, process, timeout), daemon=True).start()
            except Exception:
                self._signal(process, signal.SIGKILL)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.logger.error("job_reap_timeout", extra={"correlation_id": correlation_id, "job_id": job_id})
                finally:
                    for stream in (process.stdout, process.stderr):
                        stream.close()
                    self.processes.pop(job_id)
                    record.update(state="failed", finished_at=now(), exit_code=process.returncode)
                    self._persist(record)
                self.logger.error("job_monitor_start_failed", extra={"correlation_id": correlation_id, "job_id": job_id})
                raise
            self.logger.info("job_started", extra={"correlation_id": correlation_id, "job_id": job_id})
            return dict(record)

    @staticmethod
    def _signal(process: subprocess.Popen, number: int) -> None:
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass

    def _monitor(self, job_id: str, process: subprocess.Popen, timeout: int) -> None:
        started = time.monotonic()
        outcome = None
        stopping = None
        exited = None
        try:
            with (
                selectors.DefaultSelector() as selector,
                self.artifacts.open_regular(str(self.state / job_id / "stdout"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, write=True) as out_fd,
                self.artifacts.open_regular(str(self.state / job_id / "stderr"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, write=True) as err_fd,
                os.fdopen(out_fd, "wb", closefd=False) as stdout,
                os.fdopen(err_fd, "wb", closefd=False) as stderr,
            ):
                for stream, name, output in ((process.stdout, "stdout", stdout), (process.stderr, "stderr", stderr)):
                    os.set_blocking(stream.fileno(), False)
                    selector.register(stream, selectors.EVENT_READ, (name, output))
                while selector.get_map() or process.poll() is None:
                    tick = time.monotonic()
                    with self.lock:
                        cancel = job_id in self.cancelled
                    if outcome is None and (cancel or tick - started >= timeout):
                        outcome = "cancelled" if cancel else "timed_out"
                        stopping = tick
                        self._signal(process, signal.SIGTERM)
                    if stopping is not None and tick - stopping >= 1:
                        self._signal(process, signal.SIGKILL)
                    if process.poll() is not None and exited is None:
                        exited = tick
                        self._signal(process, signal.SIGTERM)
                    if exited is not None and tick - exited >= 1.5:
                        self._signal(process, signal.SIGKILL)
                        break
                    for key, _ in selector.select(0.05):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        name, output = key.data
                        with self.lock:
                            record = self.records[job_id]
                            remaining = self.limits["output_bytes"] - record[name + "_bytes"]
                            kept = data[:max(0, remaining)]
                            output.write(kept)
                            output.flush()
                            record[name + "_bytes"] += len(kept)
                            if len(kept) < len(data):
                                record[name + "_truncated"] = True
                            self._persist(record)
                code = process.wait(timeout=3)
            with self.lock:
                record = self.records[job_id]
                record.update(state=outcome or ("succeeded" if code == 0 else "failed"), finished_at=now(), exit_code=code)
                self._persist(record)
        except Exception:
            self._signal(process, signal.SIGKILL)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            with self.lock:
                self.records[job_id].update(state="failed", finished_at=now(), exit_code=None)
                try:
                    self._persist(self.records[job_id])
                except (OSError, ValueError):
                    pass
            self.logger.error("job_artifact_failure", extra={"correlation_id": self.records[job_id]["request_id"], "job_id": job_id})
        finally:
            for stream in (process.stdout, process.stderr):
                stream.close()
            with self.lock:
                self.processes.pop(job_id, None)
                self.cancelled.discard(job_id)
            self.logger.info("job_finished", extra={"correlation_id": self.records[job_id]["request_id"], "job_id": job_id})

    def status(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.records:
                raise ValueError("unknown job_id")
            return dict(self.records[job_id])

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            record = self.status(job_id)
            if record["state"] == "running":
                self.cancelled.add(job_id)
            return record

    def read(self, job_id: str, stream: str, offset: int, max_bytes: int) -> dict:
        record = self.status(job_id)
        path = self.state / job_id / stream
        try:
            with self.artifacts.open_regular(str(path), os.O_RDONLY) as fd:
                data = os.pread(fd, max_bytes, offset)
        except FileNotFoundError:
            data = b""
        return {"job_id": job_id, "stream": stream, "content": data.decode("utf-8", errors="replace"),
                "next_offset": offset + len(data), "state": record["state"]}

    def jobs(self, limit: int) -> dict:
        with self.lock:
            return {"jobs": [dict(r) for r in sorted(self.records.values(), key=lambda r: r["started_at"], reverse=True)[:limit]]}

    def maintenance(self, paused: bool) -> dict:
        with self.lock:
            if paused and self.processes:
                raise ValueError("active jobs prevent update/rollback")
            self.paused = paused
            return {"paused": paused, "active_jobs": len(self.processes)}
