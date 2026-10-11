"""Descriptor-relative file access under explicitly available container roots."""

from __future__ import annotations

import base64
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import PurePosixPath

MAX_BYTES = 1048576


class FileBoundary:
    def __init__(self, roots: dict[str, bool]):
        self.roots = dict(roots)

    def _parts(self, path: object, *, write: bool = False) -> tuple[str, tuple[str, ...]]:
        if not isinstance(path, str) or len(path.encode()) > 4096 or "\0" in path:
            raise ValueError("path must be a bounded absolute path")
        p = PurePosixPath(path)
        if not p.is_absolute() or ".." in p.parts:
            raise ValueError("path must be absolute without parent traversal")
        for root, read_only in self.roots.items():
            if p == PurePosixPath(root) or PurePosixPath(root) in p.parents:
                if write and read_only:
                    raise PermissionError("mount is read-only")
                return root, p.relative_to(root).parts
        raise PermissionError("path is outside workspace and declared mounts")

    @contextmanager
    def directory(self, path: object):
        root, parts = self._parts(path)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            for part in parts:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            yield fd
        finally:
            os.close(fd)

    @contextmanager
    def _parent(self, path: object, *, write: bool = False):
        root, parts = self._parts(path, write=write)
        if not parts:
            raise ValueError("file path must name a child of an allowed root")
        with self.directory(str(PurePosixPath(root).joinpath(*parts[:-1]))) as fd:
            yield fd, parts[-1]

    def list(self, path: str, limit: int) -> dict:
        with self.directory(path) as fd:
            entries = []
            with os.scandir(fd) as scan:
                for entry in scan:
                    if len(entries) == limit:
                        return {"entries": entries, "truncated": True}
                    info = entry.stat(follow_symlinks=False)
                    kind = "directory" if stat.S_ISDIR(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else "symlink" if stat.S_ISLNK(info.st_mode) else "special"
                    entries.append({"name": entry.name, "type": kind, "size": info.st_size})
            return {"entries": entries, "truncated": False}

    def read(self, path: str, offset: int, max_bytes: int, encoding: str) -> dict:
        with self.open_regular(path, os.O_RDONLY) as fd:
            data = os.pread(fd, max_bytes, offset)
            return {"content": data.decode("utf-8", errors="replace") if encoding == "utf-8" else base64.b64encode(data).decode(),
                    "encoding": encoding, "next_offset": offset + len(data), "eof": offset + len(data) >= os.fstat(fd).st_size}

    @contextmanager
    def open_regular(self, path: str, flags: int, *, write: bool = False):
        with self._parent(path, write=write) as (parent, name):
            fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600, dir_fd=parent)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("only regular files are supported")
                yield fd
            finally:
                os.close(fd)

    def atomic_write(self, path: str, data: bytes) -> None:
        with self._parent(path, write=True) as (parent, name):
            try:
                if not stat.S_ISREG(os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode):
                    raise ValueError("persisted metadata must be a regular file")
            except FileNotFoundError:
                pass
            temporary = ".record-" + uuid.uuid4().hex
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb", closefd=False) as output:
                    output.write(data)
                    output.flush()
                    os.fsync(fd)
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(fd)
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def write(self, path: str, content: str, encoding: str, overwrite: bool) -> dict:
        data = content.encode("utf-8") if encoding == "utf-8" else base64.b64decode(content, validate=True)
        if len(data) > MAX_BYTES:
            raise ValueError("decoded content exceeds 1 MiB")
        flags = os.O_WRONLY | os.O_CREAT | (0 if overwrite else os.O_EXCL)
        with self.open_regular(path, flags, write=True) as fd:
            os.ftruncate(fd, 0)
            with os.fdopen(fd, "wb", closefd=False) as output:
                output.write(data)
                output.flush()
                os.fsync(fd)
        return {"path": path, "bytes_written": len(data)}
