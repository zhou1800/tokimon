"""Private bounded JSON and no-follow locking for machine controller state."""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from . import MAX_FRAME
from .config import absolute_path
from .files import FileBoundary


class StateStore:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = absolute_path(str(root), directory=True)
        self.files = FileBoundary({str(self.root): False})

    def read(self, name: str, default=None):
        try:
            with self.files.open_regular(str(self.root / name), os.O_RDONLY) as fd:
                data = os.read(fd, MAX_FRAME + 1)
                if len(data) > MAX_FRAME:
                    raise ValueError("machine state exceeds 2 MiB")
                return json.loads(data.decode("utf-8"))
        except FileNotFoundError:
            return default

    def write(self, name: str, value) -> None:
        data = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
        if len(data) > MAX_FRAME:
            raise ValueError("machine state exceeds 2 MiB")
        self.files.atomic_write(str(self.root / name), data)

    @contextmanager
    def locked(self):
        with self.files.open_regular(str(self.root / "controller.lock"), os.O_WRONLY | os.O_CREAT, write=True) as fd:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
