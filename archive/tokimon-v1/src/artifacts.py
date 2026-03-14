"""Artifact store for workflow steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from memory.store import MemoryStore


class ArtifactStore:
    def __init__(self, base_dir: Path, memory_store: MemoryStore | None = None) -> None:
        self.base_dir = base_dir
        self.memory_store = memory_store

    def write_step(
        self,
        task_id: str,
        step_id: str,
        artifacts: list[dict[str, Any]],
        outputs: dict[str, Any] | None = None,
        *,
        step_result: dict[str, Any] | None = None,
        replay_record: dict[str, Any] | None = None,
    ) -> str:
        step_dir = self.base_dir / step_id
        step_dir.mkdir(parents=True, exist_ok=True)
        step_result_path = step_dir / "step_result.json"
        outputs_path = step_dir / "outputs.json"
        replay_path = step_dir / "replay.json"
        step_result_path.write_text(_stable_json_dumps(step_result or {}))
        outputs_path.write_text(_stable_json_dumps(outputs or {}))
        artifacts_path = step_dir / "artifacts.json"
        artifacts_path.write_text(_stable_json_dumps(artifacts))
        replay_path.write_text(_stable_json_dumps(replay_record or {}))
        digest = self._hash_files([step_result_path, outputs_path, artifacts_path, replay_path])
        if self.memory_store:
            artifact_id = f"{task_id}-{step_id}-{digest[:8]}"
            self.memory_store.index_artifact(artifact_id, step_id, task_id, step_dir, digest, {"count": len(artifacts)})
        return digest

    def _hash_files(self, paths: list[Path]) -> str:
        hasher = hashlib.sha256()
        for path in paths:
            hasher.update(path.read_bytes())
        return hasher.hexdigest()


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
