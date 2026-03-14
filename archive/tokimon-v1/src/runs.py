"""Run context and artifact paths."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    run_id: str
    root: Path
    workflow_state_path: Path
    trace_path: Path
    logs_dir: Path
    artifacts_dir: Path
    lessons_dir: Path
    reports_dir: Path

    def write_manifest(self, metadata: dict[str, Any]) -> None:
        manifest_path = self.root / "run.json"
        payload = {"run_id": self.run_id, "created_at": datetime.now(timezone.utc).isoformat(), **metadata}
        manifest_path.write_text(json.dumps(payload, indent=2))


def create_run_context(base_dir: Path) -> RunContext:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = base_dir / f"run-{run_id}"
    logs_dir = root / "logs"
    artifacts_dir = root / "artifacts" / "steps"
    lessons_dir = root / "lessons"
    reports_dir = root / "reports"
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    lessons_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    trace_path = root / "trace.jsonl"
    workflow_state_path = root / "workflow_state.json"
    return RunContext(
        run_id=run_id,
        root=root,
        workflow_state_path=workflow_state_path,
        trace_path=trace_path,
        logs_dir=logs_dir,
        artifacts_dir=artifacts_dir,
        lessons_dir=lessons_dir,
        reports_dir=reports_dir,
    )


def load_run_context(run_root: Path) -> RunContext:
    return RunContext(
        run_id=run_root.name,
        root=run_root,
        workflow_state_path=run_root / "workflow_state.json",
        trace_path=run_root / "trace.jsonl",
        logs_dir=run_root / "logs",
        artifacts_dir=run_root / "artifacts" / "steps",
        lessons_dir=run_root / "lessons",
        reports_dir=run_root / "reports",
    )
