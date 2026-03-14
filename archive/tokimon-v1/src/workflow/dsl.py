"""Workflow DSL loader for YAML/JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import StepSpec, WorkflowSpec


def workflow_from_dict(data: dict[str, Any]) -> WorkflowSpec:
    steps = [
        StepSpec(
            step_id=step["id"],
            name=step.get("name", step["id"]),
            description=step.get("description", ""),
            worker=step.get("worker", "Implementer"),
            depends_on=step.get("depends_on", []),
            inputs_schema=step.get("inputs_schema", {}),
            outputs_schema=step.get("outputs_schema", {}),
            default_inputs=step.get("inputs", {}),
        )
        for step in data.get("steps", [])
    ]
    return WorkflowSpec(
        workflow_id=data.get("workflow_id", "workflow"),
        goal=data.get("goal", ""),
        steps=steps,
        metadata=data.get("metadata", {}),
    )


def load_workflow(path: Path) -> WorkflowSpec:
    payload = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(payload)
    else:
        data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Workflow DSL must be a JSON/YAML object")
    return workflow_from_dict(data)
