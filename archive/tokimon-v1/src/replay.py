"""Replay artifacts: record and offline-replay worker steps deterministically.

Per `docs/c4/level-3-component/tokimon/requirements.md`:
- Persist per-step `replay.json` artifacts under `artifacts/steps/<step_id>/replay.json`.
- Record model responses + invoked tool calls/results with bounded payload sizes and token redaction.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flow_types import ToolCallRecord
from tools.base import ToolResult

SCHEMA_VERSION = "1.0"

_BEARER_TOKEN_PATTERN = re.compile(r"(authorization\s*:\s*bearer\s+)(\S+)", re.IGNORECASE)


def _redact_bearer_tokens(text: str) -> str:
    return _BEARER_TOKEN_PATTERN.sub(r"\1<REDACTED>", text)


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _make_json_safe(child) for key, child in value.items()}
    return str(value)


def _truncate_jsonish(value: Any, *, max_str: int, max_list: int, max_depth: int) -> Any:
    if max_depth <= 0:
        return "<truncated>"
    if isinstance(value, str):
        value = _redact_bearer_tokens(value)
        if len(value) <= max_str:
            return value
        return value[:max_str] + f"...(truncated {len(value) - max_str} chars)"
    if isinstance(value, list):
        if len(value) <= max_list:
            return [_truncate_jsonish(v, max_str=max_str, max_list=max_list, max_depth=max_depth - 1) for v in value]
        head = value[:max_list]
        head_truncated = [
            _truncate_jsonish(v, max_str=max_str, max_list=max_list, max_depth=max_depth - 1) for v in head
        ]
        return [*head_truncated, f"...(truncated {len(value) - max_list} items)"]
    if isinstance(value, dict):
        return {
            k: _truncate_jsonish(v, max_str=max_str, max_list=max_list, max_depth=max_depth - 1)
            for k, v in value.items()
        }
    return value


def sanitize_replay_payload(value: Any) -> Any:
    """Redact + bound payload sizes deterministically for replay artifacts."""

    safe = _make_json_safe(value)
    return _truncate_jsonish(safe, max_str=20_000, max_list=100, max_depth=6)


def sha256_stable_json(payload: Any) -> str:
    safe = _make_json_safe(payload)
    blob = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class ReplayRecorder:
    step_id: str
    worker_role: str
    goal: str
    inputs: dict[str, Any]
    memory: list[str]
    model_script: list[dict[str, Any]] = field(default_factory=list)
    tool_script: list[dict[str, Any]] = field(default_factory=list)
    final_result: dict[str, Any] | None = None

    def record_model_response(self, response: dict[str, Any]) -> None:
        sanitized = sanitize_replay_payload(response)
        if isinstance(sanitized, dict):
            self.model_script.append(sanitized)

    def record_tool_invocation(self, call: dict[str, Any], record: ToolCallRecord) -> None:
        if record.cached:
            return

        tool = str(call.get("tool") or "")
        action = str(call.get("action") or "")
        raw_args = call.get("args", {}) or {}
        args = raw_args if isinstance(raw_args, dict) else {}
        args_preview = sanitize_replay_payload(args)

        entry: dict[str, Any] = {
            "tool": tool,
            "action": action,
            "args_hash": sha256_stable_json(args_preview),
            "args_preview": args_preview,
            "result": sanitize_replay_payload(
                {
                    "ok": record.ok,
                    "summary": record.summary,
                    "data": record.data,
                    "error": record.error,
                    "elapsed_ms": record.elapsed_ms,
                }
            ),
        }
        if record.call_id:
            entry["call_id"] = record.call_id
        self.tool_script.append(entry)

    def record_final(self, payload: dict[str, Any]) -> None:
        sanitized = sanitize_replay_payload(payload)
        if isinstance(sanitized, dict):
            self.final_result = sanitized
        else:
            self.final_result = {"status": "FAILURE", "summary": "invalid final_result"}

    def build(self) -> dict[str, Any]:
        recorded_at = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": recorded_at,
            "step_id": self.step_id,
            "worker_role": self.worker_role,
            "goal": sanitize_replay_payload(self.goal),
            "inputs": sanitize_replay_payload(self.inputs),
            "memory": sanitize_replay_payload(self.memory),
            "model_script": sanitize_replay_payload(self.model_script),
            "tool_script": sanitize_replay_payload(self.tool_script),
            "final_result": self.final_result or {},
        }


class ReplayAbort(BaseException):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ReplayLLMClient:
    def __init__(self, *, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.calls = 0

    def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (messages, tools, response_schema)
        self.calls += 1
        if not self.script:
            raise ReplayAbort("replay model_script exhausted", details={"calls": self.calls})
        next_item = self.script.pop(0)
        if not isinstance(next_item, dict):
            raise ReplayAbort(
                "replay model_script entry is not an object",
                details={"calls": self.calls, "entry_type": type(next_item).__name__},
            )
        return dict(next_item)


@dataclass
class ReplayToolRouter:
    tool_script: list[dict[str, Any]]
    index: int = 0

    def invoke(self, tool: str, action: str, args: dict[str, Any]) -> ToolResult:
        if self.index >= len(self.tool_script):
            raise ReplayAbort(
                "replay tool_script exhausted",
                details={"tool": tool, "action": action, "index": self.index, "count": len(self.tool_script)},
            )
        expected = self.tool_script[self.index]
        expected_tool = str(expected.get("tool") or "")
        expected_action = str(expected.get("action") or "")
        expected_hash = str(expected.get("args_hash") or "")
        args_preview = sanitize_replay_payload(args)
        actual_hash = sha256_stable_json(args_preview)
        if expected_tool != tool or expected_action != action or expected_hash != actual_hash:
            raise ReplayAbort(
                "replay tool invocation mismatch",
                details={
                    "index": self.index,
                    "expected": expected,
                    "actual": {
                        "tool": tool,
                        "action": action,
                        "args_hash": actual_hash,
                        "args_preview": args_preview,
                    },
                },
            )
        self.index += 1
        raw_result = expected.get("result") or {}
        result = raw_result if isinstance(raw_result, dict) else {}
        ok = result.get("ok")
        summary = result.get("summary")
        data = result.get("data")
        error = result.get("error")
        elapsed_ms = result.get("elapsed_ms")
        return ToolResult(
            ok=ok if isinstance(ok, bool) else False,
            summary=str(summary or ""),
            data=data if isinstance(data, dict) else {},
            elapsed_ms=float(elapsed_ms) if isinstance(elapsed_ms, (int, float)) else 0.0,
            error=str(error) if error is not None else None,
        )


class ReplayTool:
    def __init__(self, tool_name: str, router: ReplayToolRouter) -> None:
        self._tool_name = tool_name
        self._router = router

    def __getattr__(self, action: str) -> Any:
        if action.startswith("_"):
            raise AttributeError(action)

        def _fn(**kwargs: Any) -> ToolResult:
            return self._router.invoke(self._tool_name, action, kwargs)

        return _fn


def replay_step(replay_record: dict[str, Any]) -> dict[str, Any]:
    schema_version = str(replay_record.get("schema_version") or "")
    if schema_version != SCHEMA_VERSION:
        raise ReplayAbort(
            "replay schema_version mismatch",
            details={"expected": SCHEMA_VERSION, "actual": schema_version},
        )

    step_id = str(replay_record.get("step_id") or "")
    worker_role = str(replay_record.get("worker_role") or "")
    goal = str(replay_record.get("goal") or "")
    inputs_raw = replay_record.get("inputs")
    inputs = inputs_raw if isinstance(inputs_raw, dict) else {}
    memory_raw = replay_record.get("memory")
    memory_list = memory_raw if isinstance(memory_raw, list) else []
    memory = [str(item) for item in memory_list]
    model_script_raw = replay_record.get("model_script")
    model_script_items = model_script_raw if isinstance(model_script_raw, list) else []
    model_script = [item for item in model_script_items if isinstance(item, dict)]
    tool_script_raw = replay_record.get("tool_script")
    tool_script_items = tool_script_raw if isinstance(tool_script_raw, list) else []
    tool_script = [item for item in tool_script_items if isinstance(item, dict)]
    expected_final_raw = replay_record.get("final_result")
    expected_final = expected_final_raw if isinstance(expected_final_raw, dict) else {}

    router = ReplayToolRouter(tool_script=tool_script)
    tool_names = _collect_replay_tool_names(model_script, tool_script)
    tools = {name: ReplayTool(name, router) for name in sorted(tool_names)}

    llm_client = ReplayLLMClient(script=model_script)
    from agents.worker import Worker

    worker = Worker(worker_role or "Worker", llm_client, tools)
    output = worker.run(goal, step_id, inputs=inputs, memory=memory)

    if router.index != len(tool_script):
        raise ReplayAbort(
            "replay did not consume full tool_script",
            details={"consumed": router.index, "count": len(tool_script)},
        )
    if llm_client.script:
        raise ReplayAbort(
            "replay did not consume full model_script",
            details={"remaining": len(llm_client.script), "calls": llm_client.calls},
        )

    _validate_final_result(expected_final, output)
    return {"ok": True, "step_id": step_id, "status": output.status.value}


def replay_run(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    steps_dir = run_root / "artifacts" / "steps"
    if not steps_dir.exists():
        raise ReplayAbort("replay steps directory missing", details={"steps_dir": str(steps_dir)})

    step_dirs = sorted([path for path in steps_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    if not step_dirs:
        raise ReplayAbort("replay found no step directories", details={"steps_dir": str(steps_dir)})

    results: list[dict[str, Any]] = []
    for step_dir in step_dirs:
        replay_path = step_dir / "replay.json"
        if not replay_path.exists():
            raise ReplayAbort("replay.json missing", details={"step_id": step_dir.name, "path": str(replay_path)})
        try:
            record = json.loads(replay_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ReplayAbort(
                "replay.json parse error",
                details={"step_id": step_dir.name, "path": str(replay_path), "error": str(exc)},
            ) from exc
        if not isinstance(record, dict):
            raise ReplayAbort(
                "replay.json is not an object",
                details={"step_id": step_dir.name, "path": str(replay_path)},
            )
        try:
            results.append(replay_step(record))
        except ReplayAbort as exc:
            raise ReplayAbort(
                "replay step failed",
                details={"step_id": str(record.get("step_id") or step_dir.name), **(exc.details or {})},
            ) from exc
    return {"ok": True, "run_root": str(run_root), "steps": results}


def _collect_replay_tool_names(model_script: list[dict[str, Any]], tool_script: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for entry in tool_script:
        tool = entry.get("tool")
        if isinstance(tool, str) and tool.strip():
            names.add(tool.strip())
    for response in model_script:
        raw_calls = response.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for call in raw_calls:
            if not isinstance(call, dict):
                continue
            tool = call.get("tool")
            if isinstance(tool, str) and tool.strip():
                names.add(tool.strip())
    return names


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _validate_final_result(expected: dict[str, Any], output: Any) -> None:
    expected_status = str(expected.get("status") or "")
    expected_summary = str(expected.get("summary") or "")
    expected_failure = str(expected.get("failure_signature") or "")

    actual_status = str(getattr(output, "status").value)
    actual_summary = str(getattr(output, "summary") or "")
    actual_failure = str(getattr(output, "failure_signature") or "")

    if expected_status != actual_status:
        raise ReplayAbort(
            "replay final_result status mismatch",
            details={"expected": expected_status, "actual": actual_status},
        )
    if expected_summary != actual_summary:
        raise ReplayAbort(
            "replay final_result summary mismatch",
            details={"expected": expected_summary, "actual": actual_summary},
        )
    if expected_failure != actual_failure:
        raise ReplayAbort(
            "replay final_result failure_signature mismatch",
            details={"expected": expected_failure, "actual": actual_failure},
        )

    expected_metrics_raw = expected.get("metrics")
    expected_metrics = expected_metrics_raw if isinstance(expected_metrics_raw, dict) else {}
    actual_metrics_raw = getattr(output, "metrics", None)
    actual_metrics = actual_metrics_raw if isinstance(actual_metrics_raw, dict) else {}
    for key in ("model_calls", "tool_calls", "iteration_count", "schema_repairs"):
        if key not in expected_metrics:
            continue
        expected_value = _coerce_int(expected_metrics.get(key))
        actual_value = _coerce_int(actual_metrics.get(key))
        if expected_value != actual_value:
            raise ReplayAbort(
                f"replay final_result metrics mismatch: {key}",
                details={"key": key, "expected": expected_value, "actual": actual_value},
            )
