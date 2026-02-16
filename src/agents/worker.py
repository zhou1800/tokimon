"""Worker agent implementation."""

from __future__ import annotations

import inspect
import json
import time
from typing import Any

from agents.outputs import WorkerOutput
from agents.prompts import build_system_prompt
from flow_types import ToolCallRecord, WorkerStatus
from llm.client import LLMClient
from tools.base import ToolResult
from tracing import TraceLogger


class Worker:
    def __init__(self, role: str, llm_client: LLMClient, tools: dict[str, Any]) -> None:
        self.role = role
        self.llm_client = llm_client
        self.tools = tools

    def run(self, goal: str, step_id: str, inputs: dict[str, Any], memory: list[str],
            max_iterations: int = 20, *, trace: TraceLogger | None = None,
            trace_context: dict[str, Any] | None = None) -> WorkerOutput:
        start = time.perf_counter()
        model_calls = 0
        tool_call_records: list[ToolCallRecord] = []
        touched_files: set[str] = set()
        trace_base = _trace_base(
            {
                "worker_role": self.role,
                "step_id": step_id,
                **(trace_context or {}),
            }
        )
        messages = [
            {"role": "system", "content": build_system_prompt(self.role)},
            {"role": "user", "content": f"Goal: {goal}\nStep: {step_id}\nInputs: {inputs}\nMemory: {memory}"},
        ]
        for iteration in range(1, max_iterations + 1):
            _trace_log(
                trace,
                "worker_model_call",
                {
                    **trace_base,
                    "iteration": iteration,
                    "message_count": len(messages),
                    "tool_count": len(self.tools),
                },
            )
            response = self.llm_client.send(messages, tools=_tool_descriptors(self.tools))
            model_calls += 1
            _trace_log(trace, "worker_model_response", {**trace_base, "iteration": iteration, **_response_meta(response)})

            try:
                tool_calls = _parse_tool_calls(response)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _trace_log(
                    trace,
                    "worker_invalid_tool_calls",
                    {**trace_base, "iteration": iteration, "error": str(exc)},
                )
                return WorkerOutput(
                    status=WorkerStatus.FAILURE,
                    summary=f"invalid tool_calls: {exc}",
                    artifacts=[],
                    metrics={
                        "elapsed_ms": elapsed_ms,
                        "model_calls": model_calls,
                        "tool_calls": len(tool_call_records),
                        "iteration_count": iteration,
                        "tool_call_records": [record.__dict__ for record in tool_call_records],
                    },
                    next_actions=["Fix tool_calls schema or return a final response."],
                    failure_signature="worker-invalid-tool-calls",
                )
            if tool_calls:
                for call in tool_calls:
                    _trace_log(
                        trace,
                        "worker_tool_call",
                        {
                            **trace_base,
                            "iteration": iteration,
                            "call": _truncate_jsonish(call, max_str=4_000, max_list=50, max_depth=4),
                        },
                    )
                    touched_files.update(_touched_files_from_call(call))
                    record = _invoke_tool_call(self.tools, call)
                    tool_call_records.append(record)
                    _trace_log(
                        trace,
                        "worker_tool_result",
                        {
                            **trace_base,
                            "iteration": iteration,
                            "tool_name": record.tool_name,
                            "ok": record.ok,
                            "summary": record.summary,
                            "elapsed_ms": record.elapsed_ms,
                            "error": record.error,
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "name": record.tool_name,
                            "content": _format_tool_message(record),
                        }
                    )
                continue

            output = _coerce_output(response)
            elapsed_ms = (time.perf_counter() - start) * 1000
            output.metrics = dict(output.metrics)
            output.metrics.setdefault("elapsed_ms", elapsed_ms)
            output.metrics.setdefault("model_calls", model_calls)
            output.metrics.setdefault("tool_calls", len(tool_call_records))
            output.metrics.setdefault("iteration_count", iteration)
            output.metrics.setdefault("tool_call_records", [record.__dict__ for record in tool_call_records])
            output.metrics.setdefault("touched_files", sorted(touched_files))
            _trace_log(
                trace,
                "worker_final",
                {
                    **trace_base,
                    "iteration": iteration,
                    "status": output.status.value,
                    "summary": output.summary,
                    "failure_signature": output.failure_signature,
                    "model_calls": model_calls,
                    "tool_calls": len(tool_call_records),
                    "elapsed_ms": elapsed_ms,
                },
            )
            return output

        elapsed_ms = (time.perf_counter() - start) * 1000
        _trace_log(
            trace,
            "worker_max_iterations",
            {
                **trace_base,
                "iteration": max_iterations,
                "max_iterations": max_iterations,
                "model_calls": model_calls,
                "tool_calls": len(tool_call_records),
                "elapsed_ms": elapsed_ms,
            },
        )
        return WorkerOutput(
            status=WorkerStatus.FAILURE,
            summary=f"worker exceeded max_iterations={max_iterations}",
            artifacts=[],
            metrics={
                "elapsed_ms": elapsed_ms,
                "model_calls": model_calls,
                "tool_calls": len(tool_call_records),
                "iteration_count": max_iterations,
                "tool_call_records": [record.__dict__ for record in tool_call_records],
            },
            next_actions=["Reduce tool calls, change strategy, or adjust planning."],
            failure_signature="worker-max-iterations",
        )


def _coerce_output(data: dict[str, Any]) -> WorkerOutput:
    raw_status = data.get("status", "PARTIAL")
    try:
        status = WorkerStatus(raw_status)
    except ValueError:
        status = WorkerStatus.PARTIAL
    known_keys = {"status", "summary", "artifacts", "metrics", "next_actions", "failure_signature", "tool_calls"}
    extra = {key: value for key, value in data.items() if key not in known_keys}
    return WorkerOutput(
        status=status,
        summary=data.get("summary", ""),
        artifacts=data.get("artifacts", []),
        metrics=data.get("metrics", {}),
        next_actions=data.get("next_actions", []),
        failure_signature=data.get("failure_signature", ""),
        data=extra,
    )


def _tool_descriptors(tools: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for name in sorted(tools):
        tool = tools[name]
        actions: list[str] = [
            attr
            for attr in dir(tool)
            if not attr.startswith("_") and callable(getattr(tool, attr, None))
        ]
        signatures = {}
        for action in actions:
            fn = getattr(tool, action, None)
            if not callable(fn):
                continue
            signatures[action] = _format_action_signature(action, fn)
        descriptors.append(
            {
                "name": name,
                "actions": sorted(set(actions)),
                "signatures": signatures,
            }
        )
    return descriptors


def _parse_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    if "status" in response:
        return []
    tool_calls = response.get("tool_calls")
    if not tool_calls:
        return []
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list")
    return [call for call in tool_calls if isinstance(call, dict)]


def _invoke_tool_call(tools: dict[str, Any], call: dict[str, Any]) -> ToolCallRecord:
    start = time.perf_counter()
    tool_name = str(call.get("tool", ""))
    action = str(call.get("action", ""))
    args = call.get("args", {}) or {}
    if not isinstance(args, dict):
        args = {}

    tool = tools.get(tool_name)
    if tool is None:
        return ToolCallRecord(
            tool_name=tool_name or "<missing>",
            ok=False,
            summary="unknown tool",
            data={"action": action, "args": args},
            elapsed_ms=_elapsed_ms(start),
            error="unknown tool",
        )

    fn = getattr(tool, action, None)
    if not callable(fn):
        return ToolCallRecord(
            tool_name=tool_name,
            ok=False,
            summary="unknown action",
            data={"action": action, "args": args},
            elapsed_ms=_elapsed_ms(start),
            error="unknown action",
        )

    try:
        result = fn(**args)
        if not isinstance(result, ToolResult):
            return ToolCallRecord(
                tool_name=tool_name,
                ok=True,
                summary="tool returned non-ToolResult",
                data={"result": result},
                elapsed_ms=_elapsed_ms(start),
                error=None,
            )
        return ToolCallRecord(
            tool_name=tool_name,
            ok=result.ok,
            summary=result.summary,
            data=result.data,
            elapsed_ms=result.elapsed_ms,
            error=result.error,
        )
    except Exception as exc:
        return ToolCallRecord(
            tool_name=tool_name,
            ok=False,
            summary="tool exception",
            data={"action": action, "args": args},
            elapsed_ms=_elapsed_ms(start),
            error=str(exc),
        )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _format_action_signature(action: str, fn: Any) -> str:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return f"{action}()"

    chunks: list[str] = []
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            chunks.append(f"*{param.name}")
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            chunks.append(f"**{param.name}")
            continue
        if param.default is inspect._empty:
            chunks.append(param.name)
        else:
            chunks.append(f"{param.name}=?")
    return f"{action}({', '.join(chunks)})"


def _format_tool_message(record: ToolCallRecord, *, max_chars: int | None = None) -> str:
    if max_chars is None:
        max_chars = 60_000 if record.tool_name == "file" else 20_000
    payload = {
        "ok": record.ok,
        "summary": record.summary,
        "data": record.data,
        "error": record.error,
    }
    max_str_candidates = (40_000, 20_000, 10_000, 4_000) if record.tool_name == "file" else (10_000, 4_000, 2_000, 1_000)
    for max_str in max_str_candidates:
        truncated = _truncate_jsonish(payload, max_str=max_str, max_list=100, max_depth=4)
        text = json.dumps(truncated, sort_keys=True)
        if len(text) <= max_chars:
            return text
    minimal = {"ok": record.ok, "summary": record.summary, "error": record.error, "data": {"_truncated": True}}
    return json.dumps(minimal, sort_keys=True)


def _touched_files_from_call(call: dict[str, Any]) -> set[str]:
    tool_name = str(call.get("tool") or "")
    action = str(call.get("action") or "")
    args = call.get("args") if isinstance(call.get("args"), dict) else {}

    if tool_name == "file" and action == "write":
        path = args.get("path")
        if isinstance(path, str) and path.strip():
            return {path.strip()}
        return set()

    if tool_name == "patch" and action == "apply":
        patch_text = args.get("patch_text")
        if isinstance(patch_text, str) and patch_text.strip():
            return _touched_files_from_patch_text(patch_text)
        return set()

    return set()


def _touched_files_from_patch_text(patch_text: str) -> set[str]:
    paths: set[str] = set()
    for raw_line in (patch_text or "").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for candidate in (parts[2], parts[3]):
                    normalized = _normalize_patch_path(candidate)
                    if normalized:
                        paths.add(normalized)
            continue
        if line.startswith(("+++ ", "--- ")):
            candidate = line.split(maxsplit=1)[1] if len(line.split(maxsplit=1)) == 2 else ""
            normalized = _normalize_patch_path(candidate)
            if normalized:
                paths.add(normalized)
    return paths


def _normalize_patch_path(candidate: str) -> str | None:
    candidate = (candidate or "").strip()
    if not candidate or candidate == "/dev/null":
        return None
    if candidate.startswith(("a/", "b/")):
        candidate = candidate[2:]
    if candidate.startswith(("\"a/", "\"b/")) and candidate.endswith("\""):
        candidate = candidate[3:-1]
    return candidate.strip() or None


def _truncate_jsonish(value: Any, *, max_str: int, max_list: int, max_depth: int) -> Any:
    if max_depth <= 0:
        return "<truncated>"
    if isinstance(value, str):
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


def _trace_log(trace: TraceLogger | None, event_type: str, payload: dict[str, Any]) -> None:
    if trace is None:
        return
    try:
        trace.log(event_type, payload)
    except Exception:
        return


def _trace_base(raw: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if not isinstance(key, str) or not key.strip():
            continue
        base[key.strip()] = _make_json_safe(_truncate_jsonish(value, max_str=2_000, max_list=50, max_depth=4))
    return base


def _response_meta(response: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if not isinstance(response, dict):
        return {"response_type": "invalid"}
    meta["response_keys"] = sorted(str(key) for key in response.keys())
    if "status" in response:
        meta["response_type"] = "final"
        meta["status"] = str(response.get("status") or "")
        meta["summary"] = str(response.get("summary") or "")[:500]
        meta["failure_signature"] = str(response.get("failure_signature") or "")[:200]
        return meta
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list):
        meta["response_type"] = "tool_calls"
        meta["tool_calls_count"] = len([call for call in tool_calls if isinstance(call, dict)])
    else:
        meta["response_type"] = "partial"
    return meta


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)
