"""Worker agent implementation."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from typing import Any

from agents.outputs import WorkerOutput
from agents.prompts import build_system_prompt
from flow_types import ToolCallRecord, WorkerStatus
from llm.client import LLMClient
from policy.dangerous_tools import is_side_effectful, tool_risk
from replay import ReplayRecorder
from tools.base import ToolResult
from tracing import TraceLogger
from workflow.schema import (
    SchemaValidationError,
    SchemaViolation,
    WORKER_FINAL_OUTPUT_SCHEMA,
    validate_schema,
)


class Worker:
    def __init__(self, role: str, llm_client: LLMClient, tools: dict[str, Any]) -> None:
        self.role = role
        self.llm_client = llm_client
        self.tools = tools

    def run(
        self,
        goal: str,
        step_id: str,
        inputs: dict[str, Any],
        memory: list[str],
        max_iterations: int = 20,
        *,
        trace: TraceLogger | None = None,
        trace_context: dict[str, Any] | None = None,
        replay_recorder: ReplayRecorder | None = None,
    ) -> WorkerOutput:
        def finalize(output: WorkerOutput) -> WorkerOutput:
            if replay_recorder is None:
                return output
            metrics_subset: dict[str, Any] = {}
            if isinstance(output.metrics, dict):
                for key in ("elapsed_ms", "model_calls", "tool_calls", "iteration_count", "schema_repairs"):
                    if key in output.metrics:
                        metrics_subset[key] = output.metrics.get(key)
            replay_recorder.record_final(
                {
                    "status": output.status.value,
                    "summary": output.summary,
                    "failure_signature": str(output.failure_signature or ""),
                    "metrics": metrics_subset,
                }
            )
            return output

        start = time.perf_counter()
        model_calls = 0
        tool_call_records: list[ToolCallRecord] = []
        tool_call_cache: dict[str, ToolCallRecord] = {}
        touched_files: set[str] = set()
        schema_repairs = 0
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
            if replay_recorder is not None:
                replay_recorder.record_model_response(response)

            try:
                tool_calls = _parse_tool_calls(response)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _trace_log(
                    trace,
                    "worker_invalid_tool_calls",
                    {**trace_base, "iteration": iteration, "error": str(exc)},
                )
                return finalize(
                    WorkerOutput(
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
                )
            if tool_calls:
                for call in tool_calls:
                    tool_name = str(call.get("tool", ""))
                    action = str(call.get("action", ""))
                    raw_args = call.get("args", {}) or {}
                    args = raw_args if isinstance(raw_args, dict) else {}
                    call_id = _coerce_tool_call_id(call)
                    policy_decision = _tool_policy_decision(tool_name, action, args)
                    cache_key = _tool_call_cache_key(tool_name, action, args) if _is_side_effectful_tool_call(tool_name, action) else None
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
                    if cache_key is not None and cache_key in tool_call_cache:
                        cached = tool_call_cache[cache_key]
                        record = ToolCallRecord(
                            tool_name=cached.tool_name,
                            call_id=call_id,
                            policy_decision=policy_decision,
                            cached=True,
                            ok=cached.ok,
                            summary=cached.summary,
                            data=cached.data,
                            elapsed_ms=cached.elapsed_ms,
                            error=cached.error,
                        )
                    else:
                        record = _invoke_tool_call(self.tools, call, policy_decision=policy_decision)
                        if cache_key is not None:
                            tool_call_cache[cache_key] = record
                    tool_call_records.append(record)
                    if replay_recorder is not None:
                        replay_recorder.record_tool_invocation(call, record)
                    _trace_log(
                        trace,
                        "worker_tool_result",
                        {
                            **trace_base,
                            "iteration": iteration,
                            "tool_name": record.tool_name,
                            "tool_call_id": record.call_id,
                            "ok": record.ok,
                            "summary": record.summary,
                            "elapsed_ms": record.elapsed_ms,
                            "error": record.error,
                            "cached": record.cached,
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

            try:
                _validate_worker_final_response(response)
            except SchemaValidationError as exc:
                if schema_repairs >= 2:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    failure_signature = _schema_failure_signature(exc)
                    _trace_log(
                        trace,
                        "worker_output_schema_invalid",
                        {
                            **trace_base,
                            "iteration": iteration,
                            "repairs": schema_repairs,
                            "failure_signature": failure_signature,
                            "error": str(exc),
                        },
                    )
                    return finalize(
                        WorkerOutput(
                        status=WorkerStatus.FAILURE,
                        summary="worker produced invalid structured output (schema validation failed)",
                        artifacts=[],
                        metrics={
                            "elapsed_ms": elapsed_ms,
                            "model_calls": model_calls,
                            "tool_calls": len(tool_call_records),
                            "iteration_count": iteration,
                            "tool_call_records": [record.__dict__ for record in tool_call_records],
                            "schema_repairs": schema_repairs,
                        },
                        next_actions=[
                            "Return a schema-valid final JSON object (status/summary/artifacts/metrics/next_actions/failure_signature).",
                        ],
                        failure_signature=failure_signature,
                        )
                    )
                schema_repairs += 1
                _trace_log(
                    trace,
                    "worker_output_schema_repair",
                    {
                        **trace_base,
                        "iteration": iteration,
                        "repairs": schema_repairs,
                        "error": str(exc),
                    },
                )
                messages.extend(_schema_repair_messages(response, exc, remaining=2 - schema_repairs))
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
            output.metrics.setdefault("schema_repairs", schema_repairs)
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
                    "schema_repairs": schema_repairs,
                },
            )
            return finalize(output)

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
                "schema_repairs": schema_repairs,
            },
        )
        return finalize(
            WorkerOutput(
            status=WorkerStatus.FAILURE,
            summary=f"worker exceeded max_iterations={max_iterations}",
            artifacts=[],
            metrics={
                "elapsed_ms": elapsed_ms,
                "model_calls": model_calls,
                "tool_calls": len(tool_call_records),
                "iteration_count": max_iterations,
                "tool_call_records": [record.__dict__ for record in tool_call_records],
                "schema_repairs": schema_repairs,
            },
            next_actions=["Reduce tool calls, change strategy, or adjust planning."],
            failure_signature="worker-max-iterations",
            )
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


def _coerce_tool_call_id(call: dict[str, Any]) -> str | None:
    raw = call.get("call_id")
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    return cleaned or None


def _tool_call_cache_key(tool_name: str, action: str, args: dict[str, Any]) -> str:
    payload = {"tool": str(tool_name or ""), "action": str(action or ""), "args": _make_json_safe(args)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_side_effectful_tool_call(tool_name: str, action: str) -> bool:
    return is_side_effectful(tool_name, action)


def _tool_policy_decision(tool_name: str, action: str, args: dict[str, Any]) -> dict[str, Any]:
    _ = args
    risk = tool_risk(tool_name, action)
    risk_tier = risk.risk_tier if risk is not None else "low"
    requires_approval = bool(risk.requires_approval) if risk is not None else False
    reason = "default allow"
    if risk is not None and str(risk.notes or "").strip():
        notes = str(risk.notes).strip()
        reason = f"default allow ({notes[:200]})"
    elif risk_tier == "high":
        reason = "default allow (side-effectful)"
    elif risk_tier == "medium":
        reason = "default allow (external interaction)"
    return {
        "decision": "allow",
        "risk_tier": risk_tier,
        "reason": reason,
        "policy_id": "default-v1",
        "requires_approval": requires_approval,
    }


def _invoke_tool_call(tools: dict[str, Any], call: dict[str, Any], *, policy_decision: dict[str, Any]) -> ToolCallRecord:
    start = time.perf_counter()
    tool_name = str(call.get("tool", ""))
    call_id = _coerce_tool_call_id(call)
    action = str(call.get("action", ""))
    args = call.get("args", {}) or {}
    if not isinstance(args, dict):
        args = {}

    tool = tools.get(tool_name)
    if tool is None:
        return ToolCallRecord(
            tool_name=tool_name or "<missing>",
            call_id=call_id,
            policy_decision=policy_decision,
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
            call_id=call_id,
            policy_decision=policy_decision,
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
                call_id=call_id,
                policy_decision=policy_decision,
                ok=True,
                summary="tool returned non-ToolResult",
                data={"result": result},
                elapsed_ms=_elapsed_ms(start),
                error=None,
            )
        return ToolCallRecord(
            tool_name=tool_name,
            call_id=call_id,
            policy_decision=policy_decision,
            ok=result.ok,
            summary=result.summary,
            data=result.data,
            elapsed_ms=result.elapsed_ms,
            error=result.error,
        )
    except Exception as exc:
        return ToolCallRecord(
            tool_name=tool_name,
            call_id=call_id,
            policy_decision=policy_decision,
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
        "call_id": record.call_id,
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
    minimal = {
        "call_id": record.call_id,
        "ok": record.ok,
        "summary": record.summary,
        "error": record.error,
        "data": {"_truncated": True},
    }
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


def _validate_worker_final_response(response: dict[str, Any]) -> None:
    if not isinstance(response, dict):
        raise SchemaValidationError(
            [SchemaViolation(code="type_mismatch", path="$", message="response must be object")]
        )
    if "tool_calls" in response and "status" in response:
        raise SchemaValidationError(
            [SchemaViolation(code="mutually_exclusive", path="$", message="cannot include both status and tool_calls")]
        )
    validate_schema(response, WORKER_FINAL_OUTPUT_SCHEMA)


def _schema_failure_signature(exc: SchemaValidationError) -> str:
    violation = exc.violations[0] if getattr(exc, "violations", None) else None
    if violation is None:
        return "worker-output-schema-invalid"
    path = (violation.path or "$").replace(" ", "")
    code = (violation.code or "invalid").replace(" ", "")
    return f"worker-output-schema-invalid:{code}:{path}"


def _schema_repair_messages(
    response: dict[str, Any],
    exc: SchemaValidationError,
    *,
    remaining: int,
) -> list[dict[str, str]]:
    truncated_response = _truncate_jsonish(response, max_str=6_000, max_list=100, max_depth=4)
    schema_text = json.dumps(WORKER_FINAL_OUTPUT_SCHEMA, sort_keys=True)
    error_text = str(exc)[:2_000]
    instructions = "\n".join(
        [
            "Your last response failed schema validation.",
            f"Schema: {schema_text}",
            f"Validation error: {error_text}",
            f"Repairs remaining (including this one): {max(remaining, 0)}",
            "",
            "Return ONLY a single JSON object that validates against the schema.",
            "Do not include tool_calls in the response unless you are requesting a tool call (in which case omit status).",
        ]
    )
    return [
        {"role": "assistant", "content": json.dumps(truncated_response, sort_keys=True)},
        {"role": "user", "content": instructions},
    ]
