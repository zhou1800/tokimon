"""Deterministic run metrics + dashboard generation."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

METRICS_SCHEMA_VERSION = "1.0"


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def normalize_step_metrics(
    *,
    step_id: str,
    attempt_id: int,
    status: str,
    artifacts: list[dict[str, Any]] | None = None,
    raw_metrics: dict[str, Any] | None = None,
    failure_signature: str | None = None,
) -> dict[str, Any]:
    raw_metrics = raw_metrics or {}
    artifacts = artifacts or []
    model_calls = _as_int(raw_metrics.get("model_calls"))
    tool_calls = _as_int(raw_metrics.get("tool_calls"))
    energy = model_calls + tool_calls if model_calls is not None and tool_calls is not None else None

    tool_errors = _as_int(raw_metrics.get("tool_errors"))
    if tool_errors is None:
        tool_errors = _tool_error_count(raw_metrics.get("tool_call_records"))

    touched_files_count = _as_int(raw_metrics.get("touched_files_count"))
    if touched_files_count is None:
        touched_files_count = _list_len(raw_metrics.get("touched_files"))

    artifact_count = _as_int(raw_metrics.get("artifact_count"))
    if artifact_count is None and isinstance(artifacts, list):
        artifact_count = len(artifacts)

    elapsed_ms = _as_float(raw_metrics.get("elapsed_ms"))
    if elapsed_ms is not None:
        elapsed_ms = round(elapsed_ms, 3)

    step_metrics: dict[str, Any] = {
        "step_id": str(step_id),
        "attempt_id": int(attempt_id),
        "status": str(status),
        "elapsed_ms": elapsed_ms,
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "energy": energy,
        "iteration_count": _as_int(raw_metrics.get("iteration_count")),
        "schema_repairs": _as_int(raw_metrics.get("schema_repairs")),
        "artifact_count": artifact_count,
        "touched_files_count": touched_files_count,
        "tool_errors": tool_errors,
        "failure_signature": str(failure_signature or ""),
    }
    return step_metrics


def build_run_metrics_payload(
    *,
    run_id: str,
    runner: str,
    wall_time_s: float | None,
    steps: list[dict[str, Any]],
    tests_passed: int | None = None,
    tests_failed: int | None = None,
) -> dict[str, Any]:
    wall_time_s = round(float(wall_time_s), 3) if wall_time_s is not None else None
    step_statuses = [str(step.get("status") or "") for step in (steps or [])]
    steps_by_status = {status: count for status, count in Counter(step_statuses).items() if status}
    model_calls_sum = _sum_int(step.get("model_calls") for step in (steps or []))
    tool_calls_sum = _sum_int(step.get("tool_calls") for step in (steps or []))
    energy_sum = model_calls_sum + tool_calls_sum if model_calls_sum is not None and tool_calls_sum is not None else None
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run": {
            "run_id": str(run_id),
            "runner": str(runner),
            "wall_time_s": wall_time_s,
            "model_calls": model_calls_sum,
            "tool_calls": tool_calls_sum,
            "energy": energy_sum,
            "steps_total": len(steps or []),
            "steps_by_status": steps_by_status,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
        },
        "steps": list(steps or []),
    }


def write_metrics_and_dashboard(reports_dir: Path, metrics_payload: dict[str, Any]) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"
    dashboard_path = reports_dir / "dashboard.html"
    metrics_path.write_text(stable_json_dumps(metrics_payload), encoding="utf-8")
    dashboard_path.write_text(generate_dashboard_html(metrics_payload), encoding="utf-8")
    return metrics_path, dashboard_path


def generate_dashboard_html(metrics_payload: dict[str, Any]) -> str:
    payload = metrics_payload or {}
    schema_version = str(payload.get("schema_version") or "")
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    steps_sorted = sorted(
        (step for step in steps if isinstance(step, dict)),
        key=lambda step: (str(step.get("step_id") or ""), int(step.get("attempt_id") or 0)),
    )

    embedded_json = _json_for_html_script(payload)
    rows = []
    for step in steps_sorted:
        rows.append(
            "<tr>"
            f"<td>{_escape_html(step.get('step_id'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('attempt_id'))}</td>"
            f"<td>{_escape_html(step.get('status'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('model_calls'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('tool_calls'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('energy'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('elapsed_ms'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('artifact_count'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('touched_files_count'))}</td>"
            f"<td class=\"num\">{_escape_html(step.get('tool_errors'))}</td>"
            f"<td>{_escape_html(step.get('failure_signature'))}</td>"
            "</tr>"
        )

    def run_field(key: str) -> str:
        return _escape_html(run.get(key))

    html = "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\" />",
            "  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />",
            "  <title>Tokimon Dashboard</title>",
            "  <style>",
            "    :root { color-scheme: light dark; }",
            "    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }",
            "    header { padding: 14px 16px; border-bottom: 1px solid #4443; }",
            "    main { padding: 16px; }",
            "    h1 { margin: 0 0 4px 0; font-size: 18px; }",
            "    .meta { opacity: 0.75; font-size: 12px; }",
            "    table { border-collapse: collapse; width: 100%; margin-top: 12px; }",
            "    th, td { border: 1px solid #4443; padding: 6px 8px; font-size: 12px; vertical-align: top; }",
            "    th { text-align: left; }",
            "    td.num { text-align: right; font-variant-numeric: tabular-nums; }",
            "    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }",
            "    details { margin-top: 12px; }",
            "    pre { margin: 8px 0 0 0; padding: 10px; border: 1px solid #4443; border-radius: 6px; overflow: auto; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <header>",
            "    <h1>Tokimon Run Dashboard</h1>",
            f"    <div class=\"meta\">schema_version={_escape_html(schema_version)} "
            f"run_id={run_field('run_id')} runner={run_field('runner')}</div>",
            "  </header>",
            "  <main>",
            "    <table>",
            "      <tr><th>Metric</th><th>Value</th></tr>",
            f"      <tr><td>wall_time_s</td><td class=\"num\">{run_field('wall_time_s')}</td></tr>",
            f"      <tr><td>model_calls</td><td class=\"num\">{run_field('model_calls')}</td></tr>",
            f"      <tr><td>tool_calls</td><td class=\"num\">{run_field('tool_calls')}</td></tr>",
            f"      <tr><td>energy</td><td class=\"num\">{run_field('energy')}</td></tr>",
            f"      <tr><td>steps_total</td><td class=\"num\">{run_field('steps_total')}</td></tr>",
            f"      <tr><td>tests_passed</td><td class=\"num\">{run_field('tests_passed')}</td></tr>",
            f"      <tr><td>tests_failed</td><td class=\"num\">{run_field('tests_failed')}</td></tr>",
            "    </table>",
            "    <h2>Steps</h2>",
            "    <table>",
            "      <tr>",
            "        <th>step_id</th><th>attempt</th><th>status</th><th>model</th><th>tool</th><th>energy</th>"
            "        <th>elapsed_ms</th><th>artifacts</th><th>touched</th><th>tool_errors</th><th>failure_signature</th>",
            "      </tr>",
            *rows,
            "    </table>",
            "    <details>",
            "      <summary>Embedded metrics.json</summary>",
            "      <pre><code id=\"tokimon-metrics-json\"></code></pre>",
            "    </details>",
            "  </main>",
            "  <script type=\"application/json\" id=\"tokimon-metrics\">",
            embedded_json,
            "  </script>",
            "  <script>",
            "    (function(){",
            "      var el = document.getElementById('tokimon-metrics');",
            "      var out = document.getElementById('tokimon-metrics-json');",
            "      if (!el || !out) return;",
            "      out.textContent = el.textContent.trim();",
            "    })();",
            "  </script>",
            "</body>",
            "</html>",
            "",
        ]
    )
    return html


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            parsed = float(cleaned)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _sum_int(values: Any) -> int | None:
    total = 0
    seen = False
    for value in values:
        item = _as_int(value)
        if item is None:
            continue
        total += item
        seen = True
    return total if seen else None


def _list_len(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    return None


def _tool_error_count(tool_call_records: Any) -> int | None:
    if not isinstance(tool_call_records, list):
        return None
    errors = 0
    for record in tool_call_records:
        if not isinstance(record, dict):
            continue
        if record.get("ok") is False:
            errors += 1
    return errors


def _escape_html(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&#39;")
    )


def _json_for_html_script(payload: Any) -> str:
    text = stable_json_dumps(payload).rstrip("\n")
    return text.replace("<", "\\u003c")

