from __future__ import annotations

import json
from pathlib import Path

from llm.client import MockLLMClient
from observability.reports import build_run_metrics_payload
from observability.reports import normalize_step_metrics
from observability.reports import stable_json_dumps
from observability.reports import write_metrics_and_dashboard
from runners.baseline import BaselineRunner
from runners.hierarchical import HierarchicalRunner


def test_write_metrics_and_dashboard_are_deterministic(tmp_path: Path) -> None:
    step_metrics = normalize_step_metrics(
        step_id="step-1",
        attempt_id=1,
        status="SUCCESS",
        artifacts=[{"path": "foo.txt", "kind": "file"}, {"path": "bar.txt", "kind": "file"}],
        raw_metrics={
            "elapsed_ms": 12.34567,
            "model_calls": 2,
            "tool_calls": 3,
            "iteration_count": 1,
            "schema_repairs": 0,
            "touched_files": ["foo.txt"],
            "tool_call_records": [{"ok": True}, {"ok": False}],
        },
        failure_signature="",
    )
    payload = build_run_metrics_payload(
        run_id="run-1",
        runner="baseline",
        wall_time_s=1.23456,
        steps=[step_metrics],
        tests_passed=10,
        tests_failed=0,
    )

    metrics_path, dashboard_path = write_metrics_and_dashboard(tmp_path, payload)
    metrics_text = metrics_path.read_text(encoding="utf-8")
    parsed = json.loads(metrics_text)
    assert metrics_text == stable_json_dumps(parsed)

    dashboard_text = dashboard_path.read_text(encoding="utf-8")
    assert "http://" not in dashboard_text
    assert "https://" not in dashboard_text
    assert "tokimon-metrics" in dashboard_text

    write_metrics_and_dashboard(tmp_path, payload)
    metrics_text_second = metrics_path.read_text(encoding="utf-8")
    dashboard_text_second = dashboard_path.read_text(encoding="utf-8")
    assert metrics_text_second == metrics_text
    assert dashboard_text_second == dashboard_text


def test_baseline_runner_persists_metrics_and_dashboard(tmp_path: Path) -> None:
    llm_client = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "summary": "ok",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
            }
        ]
    )
    runner = BaselineRunner(tmp_path, llm_client, base_dir=tmp_path / "runs")
    result = runner.run("goal", task_id="t1")
    metrics_path = result.run_context.reports_dir / "metrics.json"
    dashboard_path = result.run_context.reports_dir / "dashboard.html"
    assert metrics_path.exists()
    assert dashboard_path.exists()
    metrics_text = metrics_path.read_text(encoding="utf-8")
    metrics_json = json.loads(metrics_text)
    assert metrics_json["run"]["runner"] == "baseline"
    assert metrics_text == stable_json_dumps(metrics_json)


def test_hierarchical_runner_persists_metrics_and_dashboard(tmp_path: Path) -> None:
    llm_client = MockLLMClient(
        script=[
            {
                "status": "SUCCESS",
                "summary": "ok",
                "artifacts": [],
                "metrics": {},
                "next_actions": [],
                "failure_signature": "",
            }
        ]
    )
    runner = HierarchicalRunner(tmp_path, llm_client, base_dir=tmp_path / "runs")
    result = runner.run(
        "goal",
        task_steps=[{"id": "solve", "name": "Solve", "description": "desc", "worker": "Implementer", "inputs": {}}],
        task_id="t1",
        concurrency=1,
    )
    metrics_path = result.run_context.reports_dir / "metrics.json"
    dashboard_path = result.run_context.reports_dir / "dashboard.html"
    assert metrics_path.exists()
    assert dashboard_path.exists()
    metrics_text = metrics_path.read_text(encoding="utf-8")
    metrics_json = json.loads(metrics_text)
    assert metrics_json["run"]["runner"] == "hierarchical"
    assert metrics_text == stable_json_dumps(metrics_json)
