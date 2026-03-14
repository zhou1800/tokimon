from pathlib import Path

from benchmarks.harness import EvaluationHarness


def test_harness_runs_suite(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    harness = EvaluationHarness(repo_root, runs_dir=tmp_path / "runs")
    run_context = harness.run_suite()
    report_path = run_context.reports_dir / "suite-report.json"
    assert report_path.exists()
    report = report_path.read_text()
    assert "tasks" in report
