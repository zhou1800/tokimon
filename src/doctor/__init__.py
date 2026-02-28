"""Doctor command helpers (health checks + safe repairs)."""

from .runner import DoctorCheck, DoctorDeps, DoctorReport, default_deps, render_human, report_to_json_dict, run_doctor

__all__ = [
    "DoctorCheck",
    "DoctorDeps",
    "DoctorReport",
    "default_deps",
    "render_human",
    "report_to_json_dict",
    "run_doctor",
]
