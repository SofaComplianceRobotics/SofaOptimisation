"""Active-project context for the dashboard.

The dashboard modules were originally wired to module-global lab paths. Here we
hold the live :class:`~sofaopt.project.SofaOptProject` (set once by
``launch_dashboard``) and expose the paths / catalog / constants every module
reads, so nothing in the dashboard hardcodes a project.
"""

from __future__ import annotations

from pathlib import Path

from sofaopt.project import SofaOptProject, TestSpec

# UI constants (project-independent).
TOP_X = 10  # leaderboard length
CENTERED_AVG_HALF_WINDOW = 10  # rolling-average half window
LIVE_REFRESH_SECONDS = 2.0  # live polling interval
SCORE_AGGREGATION = "mean"

_PROJECT: SofaOptProject | None = None


def set_project(project: SofaOptProject) -> None:
    global _PROJECT
    _PROJECT = project


def project() -> SofaOptProject:
    if _PROJECT is None:
        raise RuntimeError("Dashboard context not initialized — call set_project().")
    return _PROJECT


def trials_dir() -> Path:
    return project().trials_dir


def progress_file() -> Path:
    return project().progress_file


def catalog() -> dict[str, TestSpec]:
    """Test catalog as the dashboard expects it: ``{name: TestSpec}``."""
    return {t.name: t for t in project().tests}


def hard_fail_score() -> float:
    return project().hard_fail_score
