"""sofaopt — parallel CMA-ES optimization + live dashboard for SOFA scenes.

Quick start (in your project's ``project.py``)::

    from sofaopt import SofaOptProject, ParamSpec, TestSpec

    PROJECT = SofaOptProject(
        name="my_robot",
        work_dir=Path(__file__).parent,
        params=[ParamSpec("stiffness", "float", 1e3, 1e6, 1e4)],
        tests=[TestSpec("reach", scene_file="scenes/reach.py", max_score=100)],
        runsofa_exe=Path(r"C:/sofa/bin/runSofa.exe"),
        sofa_env={"SOFA_ROOT": r"C:/sofa"},
    )

Then run headless::

    from sofaopt import run_optimization
    run_optimization(PROJECT)

or open the dashboard::

    from sofaopt import launch_dashboard
    launch_dashboard(PROJECT, port=8050)

Inside a scene, read the trial and report a score::

    from sofaopt.scene import open_trial
    trial = open_trial()              # params + per-run metadata from env
    stiffness = trial.params["stiffness"]
    ...
    trial.write_score(87.3, reason="held 4.2 s")
"""

from __future__ import annotations

from .project import (
    ParamSpec,
    PrepareHook,
    SofaOptProject,
    TestSpec,
    TrialPrep,
    param_specs_from_dataclass,
)

__all__ = [
    "DashboardTab",
    "ParamSpec",
    "PrepareHook",
    "SofaOptProject",
    "TestSpec",
    "TrialPrep",
    "archive_run",
    "cleanup_trial_recordings",
    "delete_archive",
    "generate_selected_videos",
    "generate_summary_video",
    "generate_trial_video",
    "launch_dashboard",
    "list_archives",
    "param_specs_from_dataclass",
    "restore_archive",
    "run_optimization",
    "run_sensitivity_analysis",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy: keep optuna/dash out of the import path until actually needed.
    if name == "run_optimization":
        from .core.orchestrator import run_optimization

        return run_optimization
    if name == "launch_dashboard":
        from .dashboard.app import launch_dashboard

        return launch_dashboard
    if name == "run_sensitivity_analysis":
        from .sensitivity import run_sensitivity_analysis

        return run_sensitivity_analysis
    if name in ("archive_run", "list_archives", "restore_archive", "delete_archive"):
        from .core import archive

        return getattr(archive, name)
    if name == "generate_trial_video":
        from .video import generate_trial_video

        return generate_trial_video
    if name == "generate_selected_videos":
        from .video import generate_selected_videos

        return generate_selected_videos
    if name == "generate_summary_video":
        from .video import generate_summary_video

        return generate_summary_video
    if name == "cleanup_trial_recordings":
        from .video import cleanup_trial_recordings

        return cleanup_trial_recordings
    if name == "DashboardTab":
        from .dashboard.app import DashboardTab

        return DashboardTab
    raise AttributeError(f"module 'sofaopt' has no attribute {name!r}")
