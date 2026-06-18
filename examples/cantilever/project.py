"""Toy sofaopt project: tune a soft cantilever beam's material.

This is the smallest possible *real* project: two material parameters, one
scene, one score, and **no prepare hook** (nothing is generated — the scene
just reads the sampled params). Copy this directory and replace the params,
the scene, and the SOFA paths with your own to bootstrap a new project.
"""

from __future__ import annotations

import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent


def _sofa_env() -> dict:
    """Forward whatever SOFA env the shell already has (works with any build)."""
    keys = ("SOFA_ROOT", "SOFAPYTHON3_ROOT", "PYTHONPATH")
    return {k: os.environ[k] for k in keys if k in os.environ}


PROJECT = SofaOptProject(
    name="cantilever",
    title="Cantilever — sofaopt example",
    work_dir=HERE,
    # Tune the beam's material so its tip settles to a target deflection.
    params=[
        ParamSpec("young_modulus", "float", 1.0e3, 5.0e5, 5.0e4),
        ParamSpec("poisson_ratio", "float", 0.0, 0.45, 0.30),
    ],
    tests=[
        TestSpec(
            "tip_target",
            scene_file=HERE / "scene.py",
            label="Tip target",
            description="match a target tip deflection under gravity",
            max_score=100.0,
            run_count=1,
            default_selected=True,
        )
    ],
    # Point this at any runSofa that has SofaPython3 (env override recommended).
    runsofa_exe=Path(os.environ.get("RUNSOFA_EXE", "runSofa")),
    sofa_env=_sofa_env(),
    gui_mode="batch",
    # Small + quick so the example finishes in a couple of minutes.
    n_parallel=4,
    n_generations=8,
    cmaes_startup_trials=8,
    sofa_realtime_timeout=60.0,
    # Lets the dashboard's Run button start a headless optimization.
    run_script=HERE / "run.py",
)
