"""Toy "does the optimizer climb the hill?" demo.

Two parameters — a cube's **size** and **mass** — and one rule: the sooner the
cube reaches the ground, the better. The optimum is obvious by construction, so
it's a quick sanity check that the optimizer, runner, scoring and dashboard all
work end to end:

- a **bigger** cube (spawned at a fixed center height) has its bottom face lower,
  so it reaches the floor sooner;
- a **heavier** cube falls faster against a fixed upward "buoyancy" force
  (net accel ``g - F/m``), and a too-light cube floats and scores 0.

So you should watch the optimizer drive **both size and mass up**.

Unlike the cantilever example, this one uses a **prepare hook**: each trial
generates a scaled cube `.obj` (the "model"), demonstrating the geometry path.
"""

from __future__ import annotations

import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec, TrialPrep

HERE = Path(__file__).resolve().parent


def _write_cube_obj(path: Path, size: float) -> None:
    """Write a minimal axis-aligned cube OBJ of edge ``size``, centered at origin."""
    h = size / 2.0
    verts = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
    ]
    faces = [  # 1-indexed triangles
        (1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1),
    ]
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare(params, trial_dir):
    """Generate the per-trial cube model and hand its path to the scene."""
    mesh = trial_dir / "cube.obj"
    _write_cube_obj(mesh, float(params["cube_size"]))
    return TrialPrep(env={"OPT_CUBE_MESH": str(mesh)}, cleanup=[mesh])


PROJECT = SofaOptProject(
    name="cube_drop",
    title="Cube drop — sofaopt demo",
    work_dir=HERE,
    params=[
        ParamSpec("cube_size", "float", 5.0, 50.0, 10.0),
        ParamSpec("cube_mass", "float", 0.5, 50.0, 1.0),
    ],
    tests=[
        TestSpec(
            "fall",
            scene_file=HERE / "scene.py",
            label="Fall time",
            description="reach the floor as soon as possible",
            max_score=100.0,
            default_selected=True,
        )
    ],
    runsofa_exe=Path(os.environ.get("RUNSOFA_EXE", "runSofa")),
    sofa_env={k: os.environ[k] for k in ("SOFA_ROOT", "SOFAPYTHON3_ROOT", "PYTHONPATH") if k in os.environ},
    gui_mode="batch",
    prepare_trial=_prepare,
    n_parallel=4,
    n_generations=12,
    cmaes_startup_trials=8,
    sofa_realtime_timeout=60.0,
    run_script=HERE / "run.py",
)
