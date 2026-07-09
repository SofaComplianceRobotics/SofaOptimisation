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

This one uses a **prepare hook**: each trial generates a scaled cube `.obj`
(the "model"), demonstrating the geometry path. A project that only tunes scene
quantities (stiffness, mass, gains) needs no hook — the scene just reads params.
"""

from __future__ import annotations

import os
from pathlib import Path

import dataclasses

from sofaopt import ParamSpec, SofaOptProject, TestSpec, TrialPrep

HERE = Path(__file__).resolve().parent


def _write_cube_obj(path: Path, size: float) -> None:
    """Write a minimal axis-aligned cube OBJ of edge ``size``, centered at origin."""
    h = size / 2.0
    verts = [
        (-h, -h, -h),
        (h, -h, -h),
        (h, h, -h),
        (-h, h, -h),
        (-h, -h, h),
        (h, -h, h),
        (h, h, h),
        (-h, h, h),
    ]
    faces = [  # 1-indexed triangles
        (1, 2, 3),
        (1, 3, 4),
        (5, 8, 7),
        (5, 7, 6),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
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
        ParamSpec(
            "cube_size", "float", 5.0, 50.0, 10.0
        ),  # name, type, min, max, default
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
    sofa_env={
        k: os.environ[k]
        for k in ("SOFA_ROOT", "SOFAPYTHON3_ROOT", "PYTHONPATH")
        if k in os.environ
    },
    gui_mode="batch",
    prepare_trial=_prepare,
    n_parallel=4,
    n_generations=40,
    cmaes_startup_trials=8,
    sofa_realtime_timeout=60.0,
    run_script=HERE / "run.py",
)

# ---------------------------------------------------------------------------
# Variant projects — switch with: python run.py --variant <name>
# ---------------------------------------------------------------------------

# TPE (Bayesian) sampler — often converges faster than CMA-ES with ≤ 5 parameters.
PROJECT_TPE = dataclasses.replace(PROJECT, name="cube_drop_tpe", sampler="tpe")

# GP Bayesian optimization — the sample-efficient choice when each simulation
# is expensive and the searched dimensionality is small (< ~20).  Overkill for
# this toy, but demonstrates the wiring (see docs/optimization-guide.md §2).
PROJECT_GP = dataclasses.replace(PROJECT, name="cube_drop_gp", sampler="gp")

# Sobol' space-filling startup — the first `cmaes_startup_trials` candidates
# come from a scrambled Sobol' (QMC) design instead of uniform random, so the
# exploration phase covers the space (and parameter interactions) evenly.
# Change seed_sampler_seed for an independent but equally balanced design.
PROJECT_SOBOL = dataclasses.replace(
    PROJECT, name="cube_drop_sobol", seed_sampler="sobol"
)

# Python in-process runner — scene imports Sofa directly instead of runSofa.
# Gives scene code access to Sofa.Core / Sofa.Simulation while keeping
# subprocess isolation.  Useful when the scene needs to call Sofa.Simulation.reset()
# or read constraint matrices between goals.
PROJECT_PYTHON_RUNNER = dataclasses.replace(
    PROJECT,
    name="cube_drop_python",
    runner="python",
    record_frames=True,
    record_frame_skip=16,
    record_frame_size=(640, 480),
)

# Multi-objective (NSGA-II) — fall_fast vs compact.  The cube should fall
# fast (wants large size + high mass) but also be compact (wants small size).
# No single solution wins both; NSGA-II returns the full Pareto trade-off front.
_PARETO_TESTS = [
    TestSpec(
        "fall_fast",
        scene_file=HERE / "scene.py",
        label="Fall speed",
        description="reach the floor as soon as possible (big + heavy)",
        max_score=100.0,
        direction="maximize",
    ),
    TestSpec(
        "compact",
        scene_file=HERE / "scene_compact.py",
        label="Compact",
        description="prefer a small cube (competes with fall_fast)",
        max_score=100.0,
        direction="maximize",
    ),
]

PROJECT_MULTI_OBJ = dataclasses.replace(
    PROJECT,
    name="cube_drop_pareto",
    tests=_PARETO_TESTS,
    multi_objective=True,
    n_parallel=5,
)

# Same Pareto setup but using the Python in-process runner instead of runSofa.
# Use this to verify multi-objective + python runner interoperate correctly and
# to compare wall-clock time per generation vs the runSofa baseline above.
PROJECT_MULTI_OBJ_PYTHON = dataclasses.replace(
    PROJECT_MULTI_OBJ,
    name="cube_drop_pareto_python",
    runner="python",
)
