"""Single-tetra inverse material identification — the light real-SOFA tier.

Ported from SOFA's shipped ``examples/Demos/oneTetrahedron.scn``. Three material
parameters (Young's modulus, Poisson ratio, mass); the objective is to reproduce
a **target apex settle position** measured once from reference material values —
the classic inverse-identification problem, in the smallest possible FEM (one
tetrahedron, 4 nodes → near-instant trials).

Why it matters to the framework (beyond being a demo): it exercises the genuine
SOFA path (subprocess, settle self-stop, trial_state contract) at negligible
cost, and its (young_modulus, total_mass) pair forms an approximate
identifiability *ridge* (static sag ~ load/stiffness) — the structure CMA-ES's
covariance is built to learn. The ``noisy`` variant adds sensor-style score
noise, turning on racing (``run_count_min``).

Reference/target provenance: see TARGET_APEX in ``scene.py`` — measured
2026-07-13 with the reference material (young_modulus=10, poisson_ratio=0.30,
total_mass=2.0) via the python runner; command in the README.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent

# Reference material the target was measured from. The ParamSpec defaults below
# are deliberately OFF these values so the optimizer has an identification task.
REFERENCE = {"young_modulus": 10.0, "poisson_ratio": 0.30, "total_mass": 2.0}

PROJECT = SofaOptProject(
    name="tetra_material",
    title="Single-tetra material identification — sofaopt light SOFA tier",
    work_dir=HERE,
    params=[
        # name, type, min, max, default (defaults off-target on purpose)
        ParamSpec("young_modulus", "float", 2.0, 50.0, 30.0),
        ParamSpec("poisson_ratio", "float", 0.05, 0.45, 0.10),
        ParamSpec("total_mass", "float", 0.5, 8.0, 5.0),
    ],
    tests=[
        TestSpec(
            "settle",
            scene_file=HERE / "scene.py",
            label="Apex settle match",
            description="reproduce the target apex settle position (inverse material ID)",
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
    runner="python",
    n_parallel=4,
    n_generations=30,
    cmaes_sigma0=0.5,
    seed_sampler="sobol",
    sofa_realtime_timeout=60.0,
    run_script=HERE / "run.py",
)

# ---------------------------------------------------------------------------
# Variants — python run.py --variant <name>
# ---------------------------------------------------------------------------

# Noisy objective -> racing: 5 repeats declared, start with 2, add more only
# while the candidate's CI still overlaps the incumbent.
_NOISY_TEST = TestSpec(
    "settle",
    scene_file=HERE / "scene.py",
    label="Apex settle match (noisy)",
    description="noisy identification; repeats averaged, raced on CI overlap",
    max_score=100.0,
    run_count=5,
    run_count_min=2,
    score_aggregation="mean",
)
PROJECT_NOISY = dataclasses.replace(
    PROJECT,
    name="tetra_material_noisy",
    tests=[_NOISY_TEST],
    sofa_env={**PROJECT.sofa_env, "OPT_TETRA_NOISE": "4.0"},
)

_VARIANTS = {"default": PROJECT, "noisy": PROJECT_NOISY}


def get_variant(name: str) -> SofaOptProject:
    try:
        return _VARIANTS[name]
    except KeyError:
        raise SystemExit(f"Unknown variant '{name}'. Have: {sorted(_VARIANTS)}") from None
