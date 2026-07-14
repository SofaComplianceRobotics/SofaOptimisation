"""Liver elastography — the in-depth study platform (8 params, 3 tests).

The liver's stiffness is a REGIONAL FIELD (6 committed regions, per-element
Young's modulus) instead of a scalar: the reference is homogeneous E=3000 with
one stiff lesion (region 5, E=9000), and the optimizer must localize it from
settled shapes under three patient orientations — inverse elastography.

Why this example exists (the others are narrow regression assets):

- **8 parameters with graded sensitivity** — region 0 hugs the fixed nodes
  (nearly unobservable), region 5 is deep in the free bulk (strong signal) —
  the regime where optimizer features (restarts, seeding, covariance) actually
  differentiate, on a real SOFA scene at ~1 s per launch.
- **The number of load cases is a real experimental variable**: different
  orientations load different regions, so identifiability vs number-of-tests
  can be studied quantitatively (run with 1, 2 or 3 tests selected).
- **Sensitivity tooling gets ground truth**: the dashboard's importance
  analysis should rank the lesion region high and region 0 low.
- A natural multi-fidelity axis (settle-step prefix / fewer load cases) for
  the `investigate` branch.

Reference/target provenance: ``targets.json`` + ``regions.json`` — measured
via the python runner; commands in the README.
"""

from __future__ import annotations

import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent

N_REGIONS = 6
LESION_REGION = 5  # deepest region (farthest from the fixed nodes; regions.json)

# Reference stiffness field the targets were measured from: homogeneous E=3000
# with one SOFT lesion (the standard elastography phantom; measured to give ~2x
# the shape signal of a stiff one — a stiff region saturates toward rigid, a
# soft one keeps deforming). The ParamSpec defaults below are the natural blind
# start: a uniform mid-range field (no lesion assumed).
REFERENCE = {
    **{f"young_region_{k}": 3000.0 for k in range(N_REGIONS)},
    f"young_region_{LESION_REGION}": 1000.0,
    "poisson_ratio": 0.30,
    "mass_density": 1.0,
}

_CASE_DESCRIPTIONS = {
    "supine": "settled shape, gravity -y",
    "lateral": "settled shape, 45-degree roll",
    "tilt": "settled shape, 45-degree pitch",
}

PROJECT = SofaOptProject(
    name="liver_elastography",
    title="Liver elastography — sofaopt in-depth study platform (8 params)",
    work_dir=HERE,
    params=[
        *[
            ParamSpec(f"young_region_{k}", "float", 500.0, 12000.0, 5000.0)
            for k in range(N_REGIONS)
        ],
        ParamSpec("poisson_ratio", "float", 0.05, 0.45, 0.10),
        ParamSpec("mass_density", "float", 0.2, 3.0, 1.0),
    ],
    tests=[
        TestSpec(
            case,
            scene_file=HERE / "scene.py",
            label=f"{case.capitalize()} settle match",
            description=desc,
            max_score=100.0,
            weight=1.0,
        )
        for case, desc in _CASE_DESCRIPTIONS.items()
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
    n_generations=60,          # 8-dim search needs a real budget
    cmaes_sigma0=0.3,
    cmaes_restarts=4,          # the study platform ships the studied config
    restart_on_convergence=True,
    warm_restarts=True,
    seed_sampler="sobol",
    sofa_realtime_timeout=120.0,
    run_script=HERE / "run.py",
)

_VARIANTS = {"default": PROJECT}


def get_variant(name: str) -> SofaOptProject:
    try:
        return _VARIANTS[name]
    except KeyError:
        raise SystemExit(f"Unknown variant '{name}'. Have: {sorted(_VARIANTS)}") from None
