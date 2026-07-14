"""Liver multi-load-case material identification — the medium real-SOFA tier.

Ported from SOFA's shipped ``examples/Demos/liver.scn``. Three material
parameters (Young's modulus, Poisson ratio, mass density); the objective is to
reproduce the liver's **settled shape under two patient orientations** (gravity
directions), each measured once from the reference material shipped in the
``.scn`` — surgical-registration-style inverse identification.

Why it matters to the framework (beyond being a demo): it is the reference
example for the **multi-test weighted scoring pipeline** — each trial runs TWO
tests (load cases) whose normalized scores are weight-combined — on a real
tetrahedral organ mesh (181 nodes, 3-tier cost: noticeably heavier than tetra,
far lighter than caduceus), so the parallel launcher earns its keep.

Identifiability note: gravity is a body force ~ density, so
(young_modulus, mass_density) form the same E/rho ridge as the tetra example
— both load cases scale identically along it. That is intentional (it is the
physics); poisson_ratio is what the second orientation helps pin down.

Reference/target provenance: ``targets.json`` — measured via the python runner
from the reference material; command in the README.
"""

from __future__ import annotations

import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent

# Reference material the targets were measured from (the liver.scn values).
# The ParamSpec defaults below are deliberately OFF these values.
REFERENCE = {"young_modulus": 3000.0, "poisson_ratio": 0.30, "mass_density": 1.0}

PROJECT = SofaOptProject(
    name="liver_registration",
    title="Liver shape registration — sofaopt medium SOFA tier (multi-test)",
    work_dir=HERE,
    params=[
        # name, type, min, max, default (defaults off-target on purpose)
        ParamSpec("young_modulus", "float", 500.0, 10000.0, 8000.0),
        ParamSpec("poisson_ratio", "float", 0.05, 0.45, 0.10),
        ParamSpec("mass_density", "float", 0.2, 3.0, 1.0),
    ],
    tests=[
        TestSpec(
            "supine",
            scene_file=HERE / "scene.py",
            label="Supine settle match",
            description="reproduce the settled liver shape, gravity -y",
            max_score=100.0,
            weight=1.0,
        ),
        TestSpec(
            "lateral",
            scene_file=HERE / "scene.py",
            label="Lateral settle match",
            description="reproduce the settled liver shape, 45-degree roll",
            max_score=100.0,
            weight=1.0,
        ),
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
    n_generations=25,
    cmaes_sigma0=0.4,
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
