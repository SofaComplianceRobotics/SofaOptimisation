"""Cable-driven soft finger actuated reach — the soft-robot real-SOFA tier.

Ported from the SoftRobots plugin's shipped finger part. Three parameters
spanning control, material and design: the commanded cable displacement, the
silicone Young's modulus, and the cable pull-point height. The objective is
to land the fingertip on a **target steady-state position measured once from
reference values** — an actuated-reach calibration.

Why it matters to the framework (beyond being a demo): it is the first
example that **drives an actuator** the way a production pipeline does (ramped
command at a fixed rate, steady-state detection at the real interface — the
mechanical dofs), and it exercises the Lagrangian constraint stack
(FreeMotionAnimationLoop + CableConstraint) and third-party plugin loading
(SoftRobots + stlib3 prefabs) under headless parallel trials. The
(cable_displacement, young_modulus, pull_point_y) triple is redundant by
design — many combinations reach the same tip — the covariance structure
CMA-ES is built to learn.

Reference/target provenance: see TARGET_TIP in ``scene.py`` — measured via
the python runner from the reference configuration; command in the README.
"""

from __future__ import annotations

import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent

# Reference configuration the target was measured from (shipped material and
# pull point + a mid-range command). The ParamSpec defaults below are
# deliberately OFF these values so the optimizer has a calibration task.
REFERENCE = {"young_modulus": 18000.0, "cable_displacement": 15.0, "pull_point_y": 0.0}

PROJECT = SofaOptProject(
    name="soft_finger",
    title="Soft finger actuated reach — sofaopt soft-robot SOFA tier",
    work_dir=HERE,
    params=[
        # name, type, min, max, default (defaults off-target on purpose)
        ParamSpec("young_modulus", "float", 5000.0, 40000.0, 30000.0),
        ParamSpec("cable_displacement", "float", 0.0, 25.0, 5.0),
        ParamSpec("pull_point_y", "float", -5.0, 20.0, 10.0),
    ],
    tests=[
        TestSpec(
            "reach",
            scene_file=HERE / "scene.py",
            label="Tip reach match",
            description="ramp the cable, land the fingertip on the measured target",
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
    n_generations=25,
    cmaes_sigma0=0.4,
    seed_sampler="sobol",
    sofa_realtime_timeout=180.0,
    run_script=HERE / "run.py",
)

_VARIANTS = {"default": PROJECT}


def get_variant(name: str) -> SofaOptProject:
    try:
        return _VARIANTS[name]
    except KeyError:
        raise SystemExit(f"Unknown variant '{name}'. Have: {sorted(_VARIANTS)}") from None
