"""Caduceus contact-parameter identification — the heavy real-SOFA tier.

Ported from SOFA's shipped ``examples/Demos/caduceus.scn`` (the iconic demo).
The FEM snake drops onto the pod and wraps itself around it under frictional
contact; the final wrapped pose identifies (young_modulus, friction_mu,
total_mass). The objective is to reproduce a **target pose measured once from
the shipped values** — contact-parameter identification from an observed rest
pose.

Why it matters to the framework (beyond being the demo everyone recognizes):
trials run a real collision pipeline + LCP friction solve per step and take
tens of seconds — the tier that exercises wall-clock timeout wedges, the
parallel launcher under load, and restart/convergence logic at realistic
budgets. Friction makes the landscape *qualitatively* different from the
material tiers: below a mu threshold the snake slides off the pod entirely
(a basin boundary, not a smooth ridge).

Reference/target provenance: ``target.json`` — measured via the python runner
from the shipped values; command in the README.
"""

from __future__ import annotations

import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent

# Reference values the target was measured from (the caduceus.scn values).
# The ParamSpec defaults below are deliberately OFF these values.
REFERENCE = {"young_modulus": 30000.0, "friction_mu": 0.2, "total_mass": 1.0}

PROJECT = SofaOptProject(
    name="caduceus_settle",
    title="Caduceus wrapped-pose match — sofaopt heavy SOFA tier (contact)",
    work_dir=HERE,
    params=[
        # name, type, min, max, default (defaults off-target on purpose)
        ParamSpec("young_modulus", "float", 5000.0, 100000.0, 60000.0),
        ParamSpec("friction_mu", "float", 0.0, 0.6, 0.4),
        ParamSpec("total_mass", "float", 0.3, 3.0, 1.0),
    ],
    tests=[
        TestSpec(
            "wrap",
            scene_file=HERE / "scene.py",
            label="Wrapped-pose match",
            description="drop the snake onto the pod, match the measured wrapped pose",
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
    n_generations=15,
    cmaes_sigma0=0.4,
    seed_sampler="sobol",
    sofa_realtime_timeout=600.0,
    run_script=HERE / "run.py",
)

_VARIANTS = {"default": PROJECT}


def get_variant(name: str) -> SofaOptProject:
    try:
        return _VARIANTS[name]
    except KeyError:
        raise SystemExit(f"Unknown variant '{name}'. Have: {sorted(_VARIANTS)}") from None
