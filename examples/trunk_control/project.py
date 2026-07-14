"""Soft trunk control allocation — the control-flavored study platform.

Ported from the SoftRobots plugin's Trunk tutorial. Eight cable displacements
(4 long + 4 short) shape a soft continuum; the objective is to reproduce a
**target backbone curve** measured once from a reference actuation — the
classic control-allocation problem (8 redundant, antagonistic actuators for a
low-dimensional shape).

Companion to ``liver_elastography`` (the identification-flavored study
platform). Where elastography has graded sensitivity and a hidden lesion, the
trunk has **redundancy and antagonism**: opposing cables cancel, so many
displacement vectors reach the same backbone — a genuinely non-convex,
many-optima control landscape for restarts/covariance to work on. ~2 s
launches.

Reference/target provenance: see TARGET_BACKBONE in ``scene.py`` — measured
via the python runner from the reference actuation; command in the README.
"""

from __future__ import annotations

import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent

# Long cables pull harder (longer moment arm) so their range is wider. The
# reference actuation (an asymmetric curl) is applied to a FEW cables; the rest
# are zero — the optimizer must discover which cables shape the target.
_PARAM_RANGES = {
    **{f"cable_L{i}": (0.0, 40.0) for i in range(4)},
    **{f"cable_S{i}": (0.0, 30.0) for i in range(4)},
}
# Reference actuation the backbone target was measured from (see README).
REFERENCE = {
    "cable_L0": 30.0, "cable_L1": 0.0, "cable_L2": 0.0, "cable_L3": 0.0,
    "cable_S0": 0.0, "cable_S1": 20.0, "cable_S2": 0.0, "cable_S3": 0.0,
}

PROJECT = SofaOptProject(
    name="trunk_control",
    title="Soft trunk control allocation — sofaopt control study platform",
    work_dir=HERE,
    params=[
        # start all cables slack (0.0) — a blind, off-target start
        ParamSpec(name, "float", lo, hi, 0.0)
        for name, (lo, hi) in _PARAM_RANGES.items()
    ],
    tests=[
        TestSpec(
            "backbone",
            scene_file=HERE / "scene.py",
            label="Backbone match",
            description="ramp 8 cables, match the target backbone curve",
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
    n_generations=60,          # 8-dim search needs a real budget
    cmaes_sigma0=0.3,
    cmaes_restarts=4,          # the study platform ships the studied config
    restart_on_convergence=True,
    warm_restarts=True,
    seed_sampler="sobol",
    sofa_realtime_timeout=240.0,
    run_script=HERE / "run.py",
)

_VARIANTS = {"default": PROJECT}


def get_variant(name: str) -> SofaOptProject:
    try:
        return _VARIANTS[name]
    except KeyError:
        raise SystemExit(f"Unknown variant '{name}'. Have: {sorted(_VARIANTS)}") from None
