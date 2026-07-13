"""Analytic landscape harness — the optimizer-feature validation example.

Unlike ``cube_drop`` (deterministic, one basin), this example's score is a
classic optimization test function of the params (Rastrigin, Schwefel, …), so it
is genuinely **multimodal** and optionally **noisy** — exactly what the new
optimizer features need to show their value:

- **IPOP restarts** (``cmaes_restarts``) escape local basins → this default
  project turns them on, and ``run_until_converged`` lets the search self-size.
- **Racing** (``TestSpec.run_count_min``) skips wasted repeats on the noisy
  variant (``PROJECT_NOISY``).

The known global optimum makes the advantages *measurable* — see
``tests/test_landscape_features.py`` for the plain-vs-IPOP comparison bench. The
SOFA scene places a marker mass at a height proportional to the score, so a run
is watchable and records to video like any other project.

Switch functions / dimension by editing ``FUNCTION`` / ``DIM`` below, or pick a
variant with ``python run.py --variant <name>``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from sofaopt import ParamSpec, SofaOptProject, TestSpec

HERE = Path(__file__).resolve().parent

# The landscape. Multimodal defaults so restarts have something to do.
FUNCTION = "rastrigin"   # sphere | rosenbrock | rastrigin | ackley | schwefel | himmelblau
DIM = 4                  # searched dimensions (himmelblau forces 2)
NOISE = 0.0              # per-run Gaussian score noise sigma (see PROJECT_NOISY)

# Per-function search box + a deliberately off-optimum start (so there is a hill
# to climb). Kept in sync with benchmark_functions.BENCHMARKS.
_DOMAIN = {
    "sphere": (-5.12, 5.12, 3.0),
    "rosenbrock": (-2.048, 2.048, -1.5),
    "rastrigin": (-5.12, 5.12, 4.5),
    "ackley": (-32.768, 32.768, 20.0),
    "schwefel": (-500.0, 500.0, -300.0),
    "himmelblau": (-5.0, 5.0, 0.0),
}


def _params(function: str, dim: int) -> list[ParamSpec]:
    low, high, default = _DOMAIN[function]
    d = 2 if function == "himmelblau" else dim
    return [ParamSpec(f"x{i}", "float", low, high, default) for i in range(d)]


PROJECT = SofaOptProject(
    name="landscape",
    title="Analytic landscape — optimizer feature bench",
    work_dir=HERE,
    params=_params(FUNCTION, DIM),
    tests=[
        TestSpec(
            "f",
            scene_file=HERE / "scene.py",
            label="Landscape score",
            description=f"maximize {FUNCTION} score (100 = global optimum)",
            max_score=100.0,
            default_selected=True,
        )
    ],
    runsofa_exe=Path(os.environ.get("RUNSOFA_EXE", "runSofa")),
    sofa_env={
        **{k: os.environ[k]
           for k in ("SOFA_ROOT", "SOFAPYTHON3_ROOT", "PYTHONPATH")
           if k in os.environ},
        "OPT_LANDSCAPE_FN": FUNCTION,
        "OPT_LANDSCAPE_NOISE": str(NOISE),
    },
    gui_mode="batch",
    n_parallel=6,
    n_generations=200,         # restarts need a generous budget to pay off (see README)
    cmaes_sigma0=0.4,
    seed_sampler="sobol",
    # Restarts, configured the way the benchmark showed actually helps:
    cmaes_restarts=4,          # IPOP restarts on a multimodal landscape
    restart_on_convergence=True,  # trigger on REAL convergence, not a best-plateau
    warm_restarts=True,        # re-seed from the incumbent, not a random jump
    run_script=HERE / "run.py",
    record_frames=True,
    record_frame_skip=2,
    record_frame_size=(640, 480),
    sofa_realtime_timeout=60.0,
    runner="python",
)

# ---------------------------------------------------------------------------
# Variants — python run.py --variant <name>
# ---------------------------------------------------------------------------

# Self-sizing: keep restarting until restarts stop paying off, then stop —
# n_generations becomes a safety ceiling (see optimization-guide §5).
PROJECT_CONVERGED = dataclasses.replace(
    PROJECT,
    name="landscape_converged",
    run_until_converged=True,
    restart_patience=2,
    n_generations=300,         # generous ceiling; convergence decides the real length
)

# Plain CMA-ES (no restarts) — the baseline to compare restarts against on the
# dashboard's Archives tab (run this and PROJECT back to back, then overlay).
PROJECT_PLAIN = dataclasses.replace(
    PROJECT, name="landscape_plain", cmaes_restarts=0,
    restart_on_convergence=False, warm_restarts=False, stall_generations=0,
)

# The OLD restart behavior (cold restart on a best-plateau) — kept as a variant
# so the harm the convergence trigger fixes is reproducible: run this vs default
# and overlay on the Archives tab.
PROJECT_STALL_RESTART = dataclasses.replace(
    PROJECT, name="landscape_stall_restart",
    restart_on_convergence=False, warm_restarts=False, stall_generations=6,
)

# Noisy objective — turns on racing: start each trial with run_count_min repeats
# and add more only while the candidate might still beat the incumbent.
_NOISY_TEST = TestSpec(
    "f",
    scene_file=HERE / "scene.py",
    label="Landscape score (noisy)",
    description="maximize a noisy landscape; repeats averaged, raced on CI overlap",
    max_score=100.0,
    run_count=5,
    run_count_min=2,           # racing: 2 repeats first, more only if competitive
    score_aggregation="mean",
)
PROJECT_NOISY = dataclasses.replace(
    PROJECT,
    name="landscape_noisy",
    tests=[_NOISY_TEST],
    sofa_env={**PROJECT.sofa_env, "OPT_LANDSCAPE_NOISE": "6.0"},
)

_VARIANTS = {
    "default": PROJECT,
    "converged": PROJECT_CONVERGED,
    "plain": PROJECT_PLAIN,
    "stall-restart": PROJECT_STALL_RESTART,
    "noisy": PROJECT_NOISY,
}


def get_variant(name: str) -> SofaOptProject:
    try:
        return _VARIANTS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown variant '{name}'. Have: {sorted(_VARIANTS)}"
        ) from None
