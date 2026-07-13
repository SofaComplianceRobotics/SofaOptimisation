"""Feature-advantage bench — does the restart machinery help, measured honestly?

Drives the REAL sofaopt orchestrator against the analytic benchmark functions
with a SOFA-free scoring generation. Because the global optimum is known, we can
*measure* the effect of each restart configuration at EQUAL budget:

  plain : restarts off, full budget (no early stop)          — the baseline.
  stall : cold restarts on the stall plateau (old behavior)  — fires early, hurts.
  conv  : restarts on CMA-ES convergence + warm-start (new)  — do-no-harm.

The headline result (established by a wider sweep, examples/landscape/README):
the OLD stall trigger fires while the search is still productive and makes
restarts net-negative; the convergence trigger only restarts once CMA-ES has
genuinely converged, so at a tight budget it does no harm, and at a large budget
it helps (biggest on deceptive/multimodal landscapes). These tests pin the
do-no-harm invariant and that the new trigger beats the old one where the old
one hurt — both cheap to check; the large-budget gains are in the README table.

Racing's end-to-end savings live in the finalize phase (real run slots), so they
are measured on the noisy SOFA ports, not here.
"""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import optuna

from sofaopt.core import orchestrator
from sofaopt.core.algorithm import tell_safely
from sofaopt.core.runconfig import RunConfig
from sofaopt.project import ParamSpec, SofaOptProject
from sofaopt.project import TestSpec as _TestSpec

from conftest import load_benchmark_functions

bf = load_benchmark_functions()
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Restart configurations under test (all at equal n_generations budget).
PLAIN = dict(cmaes_restarts=0, stall_generations=0)
STALL = dict(cmaes_restarts=6, stall_generations=3)  # old: cold restart on plateau
CONV = dict(cmaes_restarts=6, stall_generations=0,   # new: convergence + warm-start
            restart_on_convergence=True, warm_restarts=True)


def _project(function: str, dim: int, tmp: Path, gens: int, **kw) -> SofaOptProject:
    bench = bf.get_benchmark(function)
    d = bench.fixed_dim or dim
    params = [
        ParamSpec(f"x{i}", "float", bench.low, bench.high, bench.default) for i in range(d)
    ]
    base = dict(
        name=f"land_{function}", work_dir=tmp, params=params,
        tests=[_TestSpec("f", scene_file=tmp / "s.py", max_score=100.0)],
        runner="python", sampler="cmaes", n_parallel=6, n_generations=gens,
        cmaes_sigma0=0.4,
    )
    base.update(kw)
    return SofaOptProject(**base)


def _best(function: str, dim: int, gens: int, cfg: dict) -> float:
    tmp = Path(tempfile.mkdtemp())
    project = _project(function, dim, tmp, gens, **cfg)
    searched = [p for p in project.params if not p.is_frozen]

    def fake_gen(cfg_, gi, trials, study, env, state,
                 started_at=0.0, total_gens=None, restart_state=None):
        for t in trials:
            x = [t.suggest_float(p.name, float(p.low), float(p.high)) for p in searched]
            s = bf.scored(function, x)
            tell_safely(study, t, s)
            state.record_score(s)

    saved = orchestrator.run_generation
    orchestrator.run_generation = fake_gen
    try:
        orchestrator._run(project, RunConfig.from_project(project))
    finally:
        orchestrator.run_generation = saved
    study = optuna.load_study(
        study_name=project.name, storage=f"sqlite:///{project.db_path}"
    )
    return float(study.best_value)


def _mean(function: str, dim: int, gens: int, cfg: dict, reps: int) -> float:
    return statistics.mean(_best(function, dim, gens, cfg) for _ in range(reps))


# -- the fix: convergence trigger doesn't hurt where the stall trigger did -----

def test_convergence_trigger_fixes_the_stall_trigger_regression():
    """On ackley at a tight budget, the OLD stall-triggered cold restart is
    net-negative (fires early, jumps to random points); the NEW convergence
    trigger doesn't fire (nothing converged with budget to spare), so it matches
    plain CMA-ES. This is the measured upgrade, at equal budget."""
    reps, gens = 8, 40
    plain = _mean("ackley", 2, gens, PLAIN, reps)
    stall = _mean("ackley", 2, gens, STALL, reps)
    conv = _mean("ackley", 2, gens, CONV, reps)
    assert stall < plain - 3.0, f"expected the old trigger to hurt: plain={plain:.1f} stall={stall:.1f}"
    assert conv > stall + 3.0, f"convergence trigger should beat the old one: conv={conv:.1f} stall={stall:.1f}"
    assert conv > plain - 3.0, f"convergence trigger should not harm vs plain: conv={conv:.1f} plain={plain:.1f}"


def test_convergence_restarts_do_no_harm_on_rosenbrock():
    """A second do-no-harm case (rosenbrock, where cold restarts also hurt)."""
    reps, gens = 8, 40
    plain = _mean("rosenbrock", 2, gens, PLAIN, reps)
    conv = _mean("rosenbrock", 2, gens, CONV, reps)
    assert conv > plain - 3.0, f"conv={conv:.1f} plain={plain:.1f}"


# -- control: on a unimodal bowl every config solves it ------------------------

def test_all_configs_solve_unimodal_sphere():
    reps, gens = 6, 25
    for cfg in (PLAIN, CONV):
        assert _mean("sphere", 2, gens, cfg, reps) > 95.0


# -- run_until_converged with the convergence trigger self-sizes ---------------

def test_converged_mode_with_convergence_trigger_self_sizes():
    """run_until_converged on the convergence trigger: the initial run converges
    and finds sphere's optimum (productive), the warm restart re-converges
    without improving (fruitless), and at patience=1 the run stops — well before
    the generous ceiling. (Convergence-triggered restarts cycle slowly, so a
    self-sizing run needs a generous ceiling; that's what it is for.)"""
    tmp = Path(tempfile.mkdtemp())
    ceiling = 500
    project = _project(
        "sphere", 2, tmp, gens=ceiling, cmaes_restarts=3,
        restart_on_convergence=True, warm_restarts=True,
        run_until_converged=True, restart_patience=1, stall_generations=0,
    )
    searched = [p for p in project.params if not p.is_frozen]

    def fake_gen(cfg_, gi, trials, study, env, state,
                 started_at=0.0, total_gens=None, restart_state=None):
        for t in trials:
            x = [t.suggest_float(p.name, float(p.low), float(p.high)) for p in searched]
            tell_safely(study, t, bf.scored("sphere", x))
            state.record_score(0.0)

    saved = orchestrator.run_generation
    orchestrator.run_generation = fake_gen
    try:
        orchestrator._run(project, RunConfig.from_project(project))
    finally:
        orchestrator.run_generation = saved
    study = optuna.load_study(
        study_name=project.name, storage=f"sqlite:///{project.db_path}"
    )
    gens_run = len(study.trials) // project.n_parallel
    assert gens_run < ceiling, f"did not self-size (ran {gens_run} gens)"
    assert study.best_value > 99.0  # sphere solved


def _report() -> None:
    """Plain vs stall-trigger vs convergence-trigger, equal budget — run via
    ``python tests/test_landscape_features.py``."""
    reps, gens = 10, 40
    print(f"\nMean best over {reps} reps, 2-D, {6 * gens} evals (equal budget):\n")
    print(f"{'function':<12}{'plain':>8}{'stall':>8}{'conv':>8}")
    for fn in ("sphere", "rosenbrock", "rastrigin", "ackley", "schwefel", "himmelblau"):
        p = _mean(fn, 2, gens, PLAIN, reps)
        s = _mean(fn, 2, gens, STALL, reps)
        c = _mean(fn, 2, gens, CONV, reps)
        print(f"{fn:<12}{p:>8.1f}{s:>8.1f}{c:>8.1f}")


if __name__ == "__main__":
    _report()
