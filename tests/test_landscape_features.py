"""Feature-advantage bench — does the new optimizer machinery actually help?

Drives the REAL sofaopt orchestrator (IPOP restarts, run_until_converged) against
the analytic benchmark functions, with a SOFA-free fake generation that scores
each asked trial directly. Because the global optimum is known, we can *measure*
the advantage, not just check the plumbing:

- IPOP restarts escape a bad local basin where plain CMA-ES stays trapped.
- restarts don't hurt on a unimodal function (the control).
- run_until_converged self-sizes: it stops well before the generation ceiling.

Each run is milliseconds (no SOFA), so we average over several reps for a robust,
non-flaky comparison. Run as a script for the full comparison table::

    python tests/test_landscape_features.py

Racing's end-to-end evaluation savings live in the finalize phase (real run
slots), so they are measured on the noisy SOFA e2e ports, not here; racing's
decision logic is unit-tested in test_racing.py.
"""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path

import optuna

from sofaopt.core import orchestrator
from sofaopt.core.algorithm import tell_safely
from sofaopt.core.restart import restart_index
from sofaopt.core.runconfig import RunConfig
from sofaopt.project import ParamSpec, SofaOptProject
from sofaopt.project import TestSpec as _TestSpec

from conftest import load_benchmark_functions

bf = load_benchmark_functions()
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _project(function: str, dim: int, tmp: Path, **kw) -> SofaOptProject:
    bench = bf.get_benchmark(function)
    d = bench.fixed_dim or dim
    default = kw.pop("default", bench.default)
    params = [
        ParamSpec(f"x{i}", "float", bench.low, bench.high, default=default)
        for i in range(d)
    ]
    base = dict(
        name=f"land_{function}",
        work_dir=tmp,
        params=params,
        tests=[_TestSpec("f", scene_file=tmp / "s.py", max_score=100.0)],
        runner="python",
        sampler="cmaes",
        n_parallel=6,
        n_generations=30,
        stall_generations=4,
    )
    base.update(kw)
    return SofaOptProject(**base)


def _run_best(project: SofaOptProject, function: str, noise_sigma: float = 0.0) -> tuple[float, int]:
    """Run the real orchestrator with a SOFA-free scoring generation. Returns
    (best score, restarts performed)."""
    searched = [p for p in project.params if not p.is_frozen]

    def fake_gen(cfg, gi, trials, study, env, state,
                 started_at=0.0, total_gens=None, restart_state=None):
        for t in trials:
            x = [t.suggest_float(p.name, float(p.low), float(p.high)) for p in searched]
            s = bf.scored(function, x, noise_sigma=noise_sigma)
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
    return float(study.best_value), restart_index(study)


def _mean_best(function: str, reps: int, **project_kw) -> float:
    bests = []
    for _ in range(reps):
        tmp = Path(tempfile.mkdtemp())
        best, _ = _run_best(_project(function, 2, tmp, **project_kw), function)
        bests.append(best)
    return statistics.mean(bests)


# -- IPOP restarts vs plain CMA-ES on a multimodal function --------------------

def test_ipop_beats_plain_cmaes_when_trapped_on_rastrigin():
    """Local start + tight sigma0 traps plain CMA-ES in a rim basin; IPOP's
    random-restart re-seeds escape it. Averaged over reps for robustness."""
    reps = 10
    common = dict(default=4.0, cmaes_sigma0=0.4, n_generations=30, stall_generations=3)
    plain = _mean_best("rastrigin", reps, cmaes_restarts=0, **common)
    ipop = _mean_best("rastrigin", reps, cmaes_restarts=4, **common)
    assert ipop > plain, f"IPOP {ipop:.1f} did not beat plain {plain:.1f}"


def test_ipop_finds_a_himmelblau_global_optimum():
    """Four equal optima; restarts from the centre reliably land on one.

    Averaged over reps: at least one rep should hit a near-global optimum and
    the mean should stay high (a single CMA-ES run is stochastic)."""
    bests = []
    for _ in range(5):
        tmp = Path(tempfile.mkdtemp())
        project = _project(
            "himmelblau", 2, tmp, default=0.0, cmaes_sigma0=0.5,
            cmaes_restarts=4, stall_generations=3, n_generations=40,
        )
        best, _ = _run_best(project, "himmelblau")
        bests.append(best)
    assert max(bests) > 99.0, f"no rep reached a global optimum: {bests}"
    assert statistics.mean(bests) > 90.0, f"mean too low: {bests}"


# -- control: restarts must not hurt a unimodal function -----------------------

def test_restarts_do_not_hurt_on_unimodal_sphere():
    reps = 8
    common = dict(default=3.0, cmaes_sigma0=1.0, n_generations=25)
    plain = _mean_best("sphere", reps, cmaes_restarts=0, **common)
    ipop = _mean_best("sphere", reps, cmaes_restarts=3, stall_generations=4, **common)
    assert plain > 90.0 and ipop > 90.0  # both essentially solve the bowl
    assert abs(ipop - plain) < 6.0       # the point: restarts neither help nor harm here


# -- run_until_converged self-sizes -------------------------------------------

def test_converged_stops_before_ceiling_with_a_good_optimum():
    tmp = Path(tempfile.mkdtemp())
    project = _project(
        "rastrigin", 2, tmp, default=4.0, cmaes_sigma0=0.5,
        cmaes_restarts=6, run_until_converged=True, restart_patience=2,
        stall_generations=3, n_generations=200,  # generous ceiling
    )
    best, _ = _run_best(project, "rastrigin")
    study = optuna.load_study(
        study_name=project.name, storage=f"sqlite:///{project.db_path}"
    )
    gens_run = len(study.trials) // project.n_parallel
    # Self-sized: stopped far short of the 200-generation ceiling.
    assert gens_run < 100, f"did not self-size (ran {gens_run} gens)"
    assert best > 50.0  # still reached a reasonable optimum


def _report() -> None:
    """Print the plain-vs-IPOP comparison table (the 'assess the advantages'
    artifact). Not a test — run via ``python tests/test_landscape_features.py``."""
    reps = 12
    print(f"\nMean best score over {reps} reps (2-D, local start, tight sigma0):\n")
    print(f"{'function':<12}{'plain CMA-ES':>14}{'IPOP restarts':>16}{'gain':>8}")
    for fn in ("sphere", "rosenbrock", "rastrigin", "ackley", "schwefel", "himmelblau"):
        common = dict(default=bf.get_benchmark(fn).default, cmaes_sigma0=0.4,
                      n_generations=30, stall_generations=3)
        plain = _mean_best(fn, reps, cmaes_restarts=0, **common)
        ipop = _mean_best(fn, reps, cmaes_restarts=4, **common)
        print(f"{fn:<12}{plain:>14.1f}{ipop:>16.1f}{ipop - plain:>+8.1f}")


if __name__ == "__main__":
    _report()
