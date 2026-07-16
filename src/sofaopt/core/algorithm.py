"""CMA-ES study setup and per-trial score finalization."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import optuna

import statistics

from sofaopt.core.restart import build_restart_sampler, restart_index, restart_popsize
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import aggregate_repeats, combine_weighted
from sofaopt.core.trial_state import (

    read_trial_run,
    read_trial_state,
    update_trial_summary,
)

logger = logging.getLogger(__name__)


def _seed_sampler(project) -> optuna.samplers.BaseSampler:
    """Initial-design sampler for the startup/independent phase.

    ``"sobol"`` → a scrambled Sobol' (QMC) space-filling design; otherwise a
    plain random sampler (the historical default).
    """
    if project.seed_sampler == "sobol":
        # Fixed seed → reproducible (scrambled) Sobol' design; a single sampler
        # instance drives all asks, so the parallel-seed caveat does not apply.
        # A different seed gives a different, equally balanced design — use it to
        # get an INDEPENDENT exploration for a validation run.
        return optuna.samplers.QMCSampler(
            qmc_type="sobol", scramble=True, seed=project.seed_sampler_seed
        )
    return optuna.samplers.RandomSampler()


def _cmaes_sampler(project, startup_trials: int) -> optuna.samplers.CmaEsSampler:
    # Center the search on the ParamSpec defaults (the documented contract).
    # Only float/int non-frozen params are in the CMA-ES space; bools are
    # categorical and handled by the independent sampler.
    x0 = {
        p.name: min(max(p.default, p.low), p.high)
        for p in project.params
        if not p.is_frozen and p.type in ("float", "int")
    }
    return optuna.samplers.CmaEsSampler(
        x0=x0 or None,
        sigma0=project.cmaes_sigma0,
        popsize=project.n_parallel,
        n_startup_trials=startup_trials,
        consider_pruned_trials=True,
        with_margin=project.cmaes_with_margin,
        independent_sampler=_seed_sampler(project),
    )


def _single_objective_sampler(project) -> optuna.samplers.BaseSampler:
    startup_trials = project.resolve_startup_trials()
    if project.cmaes_startup_trials is None and project.sampler in ("cmaes", "gp"):
        searched = sum(1 for p in project.params if not p.is_frozen)
        logger.info(
            f"[sampler] startup trials auto-sized to {startup_trials} "
            f"({searched} searched params, sampler={project.sampler})"
        )
    if project.sampler == "cmaes":
        return _cmaes_sampler(project, startup_trials)
    if project.sampler == "gp":
        return optuna.samplers.GPSampler(
            n_startup_trials=startup_trials,
            independent_sampler=_seed_sampler(project),
        )
    if project.sampler == "tpe":
        return optuna.samplers.TPESampler()
    return optuna.samplers.RandomSampler()


# SQLite's own default busy-wait (5s, via sqlite3.connect's own default) is
# not enough once a generation's trials finish in a tight burst -- e.g. a
# large IPOP-restart popsize means many worker processes committing scores/
# system attrs to the same file within a short window. Observed in practice:
# a 4-restart FoamBotHex run (popsize grown to 160) died with
# "sqlite3.OperationalError: database is locked" wrapped in Optuna's own
# generic StorageInternalError. 60s gives real headroom without masking a
# genuinely wedged writer forever.
_SQLITE_BUSY_TIMEOUT_S = 60


def _build_storage(db_path: Path) -> optuna.storages.RDBStorage:
    """RDBStorage tuned for many concurrent SOFA worker processes.

    Two standard SQLite-under-concurrency mitigations, not just a longer
    wait: WAL journal mode lets readers (the dashboard, an interrupted-run
    recovery scan) proceed without blocking on a writer and vice versa,
    which a longer busy-timeout alone does not fix.
    """
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": _SQLITE_BUSY_TIMEOUT_S}},
    )
    with storage.engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    return storage


def build_study(db_path: Path, cfg: RunConfig, resume: bool = False) -> optuna.Study:
    """Create (or resume) an Optuna study backed by a SQLite database.

    Supports CMA-ES (optionally with Margin), GP-BO, TPE, Random
    (single-objective) and NSGA-II (multi-objective). The startup/independent
    phase of CMA-ES and GP can be a space-filling Sobol' design
    (``project.seed_sampler == "sobol"``).

    When ``resume`` is True an existing database at ``db_path`` is **loaded**
    (prior trials are kept; CMA-ES state persists in storage — including the
    latest IPOP restart, see :func:`_restore_restart_sampler`). When False,
    any existing database is deleted so the run starts fresh.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists() and not resume:
        db_path.unlink()
        logger.info(f"[reset] Deleted {db_path.name}")

    project = cfg.project
    storage = _build_storage(db_path)

    if project.multi_objective:
        return optuna.create_study(
            study_name=project.name,
            sampler=optuna.samplers.NSGAIISampler(population_size=project.n_parallel),
            directions=[t.direction for t in cfg.selected_tests],
            storage=storage,
            load_if_exists=resume,
        )

    study = optuna.create_study(
        study_name=project.name,
        sampler=_single_objective_sampler(project),
        direction="maximize",
        storage=storage,
        load_if_exists=resume,
    )
    if resume and project.sampler == "cmaes":
        _restore_restart_sampler(study, project)
    return study


def _restore_restart_sampler(study: optuna.Study, project) -> None:
    """A resumed run that had IPOP-restarted must keep sampling from its
    latest restart: the restart-scoped sampler makes the pre-restart CMA
    state invisible (see ``core/restart.py``)."""
    index = restart_index(study)
    if index <= 0:
        return
    study.sampler = build_restart_sampler(project, index, _seed_sampler(project))
    logger.info(
        f"[resume] Continuing from IPOP restart {index} "
        f"(popsize {restart_popsize(project, index)})."
    )


def recover_interrupted_trials(study: optuna.Study) -> int:
    """Re-enqueue interrupted (RUNNING) trials so a resumed run re-evaluates them.

    A paused/killed run leaves asked-but-never-told trials behind. Their param
    vectors are enqueued (``study.enqueue_trial``) so the next generation's asks
    return them first, and the stale RUNNING records are closed as FAILED so
    they don't linger in the study. Returns how many were re-enqueued.
    """
    running = study.get_trials(
        deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)
    )
    count = 0
    for t in running:
        if t.params:
            study.enqueue_trial(t.params)
            count += 1
        # The stale RUNNING record may already be closed by a concurrent
        # resume; the enqueue above is what matters.
        with contextlib.suppress(Exception):
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    return count


def tell_safely(study: optuna.Study, trial, *args, **kwargs) -> None:
    """``study.tell`` that survives a trial someone else already closed
    ("Cannot tell a FAIL trial"): the score is on disk either way, and one
    unrecordable trial must not kill the whole run."""
    try:
        study.tell(trial, *args, **kwargs)
    except Exception as exc:
        num = getattr(trial, "number", trial)
        logger.error(f"[optuna] Could not record trial {num}: {exc}")


def _is_pruned(trial_state: dict) -> bool:
    return str(trial_state.get("state", "")).lower() == "pruned" or any(
        isinstance(run, dict) and str(run.get("state", "")).lower() == "pruned"
        for run in trial_state.get("runs", [])
    )


def _tell_pruned(study, trial, trial_state, trial_state_path, trial_index) -> float:
    tell_safely(study, trial, state=optuna.trial.TrialState.PRUNED)
    update_trial_summary(
        trial_state_path,
        {
            "state": "pruned",
            "final_score": None,
            "outcome": trial_state.get("outcome", "pruned"),
        },
    )
    logger.info(f"[score] trial_{trial_index:02d} -> pruned (generation pruned)")
    return float("-inf")


def _fail_trial(cfg, study, trial, trial_state_path, outcome: str) -> float:
    """Report a hard-failed trial to Optuna and the trial summary."""
    hard_fail = cfg.project.hard_fail_score
    if cfg.project.multi_objective:
        tell_safely(study, trial, [hard_fail] * len(cfg.selected_tests))
    else:
        tell_safely(study, trial, hard_fail)
    update_trial_summary(
        trial_state_path,
        {"state": "failed", "final_score": hard_fail, "outcome": outcome},
    )
    return hard_fail


def _read_run_results(trial_state_path: Path, runs: list[tuple]) -> list[tuple]:
    """Per-slot ``(test_name, score, test_run_total)``; a crashed slot scores -inf.

    Scores are grouped by the test name each slot recorded for itself, so
    attribution survives relaunches/out-of-order gated launches.
    """
    results: list[tuple[str, float, int | None]] = []
    for _p, _path, run_slot in runs:
        run_data = read_trial_run(trial_state_path, run_slot) or {}
        raw = run_data.get("score")
        score = float(raw) if isinstance(raw, (int, float)) else float("-inf")
        results.append(
            (str(run_data.get("test_name", "")), score, run_data.get("test_run_total"))
        )
    return results


def _one_test_details(cfg: RunConfig, test_name: str, run_results: list[tuple]) -> dict:
    """Aggregate one test's run scores into its trial-summary details dict."""
    raw_scores = [s for name, s, _ in run_results if name == test_name]
    scores_for_test = [0.0 if s == float("-inf") else s for s in raw_scores]
    crashed_runs = sum(1 for s in raw_scores if s == float("-inf"))
    test_aggregate = aggregate_repeats(
        scores_for_test, cfg.test_aggregations.get(test_name, "mean")
    )
    max_score = cfg.test_max_scores.get(test_name, 1.0)
    test_run_total = next(
        (rt for name, _, rt in run_results if name == test_name and rt is not None),
        len(scores_for_test),
    )
    return {
        "run_count": len(scores_for_test),
        "crashed_run_count": crashed_runs,
        "run_scores": [round(s, 4) for s in scores_for_test],
        "aggregate_score": round(test_aggregate, 4),
        "median_score": round(statistics.median(scores_for_test), 4),
        "run_total": test_run_total,
        "max_score": max_score,
        "weight_pct": round(cfg.test_weights.get(test_name, 0.0) * 100, 1),
        "normalized_score": round(
            min(test_aggregate / max_score, 1.0) if max_score > 0 else 0.0, 4
        ),
    }


def _aggregate_per_test(cfg: RunConfig, run_results: list[tuple]):
    """Aggregate run scores per test -> (test names in first-seen order, details)."""
    test_names_in_order = list(dict.fromkeys(name for name, _, _ in run_results if name))
    per_test_details = {
        name: _one_test_details(cfg, name, run_results) for name in test_names_in_order
    }
    return test_names_in_order, per_test_details


def _tell_multi_objective(
    cfg, trial, study, trial_index, gen_index, trial_state_path,
    per_test_details, run_results, valid_scores,
) -> float:
    """Multi-objective path: each test is one Pareto objective."""
    hard_fail = cfg.project.hard_fail_score
    run_scores = [s for _, s, _ in run_results]
    objective_values = [
        per_test_details[t.name]["aggregate_score"]
        if t.name in per_test_details
        else hard_fail
        for t in cfg.selected_tests
    ]
    tell_safely(study, trial, objective_values)
    update_trial_summary(
        trial_state_path,
        {
            "trial": trial_index,
            "gen": gen_index,
            "state": "done",
            "n_runs": len(valid_scores),
            "test_names": list(cfg.selected_names),
            "test_scores": per_test_details,
            "objective_values": [round(v, 4) for v in objective_values],
            "run_scores": [round(s, 4) if s != float("-inf") else None for s in run_scores],
        },
    )
    logger.info(
        f"\n[score] trial_{trial_index:02d} -> "
        f"objectives={[round(v, 4) for v in objective_values]}"
    )
    return objective_values[0] if objective_values else hard_fail


def _gate_and_weight(cfg: RunConfig, test_names_in_order: list[str], per_test_details: dict):
    """Apply the gated-test rule -> (counted names, normalized weights, gate_open).

    Gated (expensive) tests only count when at least one ungated test scored
    above zero; weights are renormalized over the counted tests.
    """
    gated_names_cfg = cfg.gated_test_names
    configured_gated_names = [n for n in cfg.selected_names if n in gated_names_cfg]
    ungated_names = [n for n in test_names_in_order if n not in gated_names_cfg]
    gate_open = not configured_gated_names or any(
        per_test_details.get(name, {}).get("aggregate_score", 0.0) > 0.0
        for name in ungated_names
    )
    counted_names = test_names_in_order if gate_open else ungated_names
    if not counted_names:
        counted_names = test_names_in_order

    counted_weight_total = sum(cfg.test_weights.get(name, 0.0) for name in counted_names)
    if counted_weight_total > 0:
        counted_weights = {
            name: cfg.test_weights.get(name, 0.0) / counted_weight_total
            for name in counted_names
        }
    else:
        counted_weights = {name: 1.0 / len(counted_names) for name in counted_names}
    return counted_names, counted_weights, gate_open


def _finalize_trial_score(
    cfg: RunConfig,
    trial_index: int,
    trial: "optuna.trial.Trial",
    runs: list[tuple],
    trial_state_path: Path,
    study: optuna.Study,
    gen_index: int,
) -> float:
    """Compute a completed trial's final score and report it to Optuna.

    Reads per-run scores, aggregates per test, gates expensive tests, combines
    across tests by weight, calls ``study.tell()`` and writes the trial summary.

    Returns the final score out of 100, or the project's ``hard_fail_score``
    only when every run failed (partial failures are isolated per test).
    """
    trial_state = read_trial_state(trial_state_path)
    if not isinstance(trial_state, dict):
        trial_state = {}
    if _is_pruned(trial_state):
        return _tell_pruned(study, trial, trial_state, trial_state_path, trial_index)

    run_results = _read_run_results(trial_state_path, runs)
    run_scores = [s for _, s, _ in run_results]
    valid_scores = [s for s in run_scores if s != float("-inf")]
    if not valid_scores:
        final_score = _fail_trial(cfg, study, trial, trial_state_path, "all runs failed")
        logger.info(f"[score] trial_{trial_index:02d} -> {final_score:.2f} (all runs failed)")
        return final_score

    test_names_in_order, per_test_details = _aggregate_per_test(cfg, run_results)
    if not per_test_details:
        return _fail_trial(cfg, study, trial, trial_state_path, "no valid per-test scores")

    if cfg.project.multi_objective:
        return _tell_multi_objective(
            cfg, trial, study, trial_index, gen_index, trial_state_path,
            per_test_details, run_results, valid_scores,
        )

    return _tell_single_objective(
        cfg, trial, study, trial_index, gen_index, trial_state_path,
        test_names_in_order, per_test_details, run_results, valid_scores,
    )


def _tell_single_objective(
    cfg, trial, study, trial_index, gen_index, trial_state_path,
    test_names_in_order, per_test_details, run_results, valid_scores,
) -> float:
    """Single-objective path: gate, weight, combine, tell, write the summary."""
    test_names = list(cfg.selected_names)
    run_scores = [s for _, s, _ in run_results]
    counted_names, counted_weights, gate_open = _gate_and_weight(
        cfg, test_names_in_order, per_test_details
    )
    counted_scores = [per_test_details[name]["aggregate_score"] for name in counted_names]
    counted_max_scores = {name: cfg.test_max_scores.get(name, 1.0) for name in counted_names}

    final_score = combine_weighted(
        counted_scores, counted_names, counted_weights, counted_max_scores
    )
    aggregate_score = final_score
    median_score = statistics.median(counted_scores) if counted_scores else 0.0
    tell_safely(study, trial, final_score)

    update_trial_summary(
        trial_state_path,
        {
            "trial": trial_index,
            "gen": gen_index,
            "state": "done",
            "n_runs": len(valid_scores),
            "test_names": list(test_names),
            "test_weights": {
                name: round(cfg.test_weights.get(name, 0.0) * 100, 1) for name in test_names
            },
            "gated_test_names": list(cfg.gated_test_names),
            "gate_open": gate_open,
            "active_test_names": list(counted_names),
            "active_test_weights": {
                name: round(counted_weights.get(name, 0.0) * 100, 1) for name in counted_names
            },
            "test_max_scores": {name: cfg.test_max_scores.get(name, 1.0) for name in test_names},
            "run_test_names": [name for name, _, _ in run_results],
            "test_scores": per_test_details,
            "avg_score": round(sum(valid_scores) / len(valid_scores), 4),
            "median_score": round(median_score, 4),
            "aggregate_score": round(aggregate_score, 4),
            "best_run": round(max(valid_scores), 4),
            "worst_run": round(min(valid_scores), 4),
            "final_score": round(final_score, 4),
            "run_scores": [round(s, 4) if s != float("-inf") else None for s in run_scores],
        },
    )
    logger.info(
        f"\n[score] trial_{trial_index:02d} -> {final_score:.2f}/100 "
        f"(weighted_normalized_agg: {aggregate_score:.2f}, gate={'open' if gate_open else 'closed'})"
    )
    return final_score
