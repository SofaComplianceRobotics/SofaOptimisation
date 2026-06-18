"""CMA-ES study setup and per-trial score finalization."""

from __future__ import annotations

from pathlib import Path

import optuna

from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import aggregate_trial_scores
from sofaopt.core.trial_state import (
    read_trial_run,
    read_trial_state,
    update_trial_summary,
)


def build_cmaes_study(db_path: Path, cfg: RunConfig) -> optuna.Study:
    """Create a fresh Optuna CMA-ES study backed by a SQLite database.

    Deletes any existing database at ``db_path`` first.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
        print(f"[reset] Deleted {db_path.name}")

    project = cfg.project
    sampler = optuna.samplers.CmaEsSampler(
        popsize=project.n_parallel,
        sigma0=project.cmaes_sigma0,
        n_startup_trials=project.cmaes_startup_trials,
        consider_pruned_trials=True,
        x0={
            spec["name"]: spec["default"]
            for spec in cfg.param_specs
            if not (spec["min"] == 0 and spec["max"] == 0)
        },
    )
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
    return optuna.create_study(
        study_name=project.name,
        sampler=sampler,
        direction="maximize",
        storage=storage,
    )


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
    hard_fail = cfg.project.hard_fail_score
    gated_names_cfg = cfg.gated_test_names
    test_aggregations = cfg.test_aggregations
    test_max_scores = cfg.test_max_scores
    test_weights = cfg.test_weights
    test_names = list(cfg.selected_names)

    trial_state = read_trial_state(trial_state_path)
    if not isinstance(trial_state, dict):
        trial_state = {}

    if str(trial_state.get("state", "")).lower() == "pruned" or any(
        isinstance(run, dict) and str(run.get("state", "")).lower() == "pruned"
        for run in trial_state.get("runs", [])
    ):
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        update_trial_summary(
            trial_state_path,
            {
                "state": "pruned",
                "final_score": None,
                "outcome": trial_state.get("outcome", "pruned"),
            },
        )
        print(f"[score] trial_{trial_index:02d} -> pruned (generation pruned)")
        return float("-inf")

    # Group scores by the test name each slot recorded for itself, so attribution
    # survives relaunches/out-of-order gated launches.
    run_results: list[tuple[str, float, int | None]] = []
    for _p, _path, run_slot in runs:
        run_data = read_trial_run(trial_state_path, run_slot) or {}
        raw = run_data.get("score")
        score = float(raw) if isinstance(raw, (int, float)) else float("-inf")
        run_results.append(
            (str(run_data.get("test_name", "")), score, run_data.get("test_run_total"))
        )

    run_scores = [s for _, s, _ in run_results]

    valid_scores = [s for s in run_scores if s != float("-inf")]
    if not valid_scores:
        final_score = hard_fail
        study.tell(trial, final_score)
        print(f"[score] trial_{trial_index:02d} -> {final_score:.2f} (all runs failed)")
        update_trial_summary(
            trial_state_path,
            {"state": "failed", "final_score": final_score, "outcome": "all runs failed"},
        )
        return final_score

    test_names_in_order: list[str] = []
    per_test_scores: list[float] = []
    per_test_details: dict[str, dict] = {}

    for test_name in dict.fromkeys(name for name, _, _ in run_results if name):
        raw_scores = [s for name, s, _ in run_results if name == test_name]
        scores_for_test = [0.0 if s == float("-inf") else s for s in raw_scores]
        crashed_runs = sum(1 for s in raw_scores if s == float("-inf"))
        test_names_in_order.append(test_name)
        test_aggregate, _, _, test_median = aggregate_trial_scores(
            scores_for_test,
            aggregation=test_aggregations.get(test_name, "mean"),
        )
        max_score = test_max_scores.get(test_name, 1.0)
        test_run_total = next(
            (rt for name, _, rt in run_results if name == test_name and rt is not None),
            len(scores_for_test),
        )
        per_test_scores.append(test_aggregate)
        per_test_details[test_name] = {
            "run_count": len(scores_for_test),
            "crashed_run_count": crashed_runs,
            "run_scores": [round(s, 4) for s in scores_for_test],
            "aggregate_score": round(test_aggregate, 4),
            "median_score": round(test_median, 4),
            "run_total": test_run_total,
            "max_score": max_score,
            "weight_pct": round(test_weights.get(test_name, 0.0) * 100, 1),
            "normalized_score": round(
                min(test_aggregate / max_score, 1.0) if max_score > 0 else 0.0, 4
            ),
        }

    if not per_test_scores:
        final_score = hard_fail
        study.tell(trial, final_score)
        update_trial_summary(
            trial_state_path,
            {
                "state": "failed",
                "final_score": final_score,
                "outcome": "no valid per-test scores",
            },
        )
        return final_score

    configured_gated_names = [name for name in test_names if name in gated_names_cfg]
    gated_names = [name for name in test_names_in_order if name in gated_names_cfg]
    ungated_names = [name for name in test_names_in_order if name not in gated_names_cfg]
    gate_open = not configured_gated_names or any(
        per_test_details.get(name, {}).get("aggregate_score", 0.0) > 0.0
        for name in ungated_names
    )
    counted_names = test_names_in_order if gate_open else ungated_names
    if not counted_names:
        counted_names = test_names_in_order

    counted_weight_total = sum(test_weights.get(name, 0.0) for name in counted_names)
    if counted_weight_total > 0:
        counted_weights = {
            name: test_weights.get(name, 0.0) / counted_weight_total
            for name in counted_names
        }
    else:
        counted_weights = {name: 1.0 / len(counted_names) for name in counted_names}

    counted_scores = [per_test_details[name]["aggregate_score"] for name in counted_names]
    counted_max_scores = {name: test_max_scores.get(name, 1.0) for name in counted_names}

    aggregate_score, _, final_score, median_score = aggregate_trial_scores(
        counted_scores,
        weights=counted_weights,
        names=counted_names,
        max_scores=counted_max_scores,
    )
    study.tell(trial, final_score)

    trial_stats = {
        "trial": trial_index,
        "gen": gen_index,
        "state": "done",
        "n_runs": len(valid_scores),
        "test_names": list(test_names),
        "test_weights": {
            name: round(test_weights.get(name, 0.0) * 100, 1) for name in test_names
        },
        "gated_test_names": list(gated_names_cfg),
        "gate_open": gate_open,
        "active_test_names": list(counted_names),
        "active_test_weights": {
            name: round(counted_weights.get(name, 0.0) * 100, 1) for name in counted_names
        },
        "test_max_scores": {name: test_max_scores.get(name, 1.0) for name in test_names},
        "run_test_names": [name for name, _, _ in run_results],
        "test_scores": per_test_details,
        "avg_score": round(sum(valid_scores) / len(valid_scores), 4),
        "median_score": round(median_score, 4),
        "aggregate_score": round(aggregate_score, 4),
        "best_run": round(max(valid_scores), 4),
        "worst_run": round(min(valid_scores), 4),
        "final_score": round(final_score, 4),
        "run_scores": [round(s, 4) if s != float("-inf") else None for s in run_scores],
    }
    update_trial_summary(trial_state_path, trial_stats)

    print(
        f"\n[score] trial_{trial_index:02d} -> {final_score:.2f}/100 "
        f"(weighted_normalized_agg: {aggregate_score:.2f}, gate={'open' if gate_open else 'closed'})"
    )
    return final_score
