"""Adaptive re-evaluation (racing) of noisy repeats.

A test with ``run_count_min`` set starts each trial with only that many
repeats. Once every launched run has finished, repeats are added one at a
time — but only while the trial's *optimistic* final score (the raced tests'
aggregates shifted up by a 95% confidence half-width) still reaches the
study incumbent. A candidate whose confidence interval lies entirely below
the incumbent cannot change the ranking at the top, so its remaining repeats
are skipped; contenders (including a potential new incumbent) run the full
``run_count`` so the score every later decision compares against is precise.

Why the restrictions:

- Racing requires ``score_aggregation == "mean"`` (enforced by ``TestSpec``):
  under mean-aggregation the repeats are i.i.d. noise samples of one
  scenario, so a t-based confidence interval on their mean is meaningful.
  ``sum`` / ``exponential_coverage`` repeats are *different scenarios* —
  skipping some would change what the score measures, not its precision.
- Multi-objective runs have no scalar incumbent to race against, so racing
  is disabled there (every repeat launches, as before).
- The incumbent is ``study.best_value`` — the same scalar the stall tracker
  and dashboard use. Before any trial completes (generation 1) there is no
  incumbent and every candidate runs its full repeat count: those trials are
  the baseline everything else races against.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import optuna

from sofaopt.core.runconfig import RunConfig

# Two-sided 95% Student-t critical values by degrees of freedom (df = n - 1).
# Beyond df 30 the normal approximation is within ~2% and racing decisions
# are threshold comparisons, not p-values — the z value is plenty.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z_95 = 1.960


@dataclass(frozen=True)
class RaceVerdict:
    """Outcome of one racing check for a trial."""

    keep_running: bool
    reason: str


def confidence_half_width(scores: list[float]) -> float | None:
    """Two-sided 95% t half-width of the mean; None when n < 2 (unknown)."""
    n = len(scores)
    if n < 2:
        return None
    sd = statistics.stdev(scores)
    return _T_95.get(n - 1, _Z_95) * sd / math.sqrt(n)


def deferred_by_racing(cfg: RunConfig, test_name: str, test_run_index: int) -> bool:
    """True when this repeat waits for a racing decision instead of launching."""
    if cfg.project.multi_objective:
        return False
    minimum = cfg.project.test(test_name).run_count_min
    return minimum is not None and test_run_index > minimum


def incumbent_score(study: optuna.Study) -> float | None:
    """The study's best completed value, or None while nothing completed."""
    try:
        return float(study.best_value)
    except ValueError:
        return None


def evaluate_race(
    cfg: RunConfig,
    study: optuna.Study,
    run_results: list[tuple],
    pending_test_names: set[str],
) -> RaceVerdict:
    """Decide whether a trial with pending raced repeats needs another one.

    ``run_results`` is the per-slot ``(test_name, score, run_total)`` list of
    the runs launched so far (crashed runs score ``-inf`` and aggregate as
    0.0, exactly as in the final scoring); ``pending_test_names`` are the
    raced tests that still have unlaunched repeats.
    """
    # Imported here, not at module top: algorithm.py sits above the
    # generation package in the import graph (finalize -> algorithm).
    from sofaopt.core.algorithm import _aggregate_per_test

    incumbent = incumbent_score(study)
    if incumbent is None:
        return RaceVerdict(True, "no incumbent yet")

    names_in_order, details = _aggregate_per_test(cfg, run_results)
    upper_bounds: dict[str, float] = {}
    for name in pending_test_names:
        d = details.get(name)
        if d is None:
            return RaceVerdict(True, f"{name}: no completed repeats yet")
        hw = confidence_half_width(d["run_scores"])
        if hw is None:
            return RaceVerdict(
                True, f"{name}: need >= 2 repeats for a confidence interval"
            )
        upper_bounds[name] = d["aggregate_score"] + hw

    upper = _optimistic_final_score(cfg, names_in_order, details, upper_bounds)
    if upper <= incumbent:
        return RaceVerdict(
            False,
            f"CI upper bound {upper:.2f} cannot beat incumbent {incumbent:.2f}",
        )
    return RaceVerdict(
        True, f"CI upper bound {upper:.2f} overlaps incumbent {incumbent:.2f}"
    )


def _optimistic_final_score(
    cfg: RunConfig,
    names_in_order: list[str],
    details: dict[str, dict],
    upper_bounds: dict[str, float],
) -> float:
    """Final score if every raced test landed at its CI upper bound.

    Reuses the real gate/weight/combine pipeline on patched aggregates, so
    max-score clamping and weight renormalization behave exactly as they
    will at final scoring.
    """
    from sofaopt.core.algorithm import _gate_and_weight
    from sofaopt.core.scoring import combine_weighted

    patched = {name: dict(d) for name, d in details.items()}
    for name, value in upper_bounds.items():
        patched[name]["aggregate_score"] = value
    counted, weights, _gate_open = _gate_and_weight(cfg, names_in_order, patched)
    scores = [patched[name]["aggregate_score"] for name in counted]
    max_scores = {name: cfg.test_max_scores.get(name, 1.0) for name in counted}
    return combine_weighted(scores, counted, weights, max_scores)
