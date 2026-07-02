"""Score normalization, aggregation, and progress reporting.

The score pipeline has exactly one implementation and one order
(dev_guidelines §11): per-run scores → :func:`aggregate_repeats` per test →
:func:`normalize_test_score` by the test's ``max_score`` →
:func:`combine_weighted` across tests → the 0–100 study objective.
"""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path

from sofaopt.core.io import write_json
from sofaopt.core.runconfig import RunConfig

logger = logging.getLogger(__name__)


def normalize_test_score(score: float, max_score: float) -> float:
    """Normalize a raw test score to [0, 1] by dividing by its declared maximum.

    Scores above the maximum are clamped to 1.0 rather than rewarded further.
    """
    if max_score <= 0:
        return 0.0
    return min(score / max_score, 1.0)


def aggregate_repeats(scores: list[float], aggregation: str = "mean") -> float:
    """Combine one test's repeat scores into a single per-test score.

    Modes: ``"mean"`` (default), ``"median"``, ``"sum"``, and
    ``"exponential_coverage"`` (rewards covering multiple scenarios: each
    additional positive run multiplies the sum by 1.5).
    """
    if not scores:
        return 0.0
    if aggregation == "median":
        return statistics.median(scores)
    if aggregation == "sum":
        return sum(scores)
    if aggregation == "exponential_coverage":
        n_positive = sum(1 for s in scores if s > 0)
        multiplier = 1.5 ** (n_positive - 1) if n_positive > 0 else 0.0
        return sum(scores) * multiplier
    return sum(scores) / len(scores)


def combine_weighted(
    per_test_scores: list[float],
    names: list[str],
    weights: dict[str, float],
    max_scores: dict[str, float],
) -> float:
    """Combine per-test aggregate scores into the 0–100 study objective.

    ``Σ min(score_i / max_i, 1.0) * weight_pct_i`` with ``weights`` as
    fractions summing to 1 over the counted tests.
    """
    return sum(
        normalize_test_score(score, max_scores[name]) * (weights[name] * 100)
        for score, name in zip(per_test_scores, names)
    )


def write_gen_summary(gen_dir: Path, gen_index: int, scores: list[float]) -> None:
    """Write summary.json (avg/best/worst) for a finished generation."""
    valid_scores = [s for s in scores if s not in (float("-inf"), None)]

    summary = {
        "gen": gen_index,
        "n_trials": len(scores),
        "n_valid": len(valid_scores),
        "avg_score": (
            round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None
        ),
        "best_score": round(max(valid_scores), 4) if valid_scores else None,
        "worst_score": round(min(valid_scores), 4) if valid_scores else None,
    }
    write_json(gen_dir / "summary.json", summary)
    avg_str = f"{summary['avg_score']:.2f}" if summary["avg_score"] is not None else "n/a"
    best_str = (
        f"{summary['best_score']:.2f}" if summary["best_score"] is not None else "n/a"
    )
    logger.info(
        f"[summary] Gen {gen_index:04d} - "
        f"avg: {avg_str}/100  best: {best_str}/100  "
        f"({len(valid_scores)}/{len(scores)} trials)"
    )


def write_progress(
    cfg: RunConfig,
    gen_index: int,
    trials_done_in_gen: float,
    all_scores: list[float],
    started_at: float = 0.0,
    total_gens: int | None = None,
) -> None:
    """Write progress.json for the dashboard to poll.

    ``total_gens`` is the absolute last generation of this run (offset +
    n_generations when resuming); without it a resumed run would report > 100%.
    """
    project = cfg.project
    n_parallel = project.n_parallel
    n_generations = total_gens if total_gens is not None else project.n_generations

    trials_done_in_gen = max(0.0, min(float(n_parallel), float(trials_done_in_gen)))
    total_done = (gen_index - 1) * n_parallel + trials_done_in_gen
    total = n_generations * n_parallel
    valid_scores = [s for s in all_scores if s not in (float("-inf"), None)]

    payload = {
        "gen_current": gen_index,
        "gen_total": n_generations,
        "trials_per_gen": n_parallel,
        "runs_per_trial": cfg.n_repeats,
        "test_names": list(cfg.selected_names),
        "test_weights": {
            name: round(frac * 100) for name, frac in cfg.test_weights.items()
        },
        "run_plan": [
            {
                "test_name": test_name,
                "test_run_index": test_run_index,
                "test_run_total": test_run_total,
                "run_label": f"{test_name} {test_run_index}/{test_run_total}",
            }
            for test_name, test_run_index, test_run_total in cfg.run_plan
        ],
        "tests_per_trial": len(cfg.selected_names),
        "trial_current": total_done,
        "trial_total": total,
        "pct": round(100 * total_done / total, 1) if total else 0.0,
        "best_score": round(max(valid_scores), 4) if valid_scores else None,
        "avg_score": (
            round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None
        ),
        "started_at": started_at,
        "updated_at": time.time(),
    }

    # Atomic: the dashboard polls this file while the run writes it.
    write_json(project.progress_file, payload)
