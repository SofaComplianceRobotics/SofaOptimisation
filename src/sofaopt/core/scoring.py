"""Score normalization, aggregation, and progress reporting."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from sofaopt.core.io import write_json
from sofaopt.core.runconfig import RunConfig


def normalize_test_score(score: float, max_score: float) -> float:
    """Normalize a raw test score to [0, 1] by dividing by its declared maximum.

    Scores above the maximum are clamped to 1.0 rather than rewarded further.
    """
    if max_score <= 0:
        return 0.0
    return min(score / max_score, 1.0)


def aggregate_trial_scores(
    valid_scores: list[float],
    weights: dict[str, float] | None = None,
    names: list[str] | None = None,
    max_scores: dict[str, float] | None = None,
    aggregation: str = "mean",
) -> tuple[float, float, float, float]:
    """Aggregate scores using the configured method.

    When ``weights``, ``names`` and ``max_scores`` are all provided, computes a
    0–100 score:  ``Σ min(score_i / max_i, 1.0) * weight_pct_i``. When combining
    repeats of one test, omit those and a plain mean/median/sum is used.

    Supported ``aggregation`` modes for repeats: ``"mean"`` (default),
    ``"median"``, ``"sum"``, and ``"exponential_coverage"`` (rewards covering
    multiple scenarios: each additional positive run multiplies the sum by 1.5).

    Returns:
        (aggregate_score, 0.0, final_score, median_score). The second element is
        a retained-for-compatibility penalty slot, always 0.0; ``final_score``
        equals ``aggregate_score``.
    """
    if not valid_scores:
        return 0.0, 0.0, 0.0, 0.0

    avg_score = sum(valid_scores) / len(valid_scores)
    median_score = statistics.median(valid_scores)

    if aggregation == "sum":
        aggregate_score = sum(valid_scores)
        return aggregate_score, 0.0, aggregate_score, median_score

    if aggregation == "exponential_coverage":
        n_positive = sum(1 for s in valid_scores if s > 0)
        multiplier = 1.5 ** (n_positive - 1) if n_positive > 0 else 0.0
        aggregate_score = sum(valid_scores) * multiplier
        return aggregate_score, 0.0, aggregate_score, median_score

    if (
        weights is not None
        and names is not None
        and max_scores is not None
        and len(names) == len(valid_scores)
        and all(name in weights for name in names)
        and all(name in max_scores for name in names)
    ):
        aggregate_score = sum(
            normalize_test_score(score, max_scores[name]) * (weights[name] * 100)
            for score, name in zip(valid_scores, names)
        )
    elif aggregation == "median":
        aggregate_score = median_score
    else:
        aggregate_score = avg_score

    return aggregate_score, 0.0, aggregate_score, median_score


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
    print(
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
) -> None:
    """Write progress.json for the dashboard to poll."""
    project = cfg.project
    n_parallel = project.n_parallel
    n_generations = project.n_generations

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

    project.progress_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
