"""Canonical read path for recorded optimization results.

The optimizer records each trial's outcome in ``trial_state.json``
(``final_score``, ``aggregate_score``, per-test ``test_scores`` with
``normalized_score``, gate-renormalized ``active_test_weights``,
``objective_values``). Everything that *displays or ranks* trials — the
dashboard, video selection, summaries — must read those recorded values
through this module, never recompute its own score: a recompute that drifts
from the study objective silently lies about what the optimizer optimized.

Reconstruction from raw run slots exists only as a fallback for legacy
runtime dirs that predate the recorded summary fields.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

# Repeat-aggregation guess used ONLY when reconstructing legacy state files
# that recorded neither final_score nor per-test aggregation.
_LEGACY_AGGREGATION = "mean"

_TERMINAL_STATES = {"done", "failed", "error", "pruned", "skipped", "cancelled", "interrupted"}
_FAIL_STATES = {"failed", "error", "pruned", "skipped", "cancelled", "interrupted"}


def load_trial_records(trials_dir: Path) -> list[dict]:
    """Load every trial_state.json into a flat record list, chronological.

    Each record carries the *recorded* ``final_score`` (falling back to a
    legacy reconstruction only when absent) and ``contributions`` — the
    per-test share of the final score using the gate-renormalized weights the
    optimizer actually applied, so a stacked plot sums to the study objective.
    """
    records: list[dict] = []
    chron = 0

    for gen_dir in sorted(Path(trials_dir).glob("gen_*")):
        gen_index = int(gen_dir.name.split("_")[1])
        for trial_dir in sorted(gen_dir.glob("trial_*")):
            trial_index = int(trial_dir.name.split("_")[1])
            trial_state_path = trial_dir / "trial_state.json"
            if not trial_state_path.exists():
                continue
            try:
                trial_state = json.loads(trial_state_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(trial_state, dict):
                continue

            record = _record_from_state(trial_state, gen_dir.name, trial_dir.name)
            record["gen_index"] = gen_index
            record["trial_index"] = trial_index
            record["chron"] = chron
            records.append(record)
            chron += 1

    return records


def rank_completed(records: list[dict]) -> list[dict]:
    """Completed trials sorted best → worst by the recorded final score."""
    return sorted(
        (
            r
            for r in records
            if r.get("is_complete") and r.get("final_score") is not None
        ),
        key=lambda r: r["final_score"],
        reverse=True,
    )


def load_gen_summaries(trials_dir: Path) -> list[dict]:
    """Load each generation's summary.json (written by the optimizer)."""
    summaries = []
    for gen_dir in sorted(Path(trials_dir).glob("gen_*")):
        summary_path = gen_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "gen_index": data["gen"],
                    "avg_score": data["avg_score"],
                    "best_score": data["best_score"],
                    "n_trials": data.get("n_trials"),
                    "n_valid": data.get("n_valid"),
                }
            )
        except Exception:
            continue
    return summaries


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _record_from_state(trial_state: dict, gen_name: str, trial_name: str) -> dict:
    trial_level_state = str(trial_state.get("state", "")).lower()
    is_complete = trial_level_state in _TERMINAL_STATES
    failed = trial_level_state in _FAIL_STATES
    fail_reason = str(trial_state.get("outcome", "") or "").lower()

    runs = trial_state.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    run_scores = [
        float(r.get("score"))
        for r in runs
        if isinstance(r, dict) and isinstance(r.get("score"), (int, float))
    ]

    test_scores = trial_state.get("test_scores") or None
    if not test_scores and runs and not failed:
        test_scores = _reconstruct_test_scores(trial_state, runs)

    final_score = _final_score(trial_state, test_scores, run_scores)
    contributions = _contributions(trial_state, test_scores, final_score)

    return {
        "gen_name": gen_name,
        "trial_name": trial_name,
        "score": final_score,
        "final_score": final_score,
        "contributions": contributions,
        "state": trial_level_state,
        "failed": failed,
        "fail_reason": fail_reason,
        "outcome_reason": fail_reason,
        "n_runs": len(run_scores),
        "run_scores": run_scores,
        "all_run_scores": run_scores,
        "test_scores": test_scores,
        "objective_values": trial_state.get("objective_values"),
        "params": trial_state.get("params") or {},
        "is_complete": is_complete,
    }


def _final_score(
    trial_state: dict, test_scores: dict | None, run_scores: list[float]
) -> float:
    """The trial's score as the study saw it: recorded value first."""
    recorded = trial_state.get("final_score")
    if isinstance(recorded, (int, float)):
        return float(recorded)
    # Legacy fallbacks only (old runtime dirs / pruned trials with no score).
    if test_scores:
        return _normalized_weighted_score(test_scores)
    recorded_agg = trial_state.get("aggregate_score")
    if isinstance(recorded_agg, (int, float)):
        return float(recorded_agg)
    if run_scores:
        return (
            statistics.median(run_scores)
            if _LEGACY_AGGREGATION == "median"
            else statistics.mean(run_scores)
        )
    return 0.0


def _contributions(
    trial_state: dict, test_scores: dict | None, final_score: float
) -> dict[str, float]:
    """Per-test contribution to the 0–100 final score (for stacked plots).

    Uses the recorded ``normalized_score`` and the gate-renormalized
    ``active_test_weights`` the optimizer actually applied, so the stack sums
    to the recorded final score. Tests excluded by the gate contribute 0.
    """
    if not test_scores:
        return {"score": final_score}

    active_weights = trial_state.get("active_test_weights")
    out: dict[str, float] = {}
    for name, info in test_scores.items():
        if not isinstance(info, dict):
            continue
        if isinstance(active_weights, dict):
            wpct = float(active_weights.get(name, 0.0) or 0.0)
        else:  # legacy file: declared weights are the best available
            wpct = float(info.get("weight_pct", 0.0) or 0.0)
        norm = info.get("normalized_score")
        if not isinstance(norm, (int, float)):
            max_score = float(info.get("max_score") or 0.0)
            agg = float(info.get("aggregate_score", 0.0) or 0.0)
            norm = min(agg / max_score, 1.0) if max_score > 0 else agg
        out[name] = float(norm) * wpct

    return out or {"score": final_score}


def _reconstruct_test_scores(trial_state: dict, runs: list) -> dict | None:
    """Rebuild a per-test breakdown from raw run slots (legacy dirs only)."""
    test_weights: dict = trial_state.get("test_weights") or {}
    test_max_scores: dict = trial_state.get("test_max_scores") or {}
    test_run_scores: dict[str, list[float]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        tname = run.get("test_name")
        raw = run.get("score")
        if tname and isinstance(raw, (int, float)):
            test_run_scores.setdefault(tname, []).append(float(raw))

    if not test_run_scores:
        return None

    all_run_test_names = {
        r.get("test_name") for r in runs if isinstance(r, dict) and r.get("test_name")
    }
    total_weight = sum(test_weights.get(t, 1.0) for t in all_run_test_names) or 1.0
    test_scores = {}
    for tname, tscores in test_run_scores.items():
        agg = (
            statistics.median(tscores)
            if _LEGACY_AGGREGATION == "median"
            else statistics.mean(tscores)
        )
        wpct = test_weights.get(tname, 1.0) / total_weight * 100.0
        test_scores[tname] = {
            "run_count": len(tscores),
            "run_scores": tscores,
            "aggregate_score": agg,
            "median_score": statistics.median(tscores),
            "run_total": len(tscores),
            "weight_pct": wpct,
            "max_score": float(test_max_scores.get(tname) or 0.0),
        }
    return test_scores


def _normalized_weighted_score(test_scores: dict) -> float:
    """Legacy recompute of the 0–100 score from a per-test breakdown.

    Only used for state files that never recorded ``final_score``.
    """
    total = 0.0
    total_weight = 0.0
    for info in test_scores.values():
        if not isinstance(info, dict):
            continue
        raw_score = info.get("aggregate_score", 0.0) or 0.0
        max_score = float(info.get("max_score") or 0.0)
        weight_pct = float(info.get("weight_pct", 0.0) or 0.0)
        normalized = (
            min(float(raw_score) / max_score, 1.0) if max_score > 0 else float(raw_score)
        )
        total += normalized * weight_pct
        total_weight += weight_pct

    if total_weight > 0 and abs(total_weight - 100.0) > 1.0:
        total = (total / total_weight) * 100.0
    return total
