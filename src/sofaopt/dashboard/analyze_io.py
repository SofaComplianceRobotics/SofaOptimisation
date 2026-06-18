"""Load trial results and generation summaries from the trials directory."""

from __future__ import annotations

import json
import statistics

from sofaopt.dashboard import context


def _normalized_weighted_score(test_scores: dict) -> float:
    """Recompute the 0–100 final score from per-test breakdown data.

    Formula: ``Σ min(aggregate_i / max_i, 1.0) * weight_pct_i``. Degrades
    gracefully when max_score / weight_pct are missing (older state files).
    """
    total = 0.0
    total_weight = 0.0
    for info in test_scores.values():
        if not isinstance(info, dict):
            continue
        raw_score = info.get("aggregate_score", 0.0) or 0.0
        max_score = float(info.get("max_score") or 0.0)
        weight_pct = float(info.get("weight_pct", 0.0) or 0.0)
        normalized = min(float(raw_score) / max_score, 1.0) if max_score > 0 else float(raw_score)
        total += normalized * weight_pct
        total_weight += weight_pct

    if total_weight > 0 and abs(total_weight - 100.0) > 1.0:
        total = (total / total_weight) * 100.0
    return total


def load_all_trials() -> list[dict]:
    """Load every trial_state.json into a flat record list, ordered chronologically."""
    trials_dir = context.trials_dir()
    aggregation = context.SCORE_AGGREGATION
    records = []
    chron = 0

    terminal_states = {"done", "failed", "error", "pruned", "skipped", "cancelled"}
    fail_states = {"failed", "error", "pruned", "skipped", "cancelled"}

    for gen_dir in sorted(trials_dir.glob("gen_*")):
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

            trial_level_state = str(trial_state.get("state", "")).lower()
            is_complete = trial_level_state in terminal_states
            failed = trial_level_state in fail_states
            fail_reason = str(trial_state.get("outcome", "") or "").lower()

            runs = trial_state.get("runs", [])
            if not isinstance(runs, list):
                runs = []
            run_scores = [
                float(r.get("score"))
                for r in runs
                if isinstance(r, dict) and isinstance(r.get("score"), (int, float))
            ]
            valid = run_scores

            if trial_state.get("aggregate_score") is not None:
                score = float(trial_state["aggregate_score"])
            elif valid:
                score = (
                    statistics.median(valid)
                    if aggregation == "median"
                    else statistics.mean(valid)
                )
            else:
                score = 0.0

            raw_final = trial_state.get("final_score")
            final_score = float(raw_final) if isinstance(raw_final, (int, float)) else score

            test_scores = trial_state.get("test_scores") or None

            if not test_scores and runs and not failed:
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

                if test_run_scores:
                    all_run_test_names = {
                        r.get("test_name")
                        for r in runs
                        if isinstance(r, dict) and r.get("test_name")
                    }
                    total_weight = sum(
                        test_weights.get(t, 1.0) for t in all_run_test_names
                    ) or 1.0
                    test_scores = {}
                    for tname, tscores in test_run_scores.items():
                        agg = (
                            statistics.median(tscores)
                            if aggregation == "median"
                            else statistics.mean(tscores)
                        )
                        wpct = (test_weights.get(tname, 1.0) / total_weight * 100.0) if total_weight else 0.0
                        test_scores[tname] = {
                            "run_count": len(tscores),
                            "run_scores": tscores,
                            "aggregate_score": agg,
                            "median_score": statistics.median(tscores),
                            "run_total": len(tscores),
                            "weight_pct": wpct,
                            "max_score": float(test_max_scores.get(tname) or 0.0),
                        }

            if test_scores:
                final_score = _normalized_weighted_score(test_scores)
                score = final_score

            records.append(
                {
                    "gen_index": gen_index,
                    "trial_index": trial_index,
                    "gen_name": gen_dir.name,
                    "trial_name": trial_dir.name,
                    "score": score,
                    "final_score": final_score,
                    "failed": failed,
                    "fail_reason": fail_reason,
                    "outcome_reason": fail_reason,
                    "n_runs": len(valid),
                    "run_scores": valid,
                    "all_run_scores": run_scores,
                    "test_scores": test_scores,
                    "is_complete": is_complete,
                    "chron": chron,
                }
            )
            chron += 1

    return records


def load_gen_summaries() -> list[dict]:
    """Load generation summaries from each gen's summary.json."""
    summaries = []
    for gen_dir in sorted(context.trials_dir().glob("gen_*")):
        summary_path = gen_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text())
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
