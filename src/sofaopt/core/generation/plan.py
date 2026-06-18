"""Generation-local decisions: gating checks and pruning."""

from __future__ import annotations

from pathlib import Path

from sofaopt.core.runconfig import RunConfig
from sofaopt.core.trial_state import (
    read_trial_state,
    update_trial_run,
    update_trial_summary,
)


def trial_has_ungated_positive_run(cfg: RunConfig, trial_state_path: Path) -> bool:
    """True when at least one *ungated* run in the trial has a positive score."""
    trial_state = read_trial_state(trial_state_path)
    runs = trial_state.get("runs", []) if isinstance(trial_state, dict) else []
    if not isinstance(runs, list):
        return False
    gated = set(cfg.gated_test_names)
    for run in runs:
        if not isinstance(run, dict):
            continue
        if str(run.get("test_name", "")) in gated:
            continue
        raw = run.get("score")
        if isinstance(raw, (int, float)) and float(raw) > 0.0:
            return True
    return False


def prune_trial(
    cfg: RunConfig,
    gen_index: int,
    trial_index: int,
    trial_state_path: Path,
    runs: list[tuple],
    reason: str,
) -> None:
    """Kill a trial's active SOFA runs and mark every slot + the trial pruned."""
    for proc, _path, _run_slot in runs:
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    for run_slot in range(1, cfg.n_repeats + 1):
        update_trial_run(
            trial_state_path,
            run_slot,
            {
                "state": "pruned",
                "current_frame": 0,
                "total_frames": None,
                "sim_time": 0.0,
                "score": None,
                "reason": reason,
            },
        )

    update_trial_summary(
        trial_state_path,
        {"state": "pruned", "outcome": reason, "final_score": None},
    )
    print(f"[prune] Gen {gen_index:04d} Trial {trial_index:02d}: {reason}")
