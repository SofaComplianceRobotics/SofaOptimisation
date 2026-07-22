"""Live per-generation progress estimation and background writer."""

from __future__ import annotations

import threading
from pathlib import Path

from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import write_progress
from sofaopt.core.trial_state import read_trial_state

GEN_PROGRESS_POLL_INTERVAL = 0.25  # seconds between progress writes

_TERMINAL = {"done", "skipped", "failed", "error", "cancelled", "pruned"}


def _run_progress(data: dict) -> float:
    """Progress contribution of one run slot, in [0, 1]."""
    if str(data.get("state", "")).lower() in _TERMINAL:
        return 1.0
    # A relaunchable probe mid-iteration has no meaningful frame fraction.
    if data.get("probe_finished") and data.get("score") is None:
        return 0.0

    cur = float(data.get("current_frame", 0) or 0)
    total_frames = data.get("total_frames")
    if isinstance(total_frames, int) and total_frames > 0:
        return max(0.0, min(1.0, cur / float(total_frames)))
    # A run with a recorded stop reason but no frame counter is over.
    return 1.0 if str(data.get("reason", "")).lower() else 0.0


def _trial_runs(trial_state_path: Path) -> list:
    try:
        runs = (read_trial_state(trial_state_path) or {}).get("runs", [])
        return runs if isinstance(runs, list) else []
    except Exception:
        return []


def generation_progress_fraction(trial_state_paths_by_trial: list[Path]) -> float:
    """Estimate generation progress in [0, 1] from per-run frame counts."""
    if not trial_state_paths_by_trial:
        return 0.0

    total = 0.0
    for trial_state_path in trial_state_paths_by_trial:
        runs = _trial_runs(trial_state_path)
        if runs:
            trial_total = sum(
                _run_progress(r if isinstance(r, dict) else {}) for r in runs
            )
            total += trial_total / len(runs)

    return max(0.0, min(1.0, total / len(trial_state_paths_by_trial)))


def generation_progress_writer(
    cfg: RunConfig,
    gen_index: int,
    trial_state_paths_by_trial: list[Path],
    all_scores: list[float],
    stop_event: threading.Event,
    started_at: float = 0.0,
    total_gens: int | None = None,
    restart_state: dict | None = None,
) -> None:
    """Write progress.json on a fixed interval until ``stop_event`` is set.

    ``restart_state`` is the frozen per-generation IPOP snapshot — stall/restart
    state only changes between generations, so re-writing the same snapshot
    alongside the live trial fraction is correct.
    """
    n_parallel = cfg.project.n_parallel
    while not stop_event.is_set():
        frac = generation_progress_fraction(trial_state_paths_by_trial)
        write_progress(
            cfg, gen_index, frac * n_parallel, all_scores, started_at,
            total_gens=total_gens, restart_state=restart_state,
        )
        stop_event.wait(GEN_PROGRESS_POLL_INTERVAL)
