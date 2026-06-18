"""Live per-generation progress estimation and background writer."""

from __future__ import annotations

import threading
from pathlib import Path

from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import write_progress
from sofaopt.core.trial_state import read_trial_state

GEN_PROGRESS_POLL_INTERVAL = 0.25  # seconds between progress writes

_TERMINAL = {"done", "skipped", "failed", "error", "cancelled", "pruned"}


def generation_progress_fraction(trial_state_paths_by_trial: list[Path]) -> float:
    """Estimate generation progress in [0, 1] from per-run frame counts."""
    if not trial_state_paths_by_trial:
        return 0.0

    total = 0.0
    for trial_state_path in trial_state_paths_by_trial:
        trial_total = 0.0
        try:
            runs = (read_trial_state(trial_state_path) or {}).get("runs", [])
            if not isinstance(runs, list):
                runs = []
        except Exception:
            runs = []

        for run_data in runs:
            data = run_data if isinstance(run_data, dict) else {}
            run_state = str(data.get("state", "")).lower()
            reason = str(data.get("reason", "")).lower()

            if run_state in _TERMINAL:
                trial_total += 1.0
                continue
            # A relaunchable probe mid-iteration has no meaningful frame fraction.
            if data.get("probe_finished") and data.get("score") is None:
                continue

            cur = float(data.get("current_frame", 0) or 0)
            total_frames = data.get("total_frames")
            if isinstance(total_frames, int) and total_frames > 0:
                trial_total += max(0.0, min(1.0, cur / float(total_frames)))
            elif reason:
                trial_total += 1.0

        if runs:
            total += trial_total / len(runs)

    return max(0.0, min(1.0, total / len(trial_state_paths_by_trial)))


def generation_progress_writer(
    cfg: RunConfig,
    gen_index: int,
    trial_state_paths_by_trial: list[Path],
    all_scores: list[float],
    stop_event: threading.Event,
    started_at: float = 0.0,
) -> None:
    """Write progress.json on a fixed interval until ``stop_event`` is set."""
    n_parallel = cfg.project.n_parallel
    while not stop_event.is_set():
        frac = generation_progress_fraction(trial_state_paths_by_trial)
        write_progress(cfg, gen_index, frac * n_parallel, all_scores, started_at)
        stop_event.wait(GEN_PROGRESS_POLL_INTERVAL)
