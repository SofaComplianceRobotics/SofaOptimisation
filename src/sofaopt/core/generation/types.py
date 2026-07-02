"""Shared structures threaded through the launch and finalize phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunHistory:
    """Final trial scores recorded across the whole run.

    Feeds the live progress file and the console best-so-far lines. (Replaces
    the old ``TrialState`` class, most of which was dead caching.)
    """

    all_scores: list[float] = field(default_factory=list)

    def record_score(self, score: float) -> None:
        self.all_scores.append(score)


@dataclass
class LaunchedTrial:
    """One candidate's launch state for a generation.

    ``runs`` holds ``(Popen, trial_state_path, run_slot)`` tuples for every
    SOFA process started for this trial (relaunches replace their slot's
    entry in place). ``pending_gated_runs`` holds ``(run_slot, test_name,
    test_run_index, test_run_total)`` for gated tests not yet launched.
    """

    trial_index: int
    trial: Any  # optuna.trial.Trial
    trial_state_path: Path
    params_path: Path
    trial_env: dict[str, str]
    runs: list[tuple] = field(default_factory=list)
    pending_gated_runs: list[tuple[int, str, int, int]] = field(default_factory=list)
    launch_times_by_slot: dict[int, float] = field(default_factory=dict)


@dataclass
class LaunchResult:
    """Everything the launch phase hands to the finalize phase."""

    trials: list[LaunchedTrial] = field(default_factory=list)
    assets_by_trial: dict[int, list[Path]] = field(default_factory=dict)
    prelaunch_scores: list[float] = field(default_factory=list)
    preview_tasks: list[tuple[Path, int]] = field(default_factory=list)
    failed_preview: Path | None = None
