"""Finalize phase: wait for runs, apply gating/pruning/relaunch, score, summarize.

The scan loop reads as named phases per trial: timeout scan → probe relaunch →
gate open/skip → score. Everything is non-blocking so the wall-clock pruner
keeps running while relaunches wait for capacity.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import optuna

from sofaopt.core.algorithm import _finalize_trial_score
from sofaopt.core.generation.launch import _relaunch_run
from sofaopt.core.generation.plan import prune_trial, trial_has_ungated_positive_run
from sofaopt.core.generation.types import LaunchedTrial, LaunchResult, RunHistory
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import write_gen_summary
from sofaopt.core.sofa_runner import active_sofa_process_count
from sofaopt.core.trial_state import read_trial_run, read_trial_state, update_trial_run
from sofaopt.core.trialprep import render_preview

logger = logging.getLogger(__name__)

_TERMINAL = {"done", "failed", "error", "pruned", "skipped", "cancelled"}

_BAR_WIDTH = 28


def finalize_generation(
    cfg: RunConfig,
    *,
    gen_index: int,
    study: optuna.Study,
    state: RunHistory,
    env: dict,
    gen_dir: Path,
    trial_state_paths_by_trial: list[Path],
    launch_result: LaunchResult,
) -> None:
    """Drive launched trials to completion and record their scores."""
    finalizer = _GenerationFinalizer(
        cfg=cfg,
        gen_index=gen_index,
        study=study,
        state=state,
        gen_dir=gen_dir,
        trial_state_paths_by_trial=trial_state_paths_by_trial,
        launch=launch_result,
    )
    try:
        finalizer.run()
    finally:
        _cleanup_trial_assets(launch_result.assets_by_trial)


@dataclass
class _GenerationFinalizer:
    """One generation's wait/gate/score coordinator (private to this module)."""

    cfg: RunConfig
    gen_index: int
    study: optuna.Study
    state: RunHistory
    gen_dir: Path
    trial_state_paths_by_trial: list[Path]
    launch: LaunchResult

    finalized: set[int] = field(default_factory=set)
    gen_scores: list[float] = field(default_factory=list)
    relaunch_counts: dict[tuple[int, int], int] = field(default_factory=dict)

    def run(self) -> None:
        project = self.cfg.project
        self.gen_scores = list(self.launch.prelaunch_scores)
        start_time = time.time()
        last_print = 0.0

        while len(self.finalized) < len(self.launch.trials):
            for entry in self.launch.trials:
                if entry.trial_index in self.finalized:
                    continue
                if self._scan_runs(entry):
                    continue  # still has an active (or just relaunched) run
                self._settle_trial(entry)

            now = time.time()
            if now - last_print >= 0.5:
                self._print_progress(start_time)
                last_print = now
            if len(self.finalized) < len(self.launch.trials):
                time.sleep(0.2)

        self._print_progress(start_time, final=True)
        self._render_previews()

        if project.on_generation_end is not None:
            try:
                project.on_generation_end(
                    self.gen_index, list(self.trial_state_paths_by_trial)
                )
            except Exception as e:
                logger.warning(f"[warn] on_generation_end hook failed: {e}")

        write_gen_summary(self.gen_dir, self.gen_index, self.gen_scores)

    # -- phases ---------------------------------------------------------------

    def _scan_runs(self, entry: LaunchedTrial) -> bool:
        """Timeout-prune wedged runs and relaunch finished probes.

        Returns True while the trial still has an active run (it cannot be
        settled yet this pass).
        """
        wall_timeout = self.cfg.project.sofa_realtime_timeout
        has_active = False
        now = time.time()
        for proc, _path, run_slot in list(entry.runs):
            if proc.poll() is None:
                launch_ts = entry.launch_times_by_slot.get(run_slot)
                if launch_ts is not None and now - launch_ts > wall_timeout:
                    prune_trial(
                        self.cfg, self.gen_index, entry.trial_index,
                        entry.trial_state_path, entry.runs,
                        f"SOFA wall-clock timeout after {wall_timeout:.0f}s",
                    )
                has_active = True
                continue
            if self._handle_probe_exit(entry, run_slot):
                has_active = True
        return has_active

    def _handle_probe_exit(self, entry: LaunchedTrial, run_slot: int) -> bool:
        """Deal with a relaunchable probe that exited non-terminally.

        A probe that flagged ``probe_finished`` wants another iteration;
        one that didn't crashed mid-run and is failed (no retry on a
        deterministic crash). Returns True when the slot is (about to be)
        active again.
        """
        run_data = read_trial_run(entry.trial_state_path, run_slot) or {}
        run_state = str(run_data.get("state", "")).lower()
        test_name = str(run_data.get("test_name", ""))
        relaunchable = {t.name for t in self.cfg.selected_tests if t.relaunchable}
        if test_name not in relaunchable or run_state in _TERMINAL:
            return False

        max_relaunches = self.cfg.project.max_run_relaunches
        key = (entry.trial_index, run_slot)
        if not run_data.get("probe_finished"):
            update_trial_run(
                entry.trial_state_path, run_slot,
                {
                    "state": "failed", "score": None,
                    "reason": "SOFA exited mid-run without completing a probe (crash)",
                },
            )
            return False
        if self.relaunch_counts.get(key, 0) >= max_relaunches:
            update_trial_run(
                entry.trial_state_path, run_slot,
                {
                    "state": "failed", "score": None,
                    "reason": f"exceeded {max_relaunches} probe relaunches",
                },
            )
            return False
        if self._at_capacity():
            # Defer the relaunch to the next scan pass instead of blocking
            # (blocking here would stall wall-timeout pruning of other trials).
            return True
        self.relaunch_counts[key] = self.relaunch_counts.get(key, 0) + 1
        _relaunch_run(
            self.cfg, entry,
            launched=self.launch.trials,
            gen_index=self.gen_index,
            run_slot=run_slot, test_name=test_name,
            test_run_index=int(run_data.get("test_run_index", run_slot)),
            test_run_total=int(run_data.get("test_run_total", 1)),
        )
        return True

    def _settle_trial(self, entry: LaunchedTrial) -> None:
        """All runs exited: score the trial, or open/skip its gated tests."""
        trial_state = read_trial_state(entry.trial_state_path)
        if str(trial_state.get("state", "")).lower() == "pruned":
            self._finalize_trial(entry)
            return

        if entry.pending_gated_runs:
            if trial_has_ungated_positive_run(self.cfg, entry.trial_state_path):
                if self._at_capacity():
                    return  # defer gated launches to the next pass
                self._launch_gated_runs(entry)
                return  # gated runs are now active; settle on a later pass
            for run_slot, *_ in list(entry.pending_gated_runs):
                update_trial_run(
                    entry.trial_state_path, run_slot,
                    {
                        "state": "skipped", "score": None,
                        "reason": "gated_test_skipped_until_ungated_success",
                    },
                )
            entry.pending_gated_runs.clear()

        self._finalize_trial(entry)

    def _launch_gated_runs(self, entry: LaunchedTrial) -> None:
        logger.info(
            f"[gate] Gen {self.gen_index:04d} Trial {entry.trial_index:02d} "
            f"ungated success; launching gated tests."
        )
        for run_slot, t_name, t_idx, t_total in list(entry.pending_gated_runs):
            _relaunch_run(
                self.cfg, entry,
                launched=self.launch.trials,
                gen_index=self.gen_index,
                run_slot=run_slot, test_name=t_name,
                test_run_index=t_idx, test_run_total=t_total,
            )
        entry.pending_gated_runs.clear()

    def _finalize_trial(self, entry: LaunchedTrial) -> None:
        self.finalized.add(entry.trial_index)
        final_score = _finalize_trial_score(
            self.cfg,
            trial_index=entry.trial_index,
            trial=entry.trial,
            runs=entry.runs,
            trial_state_path=entry.trial_state_path,
            study=self.study,
            gen_index=self.gen_index,
        )
        self.gen_scores.append(final_score)
        self.state.record_score(final_score)

    # -- support --------------------------------------------------------------

    def _at_capacity(self) -> bool:
        return (
            active_sofa_process_count(self.launch.trials)
            >= self.cfg.project.max_active_sofa_procs
        )

    def _print_progress(self, start_time: float, final: bool = False) -> None:
        total_runs = sum(len(t.runs) for t in self.launch.trials)
        if final:
            total_done = total_runs
        else:
            total_done = sum(
                sum(1 for pr, _, _ in t.runs if pr.poll() is not None)
                for t in self.launch.trials
            )
        pct = (100.0 * total_done / total_runs) if total_runs else 100.0
        filled = int(_BAR_WIDTH * total_done / total_runs) if total_runs else _BAR_WIDTH
        bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
        # In-place \r progress bar — must stay a print(): logging has no end=/flush=
        # and would emit one record per poll tick.
        print(
            f"\r[progress] Gen {self.gen_index:04d} SOFA [{bar}] "
            f"{total_done}/{total_runs} ({pct:5.1f}%)  "
            f"elapsed {time.time() - start_time:5.1f}s",
            end="" if not final else "\n",
            flush=True,
        )

    def _render_previews(self) -> None:
        """Render previews now that all SOFA GL contexts for this gen are gone."""
        if not self.launch.preview_tasks:
            return
        logger.info(
            f"[preview] Gen {self.gen_index:04d} rendering "
            f"{len(self.launch.preview_tasks)} preview(s)"
        )
        for image, trial_index in self.launch.preview_tasks:
            trial_dir = self.gen_dir / f"trial_{trial_index:02d}"
            render_preview(
                Path(image), trial_dir, self.gen_index, trial_index,
                self.cfg.project.previews_dir, self.launch.failed_preview,
            )


def _cleanup_trial_assets(assets_by_trial: dict) -> None:
    """Best-effort deletion of per-trial assets flagged by the prepare hooks."""
    for paths in assets_by_trial.values():
        for asset in paths if isinstance(paths, list) else [paths]:
            try:
                p = Path(asset)
                if p.exists():
                    p.unlink()
            except Exception as exc:
                logger.warning(f"[warn] Could not delete trial asset {asset}: {exc}")
