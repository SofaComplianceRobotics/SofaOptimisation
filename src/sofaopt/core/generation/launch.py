"""Launch phase: prepare each trial, start its SOFA runs, record launch failures."""

from __future__ import annotations

import time
from pathlib import Path

import optuna

from sofaopt.core.runconfig import RunConfig
from sofaopt.core.sofa_runner import (
    active_sofa_process_count,
    launch_sofa,
    wait_for_slot,
)
from sofaopt.core.state import TrialState
from sofaopt.core.trial_state import update_trial_run, update_trial_summary
from sofaopt.core.trialprep import params_from_trial, prepare_trial, render_preview


def _mark_all_runs(trial_state_path: Path, run_count: int, state: str) -> None:
    """Set the same pre-launch lifecycle state on every run slot of a trial.

    Reason is left empty on purpose: a frame-less run with a non-empty reason is
    counted as complete by the progress estimator, which would inflate the bar
    during these intermediate phases.
    """
    for run_slot in range(1, run_count + 1):
        update_trial_run(trial_state_path, run_slot, {"state": state, "reason": ""})


def _launch_one_run(
    cfg: RunConfig,
    *,
    gen_index: int,
    trial_index: int,
    run_slot: int,
    test_name: str,
    test_run_index: int,
    test_run_total: int,
    trial_state_path: Path,
    params_path: Path,
    trial_env: dict,
    launch_times_by_slot: dict[int, float],
) -> tuple:
    """Mark a slot launching and start its SOFA process. Returns the run tuple."""
    scene_file = cfg.project.test(test_name).scene_file
    update_trial_run(
        trial_state_path,
        run_slot,
        {
            "state": "launching",
            "current_frame": 0,
            "total_frames": None,
            "sim_time": 0.0,
            "score": None,
            "reason": "",
            "probe_finished": False,
        },
    )
    proc = launch_sofa(
        cfg.project,
        scene_file=scene_file,
        test_name=test_name,
        test_run_index=test_run_index,
        test_run_total=test_run_total,
        trial_state_path=trial_state_path,
        params_path=params_path,
        run_slot=run_slot,
        gen_index=gen_index,
        trial_index=trial_index,
        run_index=run_slot,
        env=trial_env,
    )
    print(
        f"[sofa] Gen {gen_index:04d} Trial {trial_index:02d} "
        f"Run {run_slot}/{cfg.n_repeats} [{test_name} {test_run_index}/{test_run_total}]"
    )
    launch_times_by_slot[run_slot] = time.time()
    return (proc, trial_state_path, run_slot)


def _relaunch_run(cfg: RunConfig, *, runs: list[tuple], **kwargs) -> None:
    """Relaunch one run in-place (probe iteration or deferred gated launch)."""
    run_slot = kwargs["run_slot"]
    new_run = _launch_one_run(cfg, **kwargs)
    for i, (_p, _path, old_slot) in enumerate(runs):
        if old_slot == run_slot:
            runs[i] = new_run
            break
    else:
        runs.append(new_run)


def launch_generation_trials(
    cfg: RunConfig,
    *,
    gen_index: int,
    trials: list,
    study: optuna.Study,
    env: dict,
    state: TrialState,
    gen_dir: Path,
    trial_state_paths_by_trial: list[Path],
) -> dict:
    """Prepare and launch every trial of one generation."""
    project = cfg.project
    hard_fail = project.hard_fail_score
    gated = set(cfg.gated_test_names)
    run_plan = cfg.run_plan
    max_active = project.max_active_sofa_procs

    processes: list[tuple] = []
    assets_by_trial: dict[int, list[Path]] = {}
    prelaunch_scores: list[float] = []
    # Previews are rendered at generation end: offscreen GL contends with SOFA's
    # GL init on Windows and can hang scene startup if done concurrently.
    preview_tasks: list[tuple[Path, int]] = []
    failed_preview = project.failed_preview_image

    for i, trial in enumerate(trials):
        trial_index = i + 1
        trial_dir = gen_dir / f"trial_{trial_index:02d}"
        trial_dir.mkdir(exist_ok=True)
        trial_state_path = trial_dir / "trial_state.json"
        params_path = trial_dir / "params.json"

        if active_sofa_process_count(processes) >= max_active:
            _mark_all_runs(trial_state_path, len(run_plan), "waiting-slot")
        wait_for_slot(processes, max_active, gen_index, trial_index)

        params = params_from_trial(trial, project)

        try:
            _mark_all_runs(trial_state_path, len(run_plan), "preparing")
            prep = prepare_trial(project, params, trial_dir)
            trial_env = {**env, **prep.env}
            assets_by_trial[trial_index] = list(prep.cleanup)
            if prep.preview_image is not None:
                preview_tasks.append((Path(prep.preview_image), trial_index))

            runs: list[tuple] = []
            pending_gated_runs: list[tuple[int, str, int, int]] = []
            launch_times_by_slot: dict[int, float] = {}

            for r, (test_name, test_run_index, test_run_total) in enumerate(run_plan):
                run_slot = r + 1
                if test_name in gated:
                    pending_gated_runs.append(
                        (run_slot, test_name, test_run_index, test_run_total)
                    )
                    update_trial_run(
                        trial_state_path,
                        run_slot,
                        {
                            "state": "pending",
                            "score": None,
                            "reason": "gated_test_waiting_for_ungated_success",
                        },
                    )
                    continue
                runs.append(
                    _launch_one_run(
                        cfg,
                        gen_index=gen_index,
                        trial_index=trial_index,
                        run_slot=run_slot,
                        test_name=test_name,
                        test_run_index=test_run_index,
                        test_run_total=test_run_total,
                        trial_state_path=trial_state_path,
                        params_path=params_path,
                        trial_env=trial_env,
                        launch_times_by_slot=launch_times_by_slot,
                    )
                )

            processes.append(
                (
                    trial_index,
                    trial,
                    runs,
                    pending_gated_runs,
                    trial_state_path,
                    trial_env,
                    params_path,
                    launch_times_by_slot,
                )
            )

        except Exception as e:
            print(f"[error] Gen {gen_index:04d} Trial {trial_index:02d}: {e}")
            if failed_preview is not None:
                render_preview(
                    failed_preview, trial_dir, gen_index, trial_index,
                    project.previews_dir, failed_preview,
                )
            for r in range(len(run_plan)):
                update_trial_run(
                    trial_state_path,
                    r + 1,
                    {"state": "failed", "score": None, "reason": str(e)},
                )
            study.tell(trial, hard_fail)
            update_trial_summary(
                trial_state_path,
                {
                    "state": "failed",
                    "final_score": hard_fail,
                    "outcome": f"prepare failed: {type(e).__name__}",
                },
            )
            prelaunch_scores.append(hard_fail)
            state.record_score(hard_fail)

    return {
        "processes": processes,
        "assets_by_trial": assets_by_trial,
        "prelaunch_scores": prelaunch_scores,
        "failed_preview": failed_preview,
        "preview_tasks": preview_tasks,
    }
