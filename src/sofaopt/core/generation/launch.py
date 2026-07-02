"""Launch phase: prepare each trial, start its SOFA runs, record launch failures."""

from __future__ import annotations

import time
from pathlib import Path

import optuna

from sofaopt.core.generation.types import LaunchedTrial, LaunchResult, RunHistory
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.sofa_runner import (
    active_sofa_process_count,
    launch_sofa,
    wait_for_slot,
)
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
    entry: LaunchedTrial,
    *,
    launched: list[LaunchedTrial],
    gen_index: int,
    run_slot: int,
    test_name: str,
    test_run_index: int,
    test_run_total: int,
) -> tuple:
    """Mark a slot launching and start its SOFA process. Returns the run tuple.

    The global ``max_active_sofa_procs`` cap is enforced here, per run — every
    launch path (initial, gated, probe relaunch) goes through this function, so
    no path can exceed the cap regardless of ``n_parallel × run_count``.
    """
    wait_for_slot(
        launched,
        cfg.project.max_active_sofa_procs,
        gen_index,
        entry.trial_index,
        timeout_s=cfg.project.sofa_realtime_timeout,
    )
    scene_file = cfg.project.test(test_name).scene_file
    update_trial_run(
        entry.trial_state_path,
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
        trial_state_path=entry.trial_state_path,
        params_path=entry.params_path,
        run_slot=run_slot,
        gen_index=gen_index,
        trial_index=entry.trial_index,
        run_index=run_slot,
        env=entry.trial_env,
    )
    print(
        f"[sofa] Gen {gen_index:04d} Trial {entry.trial_index:02d} "
        f"Run {run_slot}/{cfg.n_repeats} [{test_name} {test_run_index}/{test_run_total}]"
    )
    entry.launch_times_by_slot[run_slot] = time.time()
    return (proc, entry.trial_state_path, run_slot)


def _relaunch_run(
    cfg: RunConfig,
    entry: LaunchedTrial,
    *,
    launched: list[LaunchedTrial],
    gen_index: int,
    run_slot: int,
    test_name: str,
    test_run_index: int,
    test_run_total: int,
) -> None:
    """Relaunch one run in-place (probe iteration or deferred gated launch)."""
    new_run = _launch_one_run(
        cfg,
        entry,
        launched=launched,
        gen_index=gen_index,
        run_slot=run_slot,
        test_name=test_name,
        test_run_index=test_run_index,
        test_run_total=test_run_total,
    )
    for i, (_p, _path, old_slot) in enumerate(entry.runs):
        if old_slot == run_slot:
            entry.runs[i] = new_run
            break
    else:
        entry.runs.append(new_run)


def _launch_trial_runs(
    cfg: RunConfig, entry: LaunchedTrial, *, launched: list[LaunchedTrial], gen_index: int
) -> None:
    """Launch every ungated run of a trial; queue gated ones as pending."""
    gated = set() if cfg.project.multi_objective else set(cfg.gated_test_names)
    for r, (test_name, test_run_index, test_run_total) in enumerate(cfg.run_plan):
        run_slot = r + 1
        if test_name in gated:
            entry.pending_gated_runs.append(
                (run_slot, test_name, test_run_index, test_run_total)
            )
            update_trial_run(
                entry.trial_state_path,
                run_slot,
                {
                    "state": "pending",
                    "score": None,
                    "reason": "gated_test_waiting_for_ungated_success",
                },
            )
            continue
        entry.runs.append(
            _launch_one_run(
                cfg,
                entry,
                launched=launched,
                gen_index=gen_index,
                run_slot=run_slot,
                test_name=test_name,
                test_run_index=test_run_index,
                test_run_total=test_run_total,
            )
        )


def _record_prepare_failure(
    cfg: RunConfig,
    *,
    study: optuna.Study,
    trial,
    trial_state_path: Path,
    trial_dir: Path,
    gen_index: int,
    trial_index: int,
    error: Exception,
    result: LaunchResult,
    state: RunHistory,
) -> None:
    """Hard-fail one trial whose prepare/launch raised, keeping the study alive.

    Per-trial isolation (§4): the failure is recorded on every run slot, told
    to Optuna as ``hard_fail_score``, and the generation continues.
    """
    project = cfg.project
    hard_fail = project.hard_fail_score
    print(f"[error] Gen {gen_index:04d} Trial {trial_index:02d}: {error}")
    if result.failed_preview is not None:
        render_preview(
            result.failed_preview, trial_dir, gen_index, trial_index,
            project.previews_dir, result.failed_preview,
        )
    for r in range(len(cfg.run_plan)):
        update_trial_run(
            trial_state_path,
            r + 1,
            {"state": "failed", "score": None, "reason": str(error)},
        )
    if project.multi_objective:
        study.tell(trial, [hard_fail] * len(cfg.selected_tests))
    else:
        study.tell(trial, hard_fail)
    update_trial_summary(
        trial_state_path,
        {
            "state": "failed",
            "final_score": hard_fail,
            "outcome": f"prepare failed: {type(error).__name__}",
        },
    )
    result.prelaunch_scores.append(hard_fail)
    state.record_score(hard_fail)


def launch_generation_trials(
    cfg: RunConfig,
    *,
    gen_index: int,
    trials: list,
    study: optuna.Study,
    env: dict,
    state: RunHistory,
    gen_dir: Path,
    trial_state_paths_by_trial: list[Path],
) -> LaunchResult:
    """Prepare and launch every trial of one generation."""
    project = cfg.project
    result = LaunchResult(failed_preview=project.failed_preview_image)

    for i, trial in enumerate(trials):
        trial_index = i + 1
        trial_dir = gen_dir / f"trial_{trial_index:02d}"
        trial_dir.mkdir(exist_ok=True)
        trial_state_path = trial_dir / "trial_state.json"

        if active_sofa_process_count(result.trials) >= project.max_active_sofa_procs:
            _mark_all_runs(trial_state_path, len(cfg.run_plan), "waiting-slot")

        params = params_from_trial(trial, project)

        try:
            _mark_all_runs(trial_state_path, len(cfg.run_plan), "preparing")
            prep = prepare_trial(project, params, trial_dir)
            result.assets_by_trial[trial_index] = list(prep.cleanup)
            if prep.preview_image is not None:
                # Previews render at generation end: offscreen GL contends with
                # SOFA's GL init on Windows and can hang scene startup.
                result.preview_tasks.append((Path(prep.preview_image), trial_index))

            entry = LaunchedTrial(
                trial_index=trial_index,
                trial=trial,
                trial_state_path=trial_state_path,
                params_path=trial_dir / "params.json",
                trial_env={**env, **prep.env},
            )
            # Register the entry *before* launching so this trial's own earlier
            # runs count toward the throttle, and so a mid-trial launch failure
            # can't leave already-started processes untracked.
            result.trials.append(entry)
            try:
                _launch_trial_runs(cfg, entry, launched=result.trials, gen_index=gen_index)
            except Exception:
                # The trial is about to be hard-failed: untrack it and kill any
                # runs that did start, so nothing keeps running unsupervised.
                result.trials.remove(entry)
                for proc, _, _ in entry.runs:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise

        except Exception as e:
            _record_prepare_failure(
                cfg,
                study=study,
                trial=trial,
                trial_state_path=trial_state_path,
                trial_dir=trial_dir,
                gen_index=gen_index,
                trial_index=trial_index,
                error=e,
                result=result,
                state=state,
            )

    return result
