"""Run one full generation: set up trial state, launch, finalize."""

from __future__ import annotations

import threading

import optuna

from sofaopt.core.generation.finalize import finalize_generation
from sofaopt.core.generation.launch import launch_generation_trials
from sofaopt.core.generation.progress import generation_progress_writer
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.state import TrialState
from sofaopt.core.trial_state import init_trial_state


def run_generation(
    cfg: RunConfig,
    gen_index: int,
    trials: list,
    study: optuna.Study,
    env: dict,
    state: TrialState,
    started_at: float = 0.0,
) -> None:
    """Run one CMA-ES generation end to end."""
    project = cfg.project
    gen_dir = project.trials_dir / f"gen_{gen_index:04d}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    weights_pct = {
        name: round(frac * 100, 1) for name, frac in cfg.test_weights.items()
    }
    trial_state_paths_by_trial = []
    for trial_index in range(1, project.n_parallel + 1):
        trial_dir = gen_dir / f"trial_{trial_index:02d}"
        trial_dir.mkdir(exist_ok=True)
        trial_state_path = trial_dir / "trial_state.json"
        init_trial_state(
            trial_state_path,
            gen_index=gen_index,
            trial_index=trial_index,
            run_plan=list(cfg.run_plan),
            test_weights=weights_pct,
            test_max_scores=dict(cfg.test_max_scores),
        )
        trial_state_paths_by_trial.append(trial_state_path)

    progress_stop = threading.Event()
    progress_thread = threading.Thread(
        target=generation_progress_writer,
        args=(cfg, gen_index, trial_state_paths_by_trial, state.all_scores, progress_stop, started_at),
        daemon=True,
    )
    progress_thread.start()

    try:
        launch_result = launch_generation_trials(
            cfg,
            gen_index=gen_index,
            trials=trials,
            study=study,
            env=env,
            state=state,
            gen_dir=gen_dir,
            trial_state_paths_by_trial=trial_state_paths_by_trial,
        )
        finalize_generation(
            cfg,
            gen_index=gen_index,
            study=study,
            state=state,
            env=env,
            gen_dir=gen_dir,
            trial_state_paths_by_trial=trial_state_paths_by_trial,
            launch_result=launch_result,
        )
    finally:
        progress_stop.set()
        progress_thread.join(timeout=2.0)
