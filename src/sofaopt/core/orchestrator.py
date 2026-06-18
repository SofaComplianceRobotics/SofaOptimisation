"""Top-level optimization entry point: ``run_optimization(project)``."""

from __future__ import annotations

import time

from sofaopt.core.algorithm import build_cmaes_study
from sofaopt.core.generation.runner import run_generation
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import write_progress
from sofaopt.core.state import TrialState
from sofaopt.core.utils import reset_trials_dir
from sofaopt.project import SofaOptProject


def run_optimization(
    project: SofaOptProject, cfg: RunConfig | None = None
) -> None:
    """Run the full CMA-ES optimization for ``project``.

    Args:
        project: The project to optimize.
        cfg: Optional pre-built :class:`RunConfig` (test selection/weights). When
            omitted, selection is read from the environment if present, else all
            of the project's tests are used with their declared weights.
    """
    if cfg is None:
        cfg = RunConfig.from_env(project)

    reset_trials_dir(project.trials_dir, project.previews_dir)

    study = build_cmaes_study(project.db_path, cfg)
    env = cfg.base_scene_env()
    state = TrialState()
    state.load_test_specs(cfg.selected_tests)
    started_at = time.time()

    for gen in range(1, project.n_generations + 1):
        state.advance_gen()
        write_progress(cfg, gen, 0, state.all_scores, started_at)

        print(f"\n{'=' * 50}\nGeneration {gen}/{project.n_generations}\n{'=' * 50}")

        trials = [study.ask() for _ in range(project.n_parallel)]
        run_generation(cfg, gen, trials, study, env, state, started_at)

        try:
            best = study.best_trial
            print(f"[best so far] Trial {best.number} → {best.value:.2f}/100")
        except ValueError:
            print("[best so far] No valid trials yet.")

    print("\nOptimization complete.")
    try:
        best_trial = study.best_trial
        print(f"Best trial:  {best_trial.number}")
        print(f"Best value:  {best_trial.value:.4f}/100")
        print(f"Best params: {best_trial.params}")
    except ValueError:
        print("No valid trials found — all simulations failed.")
