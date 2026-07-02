"""Top-level optimization entry point: ``run_optimization(project)``."""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

from sofaopt.core import envkeys
from sofaopt.core.algorithm import build_study
from sofaopt.core.generation.runner import run_generation
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import write_progress
from sofaopt.core.state import TrialState
from sofaopt.core.utils import reset_trials_dir
from sofaopt.project import SofaOptProject


def _apply_env_overrides(project: SofaOptProject) -> SofaOptProject:
    """Apply optimizer-setting overrides from the environment (set by the
    dashboard's Run button) over the project's own fields.

    Lets a launched headless run honor UI choices for sampler / margin /
    seed-design / parallelism without editing the project. No-op when none of
    the ``OPT_*`` override keys are present, so CLI/script usage is unaffected.
    """
    overrides: dict[str, object] = {}
    sampler = os.environ.get(envkeys.SAMPLER)
    if sampler in ("cmaes", "gp", "tpe", "random"):
        overrides["sampler"] = sampler
    seed = os.environ.get(envkeys.SEED_SAMPLER)
    if seed in ("random", "sobol"):
        overrides["seed_sampler"] = seed
    margin = os.environ.get(envkeys.CMAES_MARGIN)
    if margin is not None:
        overrides["cmaes_with_margin"] = margin.strip().lower() in ("1", "true", "yes", "on")
    for key, field in ((envkeys.N_PARALLEL, "n_parallel"), (envkeys.N_GENERATIONS, "n_generations")):
        raw = os.environ.get(key)
        if raw:
            try:
                overrides[field] = int(raw)
            except ValueError:
                print(f"[override] Ignoring non-int {key}={raw!r}")
    if not overrides:
        return project
    print(f"[override] Applying env optimizer overrides: {overrides}")
    return dataclasses.replace(project, **overrides)


def _last_gen_index(trials_dir: Path) -> int:
    """Highest generation number present on disk (0 when none)."""
    last = 0
    for d in trials_dir.glob("gen_*"):
        try:
            last = max(last, int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return last


def _post_run_video(project: SofaOptProject) -> None:
    """Auto-cleanup trial recordings and generate a summary video after the run."""
    try:
        from sofaopt.video import cleanup_trial_recordings, generate_summary_video
        print("\n[video] Pruning trial recordings ...")
        cleanup_trial_recordings(
            project,
            keep_top_n=project.record_keep_top_n,
            keep_bottom_n=project.record_keep_bottom_n,
        )
        summary_path = project.runtime_dir / "summary.mp4"
        print(f"[video] Generating summary video → {summary_path}")
        generate_summary_video(
            project,
            summary_path,
            top_n=project.record_summary_top_n,
            bottom_n=project.record_summary_bottom_n,
        )
    except Exception as exc:
        print(f"[video] Post-run video step failed: {exc}")


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
    # Apply dashboard/env optimizer-setting overrides before anything reads the
    # project's sampler fields. Skip when the caller supplied a pre-built cfg
    # (it already pins project/selection and overriding would desync cfg.project).
    if cfg is None:
        project = _apply_env_overrides(project)
        cfg = RunConfig.from_env(project)

    # Windows consoles default to cp1252; make sure framework logging (and any
    # non-ASCII in scene output) never crashes the run on an encode error.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    resuming = project.db_path.exists()
    if resuming:
        project.trials_dir.mkdir(parents=True, exist_ok=True)
        project.previews_dir.mkdir(parents=True, exist_ok=True)
        print(f"[resume] Found existing study at {project.db_path} — continuing without reset.")
    else:
        reset_trials_dir(project.trials_dir, project.previews_dir)

    study = build_study(project.db_path, cfg, resume=resuming)

    # When resuming, continue generation numbering from the dirs actually on
    # disk. (Counting COMPLETE Optuna trials under-counts when a previous run
    # was killed mid-generation or trials were pruned, which made a resumed run
    # reuse — and overwrite — existing gen_XXXX directories.)
    gen_offset = _last_gen_index(project.trials_dir) if resuming else 0

    env = cfg.base_scene_env()
    state = TrialState()
    state.load_test_specs(cfg.selected_tests)
    started_at = time.time()
    _prune_count = 0  # tracks how many periodic prunes have fired (trial-count based)

    total_gens = gen_offset + project.n_generations
    for gen in range(gen_offset + 1, total_gens + 1):
        state.advance_gen()
        write_progress(cfg, gen, 0, state.all_scores, started_at, total_gens=total_gens)

        print(f"\n{'=' * 50}\nGeneration {gen}/{total_gens}\n{'=' * 50}")

        trials = [study.ask() for _ in range(project.n_parallel)]
        run_generation(cfg, gen, trials, study, env, state, started_at, total_gens=total_gens)

        if project.record_frames:
            try:
                from sofaopt.video import apply_generation_overlays
                apply_generation_overlays(project, gen)
            except Exception as exc:
                print(f"[video] Gen {gen}: overlay pass failed: {exc}")

        if project.multi_objective:
            try:
                pareto = study.best_trials
                print(f"[best so far] {len(pareto)} Pareto-optimal trial(s)")
            except Exception:
                print("[best so far] No valid trials yet.")
        else:
            try:
                best = study.best_trial
                print(f"[best so far] Trial {best.number} -> {best.value:.2f}/100")
            except ValueError:
                print("[best so far] No valid trials yet.")

        if project.record_frames and project.record_prune_every_n > 0:
            _new_prune = (gen * project.n_parallel) // project.record_prune_every_n
            if _new_prune > _prune_count:
                _prune_count = _new_prune
                try:
                    from sofaopt.video import cleanup_trial_recordings
                    total = gen * project.n_parallel
                    print(f"[video] {total} trials completed: periodic prune ...")
                    cleanup_trial_recordings(
                        project,
                        keep_top_n=project.record_keep_top_n,
                        keep_bottom_n=project.record_keep_bottom_n,
                    )
                except Exception as exc:
                    print(f"[video] Periodic prune failed: {exc}")

    print("\nOptimization complete.")
    if project.record_frames:
        _post_run_video(project)
    if project.multi_objective:
        try:
            pareto = study.best_trials
            print(f"Pareto front: {len(pareto)} trial(s)")
            for t in pareto[:5]:
                print(f"  Trial {t.number}: values={[round(v, 4) for v in t.values]}")
        except Exception:
            print("No valid trials found - all simulations failed.")
    else:
        try:
            best_trial = study.best_trial
            print(f"Best trial:  {best_trial.number}")
            print(f"Best value:  {best_trial.value:.4f}/100")
            print(f"Best params: {best_trial.params}")
        except ValueError:
            print("No valid trials found - all simulations failed.")
