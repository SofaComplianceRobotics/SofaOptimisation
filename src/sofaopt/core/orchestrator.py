"""Top-level optimization entry point: ``run_optimization(project)``."""

from __future__ import annotations

import contextlib
import logging
import dataclasses
import os
import sys
import time

from pathlib import Path

import optuna

from sofaopt.core import envkeys
from sofaopt.core.algorithm import _seed_sampler, build_study, recover_interrupted_trials
from sofaopt.core.restart import maybe_restart, restart_index, restart_popsize
from sofaopt.core.runlock import acquire_run_lock, release_run_lock
from sofaopt.core.generation.runner import run_generation
from sofaopt.core.generation.types import RunHistory
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.runtime_dirs import (
    configure_console_logging,
    last_gen_index,
    reset_trials_dir,
)
from sofaopt.core.scoring import write_progress
from sofaopt.core.trial_state import (
    read_trial_state,
    update_trial_run,
    update_trial_summary,
)
from sofaopt.project import SofaOptProject

logger = logging.getLogger(__name__)

_TERMINAL_STATES = ("done", "failed", "pruned", "interrupted", "cached")


def _mark_interrupted_on_disk(trials_dir: Path) -> None:
    """Close out trial states a killed run left non-terminal (the dashboard
    otherwise shows them as running forever and ranks their absent scores).

    Scans EVERY generation dir, not just the last: earlier pauses (or resumes
    that predate this marking) can leave stale states in older generations.
    ONLY call while no run is live — a live generation's in-flight trials are
    indistinguishable from stale ones (they would be mislabeled until their
    own writers overwrite the state).
    """
    for tdir in sorted(Path(trials_dir).glob("gen_*/trial_*")):
        path = tdir / "trial_state.json"
        state = read_trial_state(path)
        if not isinstance(state, dict):
            continue
        runs = state.get("runs") or []
        open_slots = [
            i + 1
            for i, r in enumerate(runs)
            if isinstance(r, dict) and str(r.get("state", "")) not in _TERMINAL_STATES
        ]
        if str(state.get("state", "")) in _TERMINAL_STATES and not open_slots:
            continue
        for slot in open_slots:
            update_trial_run(
                path, slot,
                {"state": "interrupted", "reason": "interrupted (run paused/killed)"},
            )
        if str(state.get("state", "")) not in _TERMINAL_STATES:
            update_trial_summary(
                path,
                {"state": "interrupted", "final_score": None,
                 "outcome": "interrupted — params re-enqueued on resume"},
            )


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
    converged = os.environ.get(envkeys.RUN_UNTIL_CONVERGED)
    if converged is not None:
        overrides["run_until_converged"] = converged.strip().lower() in ("1", "true", "yes", "on")
    for key, field in (
        (envkeys.N_PARALLEL, "n_parallel"),
        (envkeys.N_GENERATIONS, "n_generations"),
        (envkeys.RESTART_PATIENCE, "restart_patience"),
    ):
        raw = os.environ.get(key)
        if raw:
            try:
                overrides[field] = int(raw)
            except ValueError:
                logger.info(f"[override] Ignoring non-int {key}={raw!r}")
    if not overrides:
        return project
    logger.info(f"[override] Applying env optimizer overrides: {overrides}")
    return dataclasses.replace(project, **overrides)


def _post_run_video(project: SofaOptProject) -> None:
    """Auto-cleanup trial recordings and generate a summary video after the run."""
    try:
        from sofaopt.video import cleanup_trial_recordings, generate_summary_video
        logger.info("\n[video] Pruning trial recordings ...")
        cleanup_trial_recordings(
            project,
            keep_top_n=project.record_keep_top_n,
            keep_bottom_n=project.record_keep_bottom_n,
        )
        summary_path = project.runtime_dir / "summary.mp4"
        logger.info(f"[video] Generating summary video -> {summary_path}")
        generate_summary_video(
            project,
            summary_path,
            top_n=project.record_summary_top_n,
            bottom_n=project.record_summary_bottom_n,
        )
    except Exception as exc:
        logger.info(f"[video] Post-run video step failed: {exc}")


def _maybe_prune_recordings(project: SofaOptProject, gen: int, prune_count: int) -> int:
    """Periodic recording prune, fired each time the completed-trial count
    crosses a multiple of ``record_prune_every_n``. Returns the updated count."""
    if not project.record_frames or project.record_prune_every_n <= 0:
        return prune_count
    new_count = (gen * project.n_parallel) // project.record_prune_every_n
    if new_count <= prune_count:
        return prune_count
    try:
        from sofaopt.video import cleanup_trial_recordings
        logger.info(f"[video] {gen * project.n_parallel} trials completed: periodic prune ...")
        cleanup_trial_recordings(
            project,
            keep_top_n=project.record_keep_top_n,
            keep_bottom_n=project.record_keep_bottom_n,
        )
    except Exception as exc:
        logger.info(f"[video] Periodic prune failed: {exc}")
    return new_count


def _print_best_so_far(study, project: SofaOptProject) -> None:
    if project.multi_objective:
        try:
            logger.info(f"[best so far] {len(study.best_trials)} Pareto-optimal trial(s)")
        except Exception:
            logger.info("[best so far] No valid trials yet.")
        return
    try:
        best = study.best_trial
        logger.info(f"[best so far] Trial {best.number} -> {best.value:.2f}/100")
    except ValueError:
        logger.info("[best so far] No valid trials yet.")


def _report_results(study, project: SofaOptProject) -> None:
    """Final console report once the run completes."""
    if project.multi_objective:
        try:
            pareto = study.best_trials
            logger.info(f"Pareto front: {len(pareto)} trial(s)")
            for t in pareto[:5]:
                logger.info(f"  Trial {t.number}: values={[round(v, 4) for v in t.values]}")
        except Exception:
            logger.info("No valid trials found - all simulations failed.")
        return
    try:
        best_trial = study.best_trial
        logger.info(f"Best trial:  {best_trial.number}")
        logger.info(f"Best value:  {best_trial.value:.4f}/100")
        logger.info(f"Best params: {best_trial.params}")
    except ValueError:
        logger.info("No valid trials found - all simulations failed.")


def run_optimization(
    project: SofaOptProject, cfg: RunConfig | None = None
) -> None:
    """Run the full optimization for ``project``.

    Args:
        project: The project to optimize.
        cfg: Optional pre-built :class:`RunConfig` (test selection/weights). When
            omitted, selection is read from the environment if present, else all
            of the project's tests are used with their declared weights.
    """
    # Apply dashboard/env optimizer-setting overrides before anything reads the
    # project's sampler fields. Skip when the caller supplied a pre-built cfg
    # (it already pins project/selection and overriding would desync cfg.project).
    configure_console_logging()
    if cfg is None:
        project = _apply_env_overrides(project)
        cfg = RunConfig.from_env(project)

    # At most one optimizer per study: a second process resuming the same
    # study.db fails the first one's in-flight trials (see core/runlock.py).
    lock = acquire_run_lock(project.runtime_dir)
    if lock is None:
        return
    try:
        _run(project, cfg)
    finally:
        release_run_lock(lock)


def _prepare_runtime_dirs(project: SofaOptProject, resuming: bool) -> None:
    if resuming:
        project.trials_dir.mkdir(parents=True, exist_ok=True)
        project.previews_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[resume] Found existing study at {project.db_path} — continuing without reset.")
    else:
        # Starting fresh must never destroy a previous run: any existing run
        # data is moved into work_dir/archives/ first (instant — a rename).
        from sofaopt.core.archive import archive_run, runtime_has_run_data

        if runtime_has_run_data(project):
            archived = archive_run(project, name="auto")
            logger.info(f"[archive] Previous run auto-archived to {archived.name}")
        reset_trials_dir(project.trials_dir, project.previews_dir)


def _recover_resumed_study(study, project: SofaOptProject, gen_offset: int) -> None:
    # A pause/kill mid-generation leaves asked-but-unscored trials behind:
    # re-enqueue their params (they run first in the next generation) and
    # close the stale records both in Optuna and on disk.
    recovered = recover_interrupted_trials(study)
    _mark_interrupted_on_disk(project.trials_dir)
    if recovered:
        logger.info(
            f"[resume] Re-enqueued {recovered} interrupted trial(s) — "
            f"they run first in generation {gen_offset + 1}."
        )


def _apply_overlays_safely(project: SofaOptProject, gen: int) -> None:
    """Text overlays are a nicety — a failed pass must never kill the run."""
    try:
        from sofaopt.video import apply_generation_overlays
        apply_generation_overlays(project, gen)
    except Exception as exc:
        logger.info(f"[video] Gen {gen}: overlay pass failed: {exc}")


class _StallTracker:
    """Stagnation signal: no best-score improvement for ``limit`` generations
    in a row. The orchestrator then stops the run — or, when the project has
    ``cmaes_restarts`` left, performs an IPOP restart and resets the patience.

    A generation without a valid best (all trials failed) counts as stalled.
    Disabled when ``limit`` is 0 (also used for multi-objective runs, where a
    single scalar "best" does not exist).
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.best: float | None = None
        self.count = 0

    def should_stop(self, study) -> bool:
        if self.limit <= 0:
            return False
        try:
            current: float | None = float(study.best_value)
        except ValueError:
            current = None
        if current is not None and (self.best is None or current > self.best + 1e-9):
            self.best, self.count = current, 0
            return False
        self.count += 1
        return self.count >= self.limit

    def reset(self) -> None:
        """A restart is a fresh attempt: give it the full patience again (it
        still has to beat the run-global best to be counted as improving)."""
        self.count = 0


def _best_value(study) -> float | None:
    """The study's best completed value, or None before any trial completes."""
    try:
        return float(study.best_value)
    except ValueError:
        return None


def _use_convergence_trigger(project) -> bool:
    """Convergence-triggered restarts apply only to single-objective CMA-ES."""
    return (
        project.restart_on_convergence
        and project.sampler == "cmaes"
        and not project.multi_objective
    )


def _cma_converged(study) -> bool:
    """True once the CMA-ES sampler's internal optimizer has actually converged
    (``should_stop``) — the honest restart trigger, versus the best-plateau
    heuristic that fires while the search is still productive.

    Reached through the sampler's private ``_restore_optimizer`` seam (the same
    interface the restart scoping relies on). Returns False when no optimizer is
    restorable yet — early generations, or just after a restart — which also
    prevents a restart storm (the fresh restart reads 'not converged')."""
    restore = getattr(study.sampler, "_restore_optimizer", None)
    if restore is None:
        return False
    try:
        completed = study.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        optimizer = restore(completed)
        return bool(optimizer is not None and optimizer.should_stop())
    except Exception:
        return False


class _RestartProductivity:
    """Tracks whether restarts keep paying off (``run_until_converged``).

    A *basin* (the initial run, or each restart's exploration) is productive
    if the run-global best improved during it. Once ``patience`` consecutive
    restarts fail to improve, the search has converged — more restarts are not
    finding new basins worth the budget. The run-global best is never reset, so
    each basin must beat everything seen before it to count as productive.
    """

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_before_basin: float = float("-inf")
        self.streak = 0

    def basin_ended(self, current_best: float | None) -> bool:
        """Call when a stall fires. Updates the streak; returns True when the
        run has converged (``patience`` consecutive fruitless restarts)."""
        improved = current_best is not None and current_best > self.best_before_basin + 1e-9
        self.streak = 0 if improved else self.streak + 1
        return self.streak >= self.patience

    def restarted(self, current_best: float | None) -> None:
        """Call after a restart fires: the new basin is measured from here."""
        if current_best is not None:
            self.best_before_basin = current_best


def _compute_restart_state(study, project, stall: "_StallTracker", fruitless_streak: int) -> dict:
    """Per-generation IPOP snapshot for progress.json (dashboard restart panel).

    A plain read of the study/stall state — no side effects. The population
    grows only after a real restart, so ``current_popsize`` is ``n_parallel``
    until then.
    """
    index = restart_index(study) if project.sampler == "cmaes" else 0
    return {
        "restart_index": index,
        "restarts_max": project.cmaes_restarts,
        "stall_count": stall.count,
        "stall_limit": stall.limit,
        "current_popsize": restart_popsize(project, index) if index > 0 else project.n_parallel,
        "fruitless_streak": fruitless_streak,
        "restart_patience": project.restart_patience,
        "run_until_converged": project.run_until_converged,
        # Whether stall_count is the live restart trigger or just along for the
        # display ride (see _run's comment above `stalled = stall.should_stop`):
        # with a convergence trigger active, stall_count can run past
        # stall_limit indefinitely without ever firing a restart.
        "convergence_trigger": _use_convergence_trigger(project),
    }


def _handle_stall(
    study, project, stall: "_StallTracker", productivity: "_RestartProductivity",
    *, gen: int, history, total_gens: int,
) -> bool:
    """Decide what a stall means: converge-stop, IPOP restart, or plain stop.

    Returns True when the run should end (break the generation loop), False
    when a restart fired and the loop should continue with the grown
    population.
    """
    converged = productivity.basin_ended(_best_value(study))
    if project.run_until_converged and converged:
        logger.info(
            f"[converged] Search converged — {productivity.streak} restart(s) "
            f"without improvement at generation {gen}. Stopping."
        )
        return True
    event = maybe_restart(
        study, project, _seed_sampler(project),
        gen=gen, trial_chron=len(history.all_scores),
    )
    if event:
        productivity.restarted(_best_value(study))
        stall.reset()
        return False
    logger.info(
        f"[stop] No further restart available — stopping early at "
        f"generation {gen}/{total_gens}."
    )
    return True


def _run(project: SofaOptProject, cfg: RunConfig) -> None:
    # Windows consoles default to cp1252; make sure framework logging (and any
    # non-ASCII in scene output) never crashes the run on an encode error.
    for _stream in (sys.stdout, sys.stderr):
        # Non-reconfigurable stream (e.g. pytest capture) — keep the default.
        with contextlib.suppress(Exception):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    resuming = project.db_path.exists()
    _prepare_runtime_dirs(project, resuming)
    study = build_study(project.db_path, cfg, resume=resuming)

    # When resuming, continue generation numbering from the dirs actually on
    # disk (see runtime_dirs.last_gen_index for why not the Optuna trial count).
    gen_offset = last_gen_index(project.trials_dir) if resuming else 0
    if resuming:
        _recover_resumed_study(study, project, gen_offset)

    env = cfg.base_scene_env()
    history = RunHistory()
    started_at = time.time()
    prune_count = 0  # how many periodic recording prunes have fired

    total_gens = gen_offset + project.n_generations
    stall = _StallTracker(0 if project.multi_objective else project.stall_generations)
    productivity = _RestartProductivity(project.restart_patience)
    for gen in range(gen_offset + 1, total_gens + 1):
        restart_state = _compute_restart_state(study, project, stall, productivity.streak)
        write_progress(
            cfg, gen, 0, history.all_scores, started_at,
            total_gens=total_gens, restart_state=restart_state,
        )
        logger.info(f"\n{'=' * 50}\nGeneration {gen}/{total_gens}\n{'=' * 50}")

        trials = [study.ask() for _ in range(project.n_parallel)]
        run_generation(
            cfg, gen, trials, study, env, history, started_at,
            total_gens=total_gens, restart_state=restart_state,
        )

        if project.record_frames:
            _apply_overlays_safely(project, gen)

        _print_best_so_far(study, project)
        prune_count = _maybe_prune_recordings(project, gen, prune_count)

        # Keep the stall tracker updated every generation (its count/best drive
        # the dashboard patience display), but use CMA-ES's real convergence as
        # the restart trigger when configured — the plateau heuristic fires
        # while the search is still productive and makes restarts net-harmful.
        stalled = stall.should_stop(study)
        triggered = _cma_converged(study) if _use_convergence_trigger(project) else stalled
        if triggered and _handle_stall(
            study, project, stall, productivity, gen=gen, history=history,
            total_gens=total_gens,
        ):
            break

    logger.info("\nOptimization complete.")
    if project.record_frames:
        _post_run_video(project)
    _report_results(study, project)
