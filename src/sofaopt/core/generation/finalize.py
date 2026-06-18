"""Finalize phase: wait for runs, apply gating/pruning/relaunch, score, summarize."""

from __future__ import annotations

import time
from pathlib import Path

import optuna

from sofaopt.core.algorithm import _finalize_trial_score
from sofaopt.core.generation.launch import _relaunch_run
from sofaopt.core.generation.plan import prune_trial, trial_has_ungated_positive_run
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import write_gen_summary
from sofaopt.core.state import TrialState
from sofaopt.core.trial_state import read_trial_run, read_trial_state, update_trial_run
from sofaopt.core.trialprep import render_preview

_TERMINAL = {"done", "failed", "error", "pruned", "skipped", "cancelled"}


def finalize_generation(
    cfg: RunConfig,
    *,
    gen_index: int,
    study: optuna.Study,
    state: TrialState,
    env: dict,
    gen_dir: Path,
    trial_state_paths_by_trial: list[Path],
    launch_result: dict,
) -> None:
    """Drive launched trials to completion and record their scores."""
    project = cfg.project
    wall_timeout = project.sofa_realtime_timeout
    max_relaunches = project.max_run_relaunches
    relaunchable = {t.name for t in cfg.selected_tests if t.relaunchable}

    processes = launch_result["processes"]
    assets_by_trial = launch_result["assets_by_trial"]
    prelaunch_scores = launch_result["prelaunch_scores"]

    def _finalize(trial_index, trial, runs, trial_state_path):
        return _finalize_trial_score(
            cfg,
            trial_index=trial_index,
            trial=trial,
            runs=runs,
            trial_state_path=trial_state_path,
            study=study,
            gen_index=gen_index,
        )

    try:
        finalized: set[int] = set()
        gen_scores = prelaunch_scores.copy()
        relaunch_counts: dict[tuple[int, int], int] = {}
        bar_width = 28
        start_time = time.time()
        last_print = 0.0

        while len(finalized) < len(processes):
            for (
                trial_index,
                trial,
                runs,
                pending_gated_runs,
                trial_state_path,
                trial_env,
                params_path,
                launch_times_by_slot,
            ) in processes:
                if trial_index in finalized:
                    continue

                trial_has_active_run = False
                now = time.time()
                for proc, _path, run_slot in list(runs):
                    if proc.poll() is None:
                        launch_ts = launch_times_by_slot.get(run_slot)
                        if launch_ts is not None and now - launch_ts > wall_timeout:
                            prune_trial(
                                cfg, gen_index, trial_index, trial_state_path, runs,
                                f"SOFA wall-clock timeout after {wall_timeout:.0f}s",
                            )
                            trial_has_active_run = True
                            break
                        trial_has_active_run = True
                        continue

                    run_data = read_trial_run(trial_state_path, run_slot) or {}
                    run_state = str(run_data.get("state", "")).lower()
                    test_name = str(run_data.get("test_name", ""))

                    # Relaunchable probe: exited non-terminal. If it signalled
                    # probe_finished it wants another iteration; otherwise it
                    # crashed mid-run and is failed (no retry on deterministic crash).
                    if test_name in relaunchable and run_state not in _TERMINAL:
                        key = (trial_index, run_slot)
                        if not run_data.get("probe_finished"):
                            update_trial_run(
                                trial_state_path, run_slot,
                                {
                                    "state": "failed", "score": None,
                                    "reason": "SOFA exited mid-run without completing a probe (crash)",
                                },
                            )
                        elif relaunch_counts.get(key, 0) >= max_relaunches:
                            update_trial_run(
                                trial_state_path, run_slot,
                                {
                                    "state": "failed", "score": None,
                                    "reason": f"exceeded {max_relaunches} probe relaunches",
                                },
                            )
                        else:
                            relaunch_counts[key] = relaunch_counts.get(key, 0) + 1
                            _relaunch_run(
                                cfg, runs=runs,
                                gen_index=gen_index, trial_index=trial_index,
                                run_slot=run_slot, test_name=test_name,
                                test_run_index=int(run_data.get("test_run_index", run_slot)),
                                test_run_total=int(run_data.get("test_run_total", 1)),
                                trial_state_path=trial_state_path,
                                params_path=params_path, trial_env=trial_env,
                                launch_times_by_slot=launch_times_by_slot,
                            )
                            trial_has_active_run = True

                if trial_has_active_run:
                    continue

                trial_state = read_trial_state(trial_state_path)
                if str(trial_state.get("state", "")).lower() == "pruned":
                    finalized.add(trial_index)
                    final_score = _finalize(trial_index, trial, runs, trial_state_path)
                    gen_scores.append(final_score)
                    state.record_score(final_score)
                    continue

                # Ungated runs are all done — open or skip the gated runs.
                if pending_gated_runs:
                    if trial_has_ungated_positive_run(cfg, trial_state_path):
                        print(
                            f"[gate] Gen {gen_index:04d} Trial {trial_index:02d} "
                            f"ungated success; launching gated tests."
                        )
                        for run_slot, t_name, t_idx, t_total in list(pending_gated_runs):
                            _relaunch_run(
                                cfg, runs=runs,
                                gen_index=gen_index, trial_index=trial_index,
                                run_slot=run_slot, test_name=t_name,
                                test_run_index=t_idx, test_run_total=t_total,
                                trial_state_path=trial_state_path,
                                params_path=params_path, trial_env=trial_env,
                                launch_times_by_slot=launch_times_by_slot,
                            )
                        pending_gated_runs.clear()
                        continue
                    for run_slot, *_ in list(pending_gated_runs):
                        update_trial_run(
                            trial_state_path, run_slot,
                            {
                                "state": "skipped", "score": None,
                                "reason": "gated_test_skipped_until_ungated_success",
                            },
                        )
                    pending_gated_runs.clear()

                finalized.add(trial_index)
                final_score = _finalize(trial_index, trial, runs, trial_state_path)
                gen_scores.append(final_score)
                state.record_score(final_score)

            now = time.time()
            if now - last_print >= 0.5:
                total_runs = sum(len(p[2]) for p in processes)
                total_done = sum(
                    sum(1 for pr, _, _ in p[2] if pr.poll() is not None) for p in processes
                )
                pct = (100.0 * total_done / total_runs) if total_runs else 100.0
                filled = int(bar_width * total_done / total_runs) if total_runs else bar_width
                bar = "#" * filled + "-" * (bar_width - filled)
                print(
                    f"\r[progress] Gen {gen_index:04d} SOFA [{bar}] "
                    f"{total_done}/{total_runs} ({pct:5.1f}%)  elapsed {now - start_time:5.1f}s",
                    end="", flush=True,
                )
                last_print = now

            if len(finalized) < len(processes):
                time.sleep(0.2)

        total_runs = sum(len(p[2]) for p in processes)
        print(
            f"\r[progress] Gen {gen_index:04d} SOFA [{'#' * bar_width}] "
            f"{total_runs}/{total_runs} (100.0%)  elapsed {time.time() - start_time:5.1f}s"
        )

        # Render previews now that all SOFA GL contexts for this gen are gone.
        preview_tasks = launch_result.get("preview_tasks", [])
        failed_preview = launch_result.get("failed_preview")
        if preview_tasks:
            print(f"[preview] Gen {gen_index:04d} rendering {len(preview_tasks)} preview(s)")
            for image, trial_index in preview_tasks:
                trial_dir = gen_dir / f"trial_{trial_index:02d}"
                render_preview(
                    Path(image), trial_dir, gen_index, trial_index,
                    project.previews_dir, failed_preview,
                )

        if project.on_generation_end is not None:
            try:
                project.on_generation_end(gen_index, list(trial_state_paths_by_trial))
            except Exception as e:
                print(f"[warn] on_generation_end hook failed: {e}")

        write_gen_summary(gen_dir, gen_index, gen_scores)

    finally:
        for paths in assets_by_trial.values():
            for asset in paths if isinstance(paths, list) else [paths]:
                try:
                    p = Path(asset)
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
