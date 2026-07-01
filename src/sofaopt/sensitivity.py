"""One-at-a-time (OAT) sensitivity analysis for sofaopt projects.

Sweeps each non-frozen parameter from its low to high bound while holding
all others at their default values, runs the SOFA scene for each sample, and
measures how much the score changes.  The result is a ranked dict of signed
normalized sensitivities — positive means the score rises as the parameter
increases, negative means it falls.

Typical use (before a full CMA-ES run)::

    from sofaopt.sensitivity import run_sensitivity_analysis
    scores = run_sensitivity_analysis(PROJECT, n_samples=5)
    # {'spring_k': +82.3, 'damping': -14.1, 'mass': +3.6}

The most influential parameters are listed first (by absolute magnitude).
Use the result to narrow search ranges, freeze irrelevant parameters, or
choose a ``sampler`` better suited to the active dimension count.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from sofaopt.core.runconfig import RunConfig
from sofaopt.core.sofa_runner import launch_sofa
from sofaopt.core.trial_state import init_trial_state, read_trial_run
from sofaopt.core.trialprep import prepare_trial
from sofaopt.project import ParamSpec, SofaOptProject


def run_sensitivity_analysis(
    project: SofaOptProject,
    n_samples: int = 5,
    param_names: Sequence[str] | None = None,
    test_name: str | None = None,
    cfg: RunConfig | None = None,
    cleanup: bool = True,
) -> dict[str, float]:
    """Run OAT sensitivity analysis, returning signed normalized scores per param.

    For each non-frozen parameter, sweeps ``n_samples`` evenly-spaced values
    from ``low`` to ``high`` while holding all other parameters at their
    defaults.  One SOFA subprocess is launched per sample (sequentially).

    Args:
        project: The project to analyse.
        n_samples: Samples per parameter (minimum 2).
        param_names: Which params to sweep (default: all non-frozen).
        test_name: Test to score against (default: first test).
        cfg: Optional pre-built :class:`~sofaopt.core.runconfig.RunConfig`.
        cleanup: Remove the temporary sensitivity run directory when done.

    Returns:
        ``{param_name: sensitivity_pct}`` sorted by ``abs(sensitivity_pct)``
        descending.  Sensitivity is signed: positive = score rises as the
        parameter increases from ``low`` to ``high``, negative = falls.
        The magnitude is normalized to 100 % of the global score range across
        all samples.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    if cfg is None:
        _test = test_name or project.tests[0].name
        cfg = RunConfig.from_project(project, selected_names=[_test])

    selected_test = cfg.selected_tests[0]
    scene_file = selected_test.scene_file

    non_frozen: list[ParamSpec] = [p for p in project.params if not p.is_frozen]
    if param_names is not None:
        target: list[ParamSpec] = [p for p in non_frozen if p.name in set(param_names)]
    else:
        target = non_frozen

    if not target:
        print("[sensitivity] No non-frozen params to analyse.")
        return {}

    defaults: dict[str, Any] = {p.name: p.default for p in project.params}

    timestamp = int(time.time())
    sens_dir = project.runtime_dir / "sensitivity" / f"run_{timestamp}"
    sens_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(
            f"[sensitivity] Analysing {len(target)} param(s), "
            f"{n_samples} samples each, test={selected_test.name!r}"
        )

        env = cfg.base_scene_env()
        all_scores: list[float] = []
        param_scores: dict[str, list[float]] = {}

        run_num = 0
        for param in target:
            samples = _build_samples(param, n_samples)
            scores_for_param: list[float] = []

            for sample_val in samples:
                run_num += 1
                run_dir = sens_dir / f"run_{run_num:04d}"
                run_dir.mkdir()

                params = {**defaults, param.name: sample_val}
                trial_state_path = run_dir / "trial_state.json"

                init_trial_state(
                    trial_state_path,
                    gen_index=0,
                    trial_index=run_num,
                    run_plan=[(selected_test.name, 1, 1)],
                    params=params,
                    test_weights={selected_test.name: 1.0},
                    test_max_scores={selected_test.name: selected_test.max_score},
                )

                try:
                    prep = prepare_trial(project, params, run_dir)
                    run_env = {**env, **prep.env}
                except Exception as exc:
                    print(
                        f"[sensitivity] prepare failed for "
                        f"{param.name}={sample_val}: {exc}"
                    )
                    scores_for_param.append(float("-inf"))
                    continue

                params_path = run_dir / "params.json"
                proc = launch_sofa(
                    project,
                    scene_file=scene_file,
                    test_name=selected_test.name,
                    test_run_index=1,
                    test_run_total=1,
                    trial_state_path=trial_state_path,
                    params_path=params_path,
                    run_slot=1,
                    gen_index=0,
                    trial_index=run_num,
                    run_index=1,
                    env=run_env,
                )

                _wait(proc, project.sofa_realtime_timeout)

                for asset in prep.cleanup:
                    try:
                        Path(asset).unlink(missing_ok=True)
                    except Exception:
                        pass

                run_data = read_trial_run(trial_state_path, 1) or {}
                raw = run_data.get("score")
                score = float(raw) if isinstance(raw, (int, float)) else float("-inf")
                scores_for_param.append(score)
                if score != float("-inf"):
                    all_scores.append(score)

                status = f"{score:.3f}" if score != float("-inf") else "FAILED"
                print(f"[sensitivity]   {param.name}={sample_val!r} -> {status}")

            param_scores[param.name] = scores_for_param

        return _compute_sensitivities(param_scores, all_scores)

    finally:
        if cleanup:
            shutil.rmtree(sens_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_samples(param: ParamSpec, n_samples: int) -> list[Any]:
    if param.type == "bool":
        return [False, True]
    lo, hi = float(param.low), float(param.high)
    step = (hi - lo) / (n_samples - 1)
    if param.type == "int":
        return sorted({int(round(lo + i * step)) for i in range(n_samples)})
    return [lo + i * step for i in range(n_samples)]


def _wait(proc, timeout: float) -> None:
    start = time.time()
    while proc.poll() is None:
        if time.time() - start > timeout:
            proc.kill()
            break
        time.sleep(0.2)


def _compute_sensitivities(
    param_scores: dict[str, list[float]],
    all_scores: list[float],
) -> dict[str, float]:
    valid_all = [s for s in all_scores if s != float("-inf")]
    global_range = (
        (max(valid_all) - min(valid_all)) if len(valid_all) >= 2 else 1.0
    ) or 1.0

    raw: dict[str, float] = {}
    for name, scores in param_scores.items():
        valid = [s for s in scores if s != float("-inf")]
        if len(valid) < 2:
            raw[name] = 0.0
        else:
            raw[name] = round(100.0 * (valid[-1] - valid[0]) / global_range, 2)

    sorted_result = dict(
        sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )

    print("\n[sensitivity] Results (param -> effect % of global score range):")
    print(f"  {'Parameter':<28} {'Sensitivity':>12}  Direction")
    print(f"  {'-' * 28} {'-' * 12}  {'-' * 12}")
    for name, pct in sorted_result.items():
        direction = "score↑" if pct > 0 else ("score↓" if pct < 0 else "no effect")
        print(f"  {name:<28} {pct:>+10.2f}%  {direction}")
    print()

    return sorted_result
