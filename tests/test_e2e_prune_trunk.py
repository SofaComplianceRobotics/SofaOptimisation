"""Closed-loop equal-budget e2e for multi-fidelity step pruning (design §8 Gate 2).

Runs TWO full trunk_control campaigns through the real orchestrator at the
same trial budget — one in ``prune_mode="shadow"``, one in ``"kill"`` — with a
seeded Sobol' startup so the first two generations ask IDENTICAL parameter
vectors in both arms. Because the trunk scene is bit-deterministic per params,
the shadow arm then provides the ground-truth final score for every trial the
kill arm terminated early, which turns the design's open-loop replay claims
into closed-loop assertions:

- shadow arm: full decision path runs, marks would-kills, kills nothing;
- kill arm: rung-pruned trials exist, are PRUNED in the study, and carry
  their rung partial as an Optuna intermediate value (the
  ``consider_pruned_trials`` feed);
- across the matched startup generations, no generation winner was killed,
  the kill arm's best equals the shadow arm's (parity at equal budget), and
  the kill arm spent measurably fewer simulation steps.

This is deliberately the heaviest e2e (~2 x 32 trials x ~3 s): merge-tier /
after touching the pruning or finalize machinery, not the default tier.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import optuna
import pytest

pytest.importorskip("sofascene")  # provides the `sofa` marker + auto-skip

from sofaopt import run_optimization

pytestmark = pytest.mark.sofa

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
EXAMPLE_DIR = HERE.parent / "examples" / "trunk_control"

_SOFA_SITE = Path(os.environ.get("SOFA_ROOT", "")) / "lib" / "python3" / "site-packages"
if not (_SOFA_SITE / "softrobots").is_dir():
    pytest.skip("SoftRobots python packages not found under SOFA_ROOT", allow_module_level=True)

N_PARALLEL = 8
GENS = 4          # gens 1-2: seeded Sobol' startup (matched across arms); 3-4: CMA-ES
STARTUP = 16      # = 2 generations of 8
STARTUP_GENS = (1, 2)


def _load_project():
    spec = importlib.util.spec_from_file_location(
        "trunk_project_prune_e2e", EXAMPLE_DIR / "project.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROJECT


def _campaign(work_dir: Path, mode: str) -> dict:
    """Run one arm; return {(gen, trial): info} plus the study handle."""
    base = _load_project()
    pythonpath = os.pathsep.join(
        [str(SRC), str(_SOFA_SITE)]
        + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    )
    project = dataclasses.replace(
        base,
        work_dir=work_dir,
        sofa_env={**{k: str(v) for k, v in base.sofa_env.items()}, "PYTHONPATH": pythonpath},
        n_parallel=N_PARALLEL,
        n_generations=GENS,
        cmaes_startup_trials=STARTUP,
        cmaes_restarts=0, restart_on_convergence=False, warm_restarts=False,
        stall_generations=0,
        prune_mode=mode,
        run_script=None,
    )
    run_optimization(project)

    trials: dict[tuple[int, int], dict] = {}
    for state_path in sorted(project.trials_dir.glob("gen_*/trial_*/trial_state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        gen = int(state_path.parent.parent.name.split("_")[1])
        idx = int(state_path.parent.name.split("_")[1])
        run = (state.get("runs") or [{}])[0]
        params_file = state_path.parent / "params.json"
        trials[(gen, idx)] = {
            "state": state.get("state"),
            "final": state.get("final_score"),
            "run_state": run.get("state"),
            "frame": _steps_spent(state, run),
            "shadowed": "shadow_prune_step" in run,
            "params": json.loads(params_file.read_text(encoding="utf-8"))
            if params_file.exists() else None,
        }
    study = optuna.load_study(
        study_name=project.name, storage=f"sqlite:///{project.db_path}"
    )
    return {"trials": trials, "study": study}


def _steps_spent(state: dict, run: dict) -> int:
    """Simulation steps a run consumed. ``prune_trial`` zeroes the killed
    slot's ``current_frame`` (dashboard semantics), so a rung-pruned trial's
    spend is the rung step, recovered from the prune reason."""
    if state.get("state") == "pruned":
        match = re.search(r"rung (\d+)", str(state.get("outcome", "")))
        return int(match.group(1)) if match else 0
    return int(run.get("current_frame") or 0)


def _startup_finals(trials: dict, gen: int) -> dict[int, float]:
    return {
        i: t["final"]
        for (g, i), t in trials.items()
        if g == gen and isinstance(t["final"], (int, float))
    }


def _assert_shadow_arm_harmless(s_trials: dict) -> None:
    """Shadow: decisions logged, nothing killed, campaign completed."""
    assert all(t["state"] != "pruned" for t in s_trials.values())
    assert sum(t["state"] == "done" for t in s_trials.values()) >= GENS * N_PARALLEL - 2
    assert any(t["shadowed"] for t in s_trials.values()), "shadow arm marked no would-kills"


def _assert_kill_arm_feeds_optuna(kill: dict, killed: set) -> None:
    """Every rung-pruned trial is PRUNED in the study with an intermediate value."""
    assert killed, "kill arm pruned nothing — rungs never fired"
    pruned_in_study = [
        t for t in kill["study"].trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    assert len(pruned_in_study) == len(killed)
    for t in pruned_in_study:
        assert t.intermediate_values, (
            f"PRUNED trial {t.number} carries no intermediate value — the "
            "consider_pruned_trials feed is broken"
        )


def _assert_matched_startup_asks(s_trials: dict, k_trials: dict) -> None:
    """The seeded Sobol' startup must ask identical params in both arms."""
    for gen in STARTUP_GENS:
        for i in range(1, N_PARALLEL + 1):
            sp, kp = s_trials[(gen, i)]["params"], k_trials[(gen, i)]["params"]
            assert sp is not None and kp is not None
            assert sp == pytest.approx(kp), f"gen {gen} trial {i}: arms asked different params"


def _assert_do_no_harm(gen: int, s_trials: dict, k_trials: dict, killed: set) -> None:
    """Shadow-arm finals are ground truth: the kill arm never killed a winner."""
    finals = _startup_finals(s_trials, gen)
    winner = max(finals, key=finals.get)
    killed_here = [i for (g, i) in killed if g == gen]
    assert winner not in killed_here, (
        f"gen {gen}: pruning killed the generation winner "
        f"(trial {winner}, final {finals[winner]:.2f}) — do-no-harm violated"
    )
    if killed_here:
        k_finals = _startup_finals(k_trials, gen)
        assert max(k_finals.values()) == pytest.approx(finals[winner], abs=1.0), (
            f"gen {gen}: best-score parity broken at equal budget"
        )


def _assert_step_savings(s_trials: dict, k_trials: dict, killed: set) -> None:
    s_steps = sum(t["frame"] for (g, _), t in s_trials.items() if g in STARTUP_GENS)
    k_steps = sum(t["frame"] for (g, _), t in k_trials.items() if g in STARTUP_GENS)
    saved = 1.0 - k_steps / s_steps
    print(f"\n[e2e] prune closed-loop: killed={len(killed)} trials, "
          f"startup steps {s_steps} -> {k_steps} ({saved:.0%} saved)")
    assert saved > 0.04, (
        f"kill arm saved only {saved:.1%} of startup-generation steps — "
        "pruning fired but bought nothing"
    )
    assert saved < 0.70, f"implausible savings {saved:.1%} — check step accounting"


def test_kill_matches_shadow_at_equal_budget(tmp_path):
    shadow = _campaign(tmp_path / "shadow", "shadow")
    kill = _campaign(tmp_path / "kill", "kill")
    s_trials, k_trials = shadow["trials"], kill["trials"]
    killed = {gt for gt, t in k_trials.items() if t["state"] == "pruned"}

    _assert_shadow_arm_harmless(s_trials)
    _assert_kill_arm_feeds_optuna(kill, killed)
    _assert_matched_startup_asks(s_trials, k_trials)
    for gen in STARTUP_GENS:
        _assert_do_no_harm(gen, s_trials, k_trials, killed)
    _assert_step_savings(s_trials, k_trials, killed)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
