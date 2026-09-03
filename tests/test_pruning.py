"""Multi-fidelity step pruning: rung scheduler + contract validation — no SOFA.

Covers readiness gating, the quantile kill selection (terminal trials occupy
the bottom without being 'killed'), shadow vs kill application, the
missing-partial_score safety, rung idempotence, the Optuna bridge (design §8
checklist: report + tell(PRUNED) records intermediate values under ask/tell;
CmaEsSampler(consider_pruned_trials=True) keeps asking), and the
TestSpec / SofaOptProject validators.
"""

from __future__ import annotations

import optuna
import pytest

from sofaopt.core.generation.pruning import build_pruner
from sofaopt.core.generation.types import LaunchedTrial
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.trial_state import init_trial_state, read_trial_state, update_trial_run
from sofaopt.project import ParamSpec, SofaOptProject, TestSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

RUNGS = ((50, 0.75), (100, 0.5))


class _ExitedProc:
    """Stand-in for a finished Popen: prune_trial skips the kill."""

    def poll(self):
        return 0


def _cfg(tmp_path, prune_mode="kill", rungs=RUNGS, **project_kw) -> RunConfig:
    project = SofaOptProject(
        name="prune",
        work_dir=tmp_path,
        params=[ParamSpec("a", "float", 0.0, 1.0, default=0.5)],
        tests=[
            TestSpec(
                "t", scene_file=tmp_path / "s.py", max_score=100.0,
                prunable=True, prune_rungs=rungs,
            )
        ],
        runner="python",
        n_parallel=8,
        prune_mode=prune_mode,
        **project_kw,
    )
    return RunConfig.from_project(project)


def _entries(tmp_path, study, partials, frame=200):
    """One LaunchedTrial per partial score, with a live run slot at ``frame``.

    ``partials``: list of float (live run with that partial_score) or the
    string ``"failed"`` (terminal, unscored).
    """
    entries = []
    for i, partial in enumerate(partials, start=1):
        path = tmp_path / f"trial_{i:02d}" / "trial_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        init_trial_state(path, gen_index=1, trial_index=i, run_plan=[("t", 1, 1)])
        if partial == "failed":
            update_trial_run(path, 1, {"state": "failed", "score": None})
        else:
            update_trial_run(
                path, 1,
                {"state": "running", "current_frame": frame, "partial_score": partial},
            )
        entries.append(
            LaunchedTrial(
                trial_index=i,
                trial=study.ask(),
                trial_state_path=path,
                params_path=path.parent / "params.json",
                trial_env={},
                runs=[(_ExitedProc(), path, 1)],
            )
        )
    return entries


def _study() -> optuna.Study:
    return optuna.create_study(direction="maximize")


def _states(entries) -> list[str]:
    """Run-slot states (the level the rung scheduler reads and writes)."""
    return [
        str(read_trial_state(e.trial_state_path)["runs"][0].get("state"))
        for e in entries
    ]


# -- build_pruner gating --------------------------------------------------------


def test_build_pruner_none_when_off(tmp_path):
    assert build_pruner(_cfg(tmp_path, prune_mode="off"), 1, []) is None


def test_build_pruner_active_with_slot_and_rungs(tmp_path):
    pruner = build_pruner(_cfg(tmp_path), 1, [])
    assert pruner is not None
    assert pruner.run_slot == 1
    assert pruner.rungs == RUNGS


# -- readiness -------------------------------------------------------------------


def test_rung_waits_for_lagging_run(tmp_path):
    study = _study()
    entries = _entries(tmp_path, study, [10.0] * 7 + [20.0], frame=200)
    update_trial_run(
        entries[0].trial_state_path, 1,
        {"state": "running", "current_frame": 30, "partial_score": 5.0},
    )
    pruner = build_pruner(_cfg(tmp_path), 1, entries)
    pruner.check()
    assert pruner.fired == set()
    assert all(s == "running" for s in _states(entries))


def test_missing_partial_score_disables_pruning(tmp_path, caplog):
    study = _study()
    entries = _entries(tmp_path, study, [10.0] * 8, frame=200)
    update_trial_run(entries[2].trial_state_path, 1,
                     {"state": "running", "current_frame": 200, "partial_score": None})
    pruner = build_pruner(_cfg(tmp_path), 1, entries)
    with caplog.at_level("WARNING"):
        pruner.check()
    assert pruner.disabled
    assert "partial_score" in caplog.text
    assert all(s == "running" for s in _states(entries))


# -- quantile kills ---------------------------------------------------------------


def test_kill_mode_prunes_bottom_quantile(tmp_path):
    study = _study()
    partials = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    entries = _entries(tmp_path, study, partials)
    pruner = build_pruner(_cfg(tmp_path), 1, entries)
    pruner.check()  # both rungs ready at frame 200 -> fire in order

    # rung 1 keeps ceil(8*0.75)=6 (kills 2), rung 2 keeps 4 (kills 2 more)
    assert pruner.fired == {0, 1}
    states = _states(entries)
    assert states[:4] == ["pruned"] * 4
    assert all(s == "running" for s in states[4:])
    reason = str(read_trial_state(entries[0].trial_state_path).get("outcome"))
    assert "rung 50" in reason and "kept 6/8" in reason


def test_terminal_trials_occupy_the_bottom_without_kills(tmp_path):
    study = _study()
    entries = _entries(
        tmp_path, study,
        ["failed", "failed", 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    pruner = build_pruner(_cfg(tmp_path, rungs=((50, 0.5),)), 1, entries)
    pruner.check()
    states = _states(entries)
    # keep 4: doomed = 2 failed (-inf, no kill possible) + 2 lowest live
    assert states[0] == "failed" == states[1]
    assert states[2] == "pruned" == states[3]
    assert all(s == "running" for s in states[4:])


def test_rung_fires_once(tmp_path):
    study = _study()
    entries = _entries(tmp_path, study, [float(10 * i) for i in range(1, 9)])
    pruner = build_pruner(_cfg(tmp_path, rungs=((50, 0.5),)), 1, entries)
    pruner.check()
    first = _states(entries)
    pruner.check()
    assert _states(entries) == first
    assert pruner.fired == {0}


# -- shadow mode ------------------------------------------------------------------


def test_shadow_mode_marks_but_never_kills(tmp_path):
    study = _study()
    entries = _entries(tmp_path, study, [float(10 * i) for i in range(1, 9)])
    pruner = build_pruner(_cfg(tmp_path, prune_mode="shadow", rungs=((50, 0.5),)), 1, entries)
    pruner.check()
    assert all(s == "running" for s in _states(entries))
    for i, e in enumerate(entries):
        run = read_trial_state(e.trial_state_path)["runs"][0]
        if i < 4:
            assert run["shadow_prune_step"] == 50
            assert "rung 50" in run["shadow_prune_note"]
        else:
            assert "shadow_prune_step" not in run


# -- the Optuna bridge (design §8 checklist items) ---------------------------------


def test_pruned_trial_carries_partial_as_intermediate_value(tmp_path):
    study = _study()
    entries = _entries(tmp_path, study, [float(10 * i) for i in range(1, 9)])
    pruner = build_pruner(_cfg(tmp_path, rungs=((50, 0.5),)), 1, entries)
    pruner.check()

    # finalize would tell the killed trials PRUNED; do it here and assert the
    # partial reported at the rung survives as an intermediate value.
    killed = entries[0]
    study.tell(killed.trial, state=optuna.trial.TrialState.PRUNED)
    frozen = study.trials[killed.trial.number]
    assert frozen.state == optuna.trial.TrialState.PRUNED
    assert frozen.intermediate_values == {50: 10.0}


def test_cmaes_keeps_asking_with_pruned_intermediate_values(tmp_path):
    sampler = optuna.samplers.CmaEsSampler(
        popsize=4, n_startup_trials=4, consider_pruned_trials=True,
        independent_sampler=optuna.samplers.RandomSampler(seed=7),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for _ in range(4):  # 4 generations of popsize 4, half pruned each
        trials = [study.ask() for _ in range(4)]
        for i, t in enumerate(trials):
            t.suggest_float("a", 0.0, 1.0)
            t.report(float(i), step=50)
            if i < 2:
                study.tell(t, state=optuna.trial.TrialState.PRUNED)
            else:
                study.tell(t, float(i) * 10)
    assert len(study.trials) == 16
    assert study.best_value == 30.0
    # and the sampler still produces valid params afterwards
    t = study.ask()
    assert 0.0 <= t.suggest_float("a", 0.0, 1.0) <= 1.0


# -- validators --------------------------------------------------------------------


def _spec(**kw) -> TestSpec:
    return TestSpec("t", scene_file="s.py", **kw)


def test_testspec_prunable_contract():
    with pytest.raises(ValueError, match="prunable=False"):
        _spec(prune_rungs=((50, 0.5),))
    with pytest.raises(ValueError, match="run_count == 1"):
        _spec(prunable=True, run_count=3)
    with pytest.raises(ValueError, match="ungated"):
        _spec(prunable=True, gated=True)
    with pytest.raises(ValueError, match="non-relaunchable"):
        _spec(prunable=True, relaunchable=True)
    with pytest.raises(ValueError, match="ascending"):
        _spec(prunable=True, prune_rungs=((100, 0.75), (50, 0.5)))
    with pytest.raises(ValueError, match="non-increasing"):
        _spec(prunable=True, prune_rungs=((50, 0.5), (100, 0.75)))
    _spec(prunable=True, prune_rungs=RUNGS)  # valid


def _project(tests, n_parallel=8, **kw) -> SofaOptProject:
    return SofaOptProject(
        name="p", work_dir=".", runner="python", n_parallel=n_parallel,
        params=[ParamSpec("a", "float", 0.0, 1.0, default=0.5)],
        tests=tests, **kw,
    )


def test_project_prune_mode_requires_v1_shape(tmp_path):
    ok = [_spec(prunable=True, prune_rungs=RUNGS)]
    _project(ok, prune_mode="shadow")  # valid
    with pytest.raises(ValueError, match="exactly one test"):
        _project([_spec(), TestSpec("u", scene_file="u.py")], prune_mode="kill")
    with pytest.raises(ValueError, match="exactly one test"):
        _project([_spec()], prune_mode="kill")  # not prunable
    with pytest.raises(ValueError, match="single-objective"):
        _project(
            [_spec(prunable=True, prune_rungs=RUNGS), TestSpec("u", scene_file="u.py")],
            prune_mode="kill", multi_objective=True,
        )


def test_project_prune_warnings():
    ok = [_spec(prunable=True, prune_rungs=RUNGS)]
    with pytest.warns(UserWarning, match="n_parallel"):
        _project(ok, prune_mode="kill", n_parallel=4)
    with pytest.warns(UserWarning, match="mu"):
        _project(
            [_spec(prunable=True, prune_rungs=((50, 0.75), (100, 0.4)))],
            prune_mode="kill",
        )
