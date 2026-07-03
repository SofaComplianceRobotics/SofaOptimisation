"""Duplicate-trial score cache + stall-stop settings (no SOFA needed)."""

from __future__ import annotations

import dataclasses

import optuna

from sofaopt.core.generation.launch import completed_score_index
from sofaopt.project import SofaOptProject


def _study_with(values: list[tuple[int, float]]) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    for x, v in values:
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": x},
                distributions={"x": optuna.distributions.IntDistribution(0, 10)},
                value=v,
            )
        )
    return study


def test_completed_score_index_dedups_and_skips_hard_fails():
    study = _study_with([(1, 50.0), (1, 60.0), (2, -3.0), (3, 70.0)])
    idx = completed_score_index(study, hard_fail_score=-3.0)
    # duplicates keep the best recorded value
    assert idx[(("x", 1),)][0] == 60.0
    # hard-fails are not cached (may be transient) -> a duplicate re-runs
    assert (("x", 2),) not in idx
    # (value, source trial number)
    assert idx[(("x", 3),)] == (70.0, 3)


def test_index_ignores_running_and_failed_trials():
    study = _study_with([(5, 42.0)])
    study.ask()  # RUNNING trial must not appear
    idx = completed_score_index(study, hard_fail_score=-3.0)
    assert list(idx) == [(("x", 5),)]


def test_new_features_default_off():
    defaults = {f.name: f.default for f in dataclasses.fields(SofaOptProject)}
    assert defaults["dedup_trials"] is False
    assert defaults["stall_generations"] == 0


def test_recover_interrupted_trials_reenqueues_and_closes():
    from sofaopt.core.algorithm import recover_interrupted_trials

    study = optuna.create_study(direction="maximize")
    t = study.ask()
    v = t.suggest_int("x", 0, 10)          # asked + suggested, never told (a pause)
    bare = study.ask()                     # asked, nothing suggested
    assert bare is not None

    n = recover_interrupted_trials(study)
    assert n == 1                          # only the trial WITH params is re-enqueued

    # stale RUNNING records are closed as FAILED
    assert not study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
    assert len(study.get_trials(states=(optuna.trial.TrialState.FAIL,))) == 2

    # the interrupted params come back on the next ask
    t2 = study.ask()
    assert t2.suggest_int("x", 0, 10) == v
