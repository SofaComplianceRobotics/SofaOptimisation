"""IPOP-style CMA-ES restarts — no SOFA needed.

The load-bearing test proves the restart-scoped sampler genuinely starts a
FRESH CMA (the pre-restart serialized state is invisible to it) while the
old state stays readable by the old sampler — the seam ``core/restart.py``
relies on. If an Optuna upgrade changes the ``_attr_prefix`` /
``_restore_optimizer`` internals, this file trips before a run does.
"""

from __future__ import annotations

import optuna
import pytest

from sofaopt.core.restart import (
    RESTART_ATTR,
    _random_x0,
    _RestartScopedCmaEsSampler,
    maybe_restart,
    restart_index,
    restart_popsize,
)
from sofaopt.project import ParamSpec, SofaOptProject, TestSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

_COMPLETE = (optuna.trial.TrialState.COMPLETE,)


def _project(tmp_path, **kw):
    base = dict(
        name="restart",
        work_dir=tmp_path,
        params=[
            ParamSpec("x", "float", 0.0, 1.0, default=0.5),
            ParamSpec("y", "float", 0.0, 1.0, default=0.5),
        ],
        tests=[TestSpec("t", scene_file=tmp_path / "s.py", max_score=100.0)],
        runner="python",
        n_parallel=4,
        stall_generations=2,
        cmaes_restarts=2,
    )
    base.update(kw)
    return SofaOptProject(**base)


def _objective(trial) -> float:
    x = trial.suggest_float("x", 0.0, 1.0)
    y = trial.suggest_float("y", 0.0, 1.0)
    return -((x - 0.3) ** 2) - (y - 0.7) ** 2


def test_restart_config_validation(tmp_path):
    with pytest.raises(ValueError, match="stall_generations"):
        _project(tmp_path, cmaes_restarts=1, stall_generations=0)
    with pytest.raises(ValueError, match="cmaes_restarts"):
        _project(tmp_path, cmaes_restarts=-1)
    with pytest.raises(ValueError, match="cmaes_inc_popsize"):
        _project(tmp_path, cmaes_inc_popsize=0)


def test_maybe_restart_only_applies_to_single_objective_cmaes(tmp_path):
    study = optuna.create_study(direction="maximize")
    rs = optuna.samplers.RandomSampler()
    off = _project(tmp_path, cmaes_restarts=0, stall_generations=2)
    assert not maybe_restart(study, off, rs)
    gp = _project(tmp_path, sampler="gp")
    assert not maybe_restart(study, gp, rs)
    multi = _project(
        tmp_path,
        multi_objective=True,
        tests=[
            TestSpec("t", scene_file=tmp_path / "s.py"),
            TestSpec("u", scene_file=tmp_path / "s.py"),
        ],
    )
    assert not maybe_restart(study, multi, rs)


def test_random_x0_deterministic_and_in_bounds(tmp_path):
    project = _project(
        tmp_path,
        params=[
            ParamSpec("x", "float", -2.0, 3.0, default=0.0),
            ParamSpec("n", "int", 1, 8, default=4),
            ParamSpec("frozen", "float", 5.0, 5.0, default=5.0),
            ParamSpec("flag", "bool", default=True),
        ],
    )
    x0 = _random_x0(project, 1)
    assert set(x0) == {"x", "n"}  # frozen and bool params are excluded
    assert -2.0 <= x0["x"] <= 3.0
    assert 1 <= x0["n"] <= 8 and isinstance(x0["n"], int)
    assert x0 == _random_x0(project, 1)  # resume rebuilds the same sampler
    assert x0 != _random_x0(project, 2)  # each restart re-seeds elsewhere


def test_restart_swaps_in_fresh_scoped_cma(tmp_path):
    """The core mechanism: after maybe_restart the study's sampler is a fresh
    CMA-ES with a grown population that cannot see the old serialized state."""
    project = _project(tmp_path)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(popsize=4, n_startup_trials=2, seed=1),
    )
    study.optimize(_objective, n_trials=12)

    old = study.sampler
    completed = study.get_trials(deepcopy=False, states=_COMPLETE)
    # Precondition: the initial CMA has serialized state in trial attrs.
    assert old._restore_optimizer(completed) is not None

    assert maybe_restart(study, project, optuna.samplers.RandomSampler())
    new = study.sampler
    assert restart_index(study) == 1
    assert study.user_attrs[RESTART_ATTR] == 1
    assert isinstance(new, _RestartScopedCmaEsSampler)
    assert new._popsize == restart_popsize(project, 1) == 8
    # The restart is genuine: the old state is invisible to the new sampler,
    # so its first sample initializes a fresh optimizer at the new popsize.
    assert new._restore_optimizer(completed) is None

    # The new sampler persists its own state under the restart-scoped prefix.
    study.optimize(_objective, n_trials=10)
    completed = study.get_trials(deepcopy=False, states=_COMPLETE)
    assert new._restore_optimizer(completed) is not None
    assert any(
        key.startswith("cma:r1:optimizer")
        for t in completed
        for key in t.system_attrs
    )

    # Budget: 2 restarts allowed, then the stall stops the run as before.
    assert maybe_restart(study, project, optuna.samplers.RandomSampler())
    assert restart_index(study) == 2
    assert not maybe_restart(study, project, optuna.samplers.RandomSampler())


def test_build_study_resume_restores_restart_sampler(tmp_path):
    from sofaopt.core.algorithm import build_study
    from sofaopt.core.runconfig import RunConfig

    project = _project(tmp_path)
    cfg = RunConfig.from_project(project)
    db = project.db_path
    db.parent.mkdir(parents=True, exist_ok=True)

    s1 = build_study(db, cfg, resume=False)
    s1.set_user_attr(RESTART_ATTR, 1)

    s2 = build_study(db, cfg, resume=True)
    assert isinstance(s2.sampler, _RestartScopedCmaEsSampler)
    assert s2.sampler._attr_prefix == "cma:r1:"
    assert s2.sampler._popsize == restart_popsize(project, 1)

    # A run that never restarted resumes with the plain sampler.
    s1.set_user_attr(RESTART_ATTR, 0)
    s3 = build_study(db, cfg, resume=True)
    assert not isinstance(s3.sampler, _RestartScopedCmaEsSampler)


def test_stall_tracker_reset_gives_full_patience_again():
    from sofaopt.core.orchestrator import _StallTracker

    study = optuna.create_study(direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 0.5},
            distributions={"x": optuna.distributions.FloatDistribution(0.0, 1.0)},
            value=10.0,
        )
    )
    stall = _StallTracker(limit=2)
    assert not stall.should_stop(study)  # sets the baseline best
    assert not stall.should_stop(study)  # 1 stalled generation
    assert stall.should_stop(study)      # 2 -> fire
    stall.reset()
    assert not stall.should_stop(study)  # 1 stalled generation after reset
    assert stall.should_stop(study)      # 2 -> fires again
