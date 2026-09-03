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
    _WARM_SIGMA_INFLATE,
    build_restart_sampler,
    maybe_restart,
    restart_index,
    restart_popsize,
    warm_x0,
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
    with pytest.raises(ValueError, match="restart trigger"):
        _project(tmp_path, cmaes_restarts=1, stall_generations=0)
    with pytest.raises(ValueError, match="cmaes_restarts"):
        _project(tmp_path, cmaes_restarts=-1)
    with pytest.raises(ValueError, match="cmaes_inc_popsize"):
        _project(tmp_path, cmaes_inc_popsize=0)
    with pytest.raises(ValueError, match="warm_restarts"):
        _project(tmp_path, cmaes_restarts=0, stall_generations=0, warm_restarts=True)


def test_convergence_trigger_is_a_valid_restart_trigger(tmp_path):
    # restart_on_convergence replaces the stall plateau as the trigger, so
    # stall_generations > 0 is no longer required.
    project = _project(
        tmp_path, cmaes_restarts=2, stall_generations=0, restart_on_convergence=True
    )
    assert project.restart_on_convergence
    # ...and it can drive run_until_converged without a stall limit.
    conv = _project(
        tmp_path, cmaes_restarts=4, stall_generations=0, restart_on_convergence=True,
        run_until_converged=True, restart_patience=2,
    )
    assert conv.run_until_converged


def _restart(study, project, sampler, *, gen=1, trial_chron=0):
    return maybe_restart(study, project, sampler, gen=gen, trial_chron=trial_chron)


def test_maybe_restart_only_applies_to_single_objective_cmaes(tmp_path):
    study = optuna.create_study(direction="maximize")
    rs = optuna.samplers.RandomSampler()
    off = _project(tmp_path, cmaes_restarts=0, stall_generations=2)
    assert _restart(study, off, rs) is None
    gp = _project(tmp_path, sampler="gp")
    assert _restart(study, gp, rs) is None
    multi = _project(
        tmp_path,
        multi_objective=True,
        tests=[
            TestSpec("t", scene_file=tmp_path / "s.py"),
            TestSpec("u", scene_file=tmp_path / "s.py"),
        ],
    )
    assert _restart(study, multi, rs) is None


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

    assert _restart(study, project, optuna.samplers.RandomSampler())
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
    assert _restart(study, project, optuna.samplers.RandomSampler())
    assert restart_index(study) == 2
    assert _restart(study, project, optuna.samplers.RandomSampler()) is None


def test_maybe_restart_returns_event_and_writes_restarts_json(tmp_path):
    from sofaopt.core.restart_events import load_restart_events

    project = _project(tmp_path)
    project.trials_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(popsize=4, n_startup_trials=2, seed=1),
    )
    study.optimize(_objective, n_trials=12)

    event = maybe_restart(
        study, project, optuna.samplers.RandomSampler(), gen=3, trial_chron=12
    )
    assert event is not None
    assert event.restart_index == 1
    assert event.gen == 3
    assert event.trial_chron == 12
    assert event.old_popsize == 4
    assert event.new_popsize == restart_popsize(project, 1) == 8
    # Incumbent snapshot matches the study's best completed trial.
    assert event.incumbent_score == study.best_value
    assert event.incumbent_params == study.best_trial.params

    # The event is persisted to restarts.json (display log), deduped by index.
    persisted = load_restart_events(project.trials_dir)
    assert len(persisted) == 1
    assert persisted[0]["restart_index"] == 1
    assert persisted[0]["new_popsize"] == 8


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


def test_warm_x0_filters_to_searched_float_int_params(tmp_path):
    project = _project(
        tmp_path,
        params=[
            ParamSpec("x", "float", 0.0, 1.0, default=0.5),
            ParamSpec("n", "int", 1, 8, default=4),
            ParamSpec("frozen", "float", 5.0, 5.0, default=5.0),
            ParamSpec("flag", "bool", default=True),
        ],
    )
    incumbent = {"x": 0.7, "n": 6, "frozen": 5.0, "flag": False, "stray": 9.0}
    x0 = warm_x0(project, incumbent)
    assert x0 == {"x": 0.7, "n": 6}  # frozen, bool, and unknown keys excluded


def test_build_restart_sampler_warm_uses_incumbent_and_inflated_sigma(tmp_path):
    project = _project(tmp_path, cmaes_sigma0=0.3)
    warm = {"x": 0.7, "y": 0.2}
    sampler = build_restart_sampler(project, 1, optuna.samplers.RandomSampler(), warm_start=warm)
    assert sampler._x0 == warm
    assert sampler._sigma0 == project.cmaes_sigma0 * _WARM_SIGMA_INFLATE
    # Cold restart (no warm_start) keeps the base sigma and a random x0.
    cold = build_restart_sampler(project, 1, optuna.samplers.RandomSampler())
    assert cold._sigma0 == project.cmaes_sigma0
    assert cold._x0 != warm


def test_warm_restart_seeds_from_incumbent(tmp_path):
    project = _project(tmp_path, warm_restarts=True)
    project.trials_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(popsize=4, n_startup_trials=2, seed=1),
    )
    study.optimize(_objective, n_trials=12)
    event = maybe_restart(
        study, project, optuna.samplers.RandomSampler(), gen=3, trial_chron=12
    )
    assert event is not None
    # The new sampler's x0 is the incumbent's searched params, not random.
    assert study.sampler._x0 == warm_x0(project, study.best_trial.params)


def test_cma_converged_seam(tmp_path):
    """The convergence trigger's seam: should_stop becomes reachable+True once
    a CMA-ES study converges. Trips if Optuna moves the _restore_optimizer API."""
    from sofaopt.core.orchestrator import _cma_converged

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.CmaEsSampler(
            x0={"x0": 0.0, "x1": 0.0}, sigma0=0.2, popsize=6, n_startup_trials=6
        ),
    )

    def sphere(t):
        return sum(t.suggest_float(f"x{i}", -5.0, 5.0) ** 2 for i in range(2))

    assert not _cma_converged(study)  # nothing converged early
    study.optimize(sphere, n_trials=6 * 120)
    assert _cma_converged(study)      # a converged sphere reports should_stop
    assert study.best_value < 1e-6


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
