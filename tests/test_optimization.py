"""Tests for the optimizer/analysis layer added to sofaopt.

Covers the sampler wiring (CMA-ES-with-Margin, GP-BO, Sobol seeding) and the
interaction-analysis module (fANOVA main effects + surrogate interaction map).

Run with ``pytest`` from the repo root, or directly: ``python tests/test_optimization.py``.
No SOFA needed. The interaction-analysis tests need scikit-learn (the
``[analysis]`` extra) and skip cleanly when it is not installed.
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

import sofaopt.analysis as an
from sofaopt.core import algorithm as alg

optuna.logging.set_verbosity(optuna.logging.WARNING)


class _FakeProject:
    """Minimal stand-in exposing only the fields _seed_sampler reads."""

    def __init__(self, seed_sampler="random"):
        self.seed_sampler = seed_sampler


def test_seed_sampler_selection():
    assert type(alg._seed_sampler(_FakeProject("random"))).__name__ == "RandomSampler"
    assert type(alg._seed_sampler(_FakeProject("sobol"))).__name__ == "QMCSampler"


def test_new_samplers_construct():
    """CMA-ES-with-Margin and GP-BO build with a Sobol independent sampler."""
    seed = alg._seed_sampler(_FakeProject("sobol"))
    cm = optuna.samplers.CmaEsSampler(
        popsize=10, n_startup_trials=24, consider_pruned_trials=True,
        with_margin=True, independent_sampler=seed,
    )
    gp = optuna.samplers.GPSampler(n_startup_trials=24, independent_sampler=seed)
    assert isinstance(cm, optuna.samplers.CmaEsSampler)
    assert isinstance(gp, optuna.samplers.GPSampler)


def _coupled_study(n_trials=120, seed=0):
    """Synthetic study with a strong a*b interaction and a weak independent c."""
    rng = np.random.default_rng(seed)

    def objective(trial):
        a = trial.suggest_float("a", 0.0, 1.0)
        b = trial.suggest_float("b", 0.0, 1.0)
        c = trial.suggest_int("c", 0, 5)
        return 3.0 * a * b + 0.2 * c + rng.normal(0, 0.01)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def test_interaction_map_recovers_coupling():
    """The a*b pair must be the strongest interaction in the matrix."""
    pytest.importorskip("sklearn", reason="needs the [analysis] extra")
    report = an.analyze(_coupled_study(), method="h_stat")
    assert set(report.param_names) == {"a", "b", "c"}
    top = report.top_interactions(3)
    assert {top[0][0], top[0][1]} == {"a", "b"}, f"expected a*b strongest, got {top}"
    # a*b interaction clearly dominates the next pair.
    assert top[0][2] > top[1][2]


def test_main_effects_present_and_normalized():
    pytest.importorskip("sklearn", reason="needs the [analysis] extra")
    report = an.analyze(_coupled_study(), method="h_stat")
    me = report.main_effects
    assert set(me) == {"a", "b", "c"}
    assert abs(sum(me.values()) - 1.0) < 1e-6
    # a and b carry far more variance than the weak c term.
    assert me["a"] > me["c"] and me["b"] > me["c"]


def _minimal_project(**kw):
    import tempfile
    from pathlib import Path

    from sofaopt.project import ParamSpec, SofaOptProject, TestSpec

    tmp = Path(tempfile.mkdtemp())
    base = dict(
        name="ovr", work_dir=tmp,
        params=[ParamSpec("a", "float", 0.0, 1.0, default=0.5)],
        tests=[TestSpec("t", scene_file=tmp / "s.py", max_score=100.0, default_selected=True)],
        runner="python", n_parallel=10,
    )
    base.update(kw)
    return SofaOptProject(**base)


def test_env_overrides_applied(monkeypatch=None):
    """Dashboard env keys override project sampler fields in run_optimization."""
    import os

    from sofaopt.core import envkeys
    from sofaopt.core.orchestrator import _apply_env_overrides

    project = _minimal_project(sampler="cmaes", seed_sampler="random", cmaes_with_margin=False)
    saved = {k: os.environ.get(k) for k in
             (envkeys.SAMPLER, envkeys.SEED_SAMPLER, envkeys.CMAES_MARGIN, envkeys.N_PARALLEL)}
    try:
        os.environ[envkeys.SAMPLER] = "gp"
        os.environ[envkeys.SEED_SAMPLER] = "sobol"
        os.environ[envkeys.CMAES_MARGIN] = "1"
        os.environ[envkeys.N_PARALLEL] = "4"
        out = _apply_env_overrides(project)
        assert out.sampler == "gp"
        assert out.seed_sampler == "sobol"
        assert out.cmaes_with_margin is True
        assert out.n_parallel == 4
        # Original project is unchanged (frozen dataclass → replace returns a copy).
        assert project.sampler == "cmaes"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_env_overrides_noop_without_keys():
    from sofaopt.core.orchestrator import _apply_env_overrides

    project = _minimal_project(sampler="cmaes")
    assert _apply_env_overrides(project) is project  # no override keys → same object


def test_build_study_resume_preserves_trials():
    """Regression: resume=True must load prior trials, not wipe them."""
    import optuna

    from sofaopt.core.algorithm import build_study

    project = _minimal_project(sampler="random")  # stateless sampler, simple

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.project = project
    db = project.db_path
    db.parent.mkdir(parents=True, exist_ok=True)

    s1 = build_study(db, cfg, resume=False)
    s1.add_trial(optuna.trial.create_trial(
        params={"a": 0.5},
        distributions={"a": optuna.distributions.FloatDistribution(0.0, 1.0)},
        value=42.0,
    ))
    assert len(s1.trials) == 1

    # The bug: build_study used to delete the DB here. With the fix, resume
    # loads the existing study and keeps the trial.
    s2 = build_study(db, cfg, resume=True)
    assert len(s2.trials) == 1
    assert s2.trials[0].value == 42.0


def test_too_few_trials_raises():
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
    try:
        an.interaction_matrix(study)
    except ValueError:
        return
    raise AssertionError("expected ValueError for <4 completed trials")


if __name__ == "__main__":
    test_seed_sampler_selection()
    test_new_samplers_construct()
    test_interaction_map_recovers_coupling()
    test_main_effects_present_and_normalized()
    test_env_overrides_applied()
    test_env_overrides_noop_without_keys()
    test_build_study_resume_preserves_trials()
    test_too_few_trials_raises()
    print("All optimization-layer tests passed.")