"""Adaptive convergence-driven termination (run_until_converged) — no SOFA."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sofaopt.core.orchestrator import _RestartProductivity
from sofaopt.project import ParamSpec, SofaOptProject
from sofaopt.project import TestSpec as _TestSpec


def _project(**kw) -> SofaOptProject:
    tmp = Path(tempfile.mkdtemp())
    base = dict(
        name="converged",
        work_dir=tmp,
        params=[ParamSpec("a", "float", 0.0, 1.0, default=0.5)],
        tests=[_TestSpec("t", scene_file=tmp / "s.py", max_score=100.0)],
        runner="python",
        n_parallel=4,
        sampler="cmaes",
        stall_generations=3,
        cmaes_restarts=5,
        run_until_converged=True,
        restart_patience=2,
    )
    base.update(kw)
    return SofaOptProject(**base)


# -- validation ----------------------------------------------------------------

def test_converged_config_is_valid():
    _project()  # the happy path must construct


@pytest.mark.parametrize(
    "override",
    [
        {"cmaes_restarts": 0, "stall_generations": 3},  # no restart budget
        {"stall_generations": 0, "cmaes_restarts": 0},  # no trigger (and no budget)
        {"restart_patience": 0},                        # patience must be >= 1
        {"sampler": "gp"},                              # cmaes-only
    ],
)
def test_converged_requires_its_preconditions(override):
    with pytest.raises(ValueError, match=r"run_until_converged|cmaes_restarts|stall_generations"):
        _project(**override)


def test_converged_rejects_multi_objective():
    with pytest.raises(ValueError):
        _project(
            multi_objective=True,
            tests=[
                _TestSpec("t", scene_file=Path(tempfile.mkdtemp()) / "s.py"),
                _TestSpec("u", scene_file=Path(tempfile.mkdtemp()) / "s.py"),
            ],
        )


def test_default_is_off():
    import dataclasses

    defaults = {f.name: f.default for f in dataclasses.fields(SofaOptProject)}
    assert defaults["run_until_converged"] is False
    assert defaults["restart_patience"] == 2


# -- productivity tracker (the adaptive stop) ----------------------------------

def test_productivity_stops_after_patience_fruitless_restarts():
    prod = _RestartProductivity(patience=2)
    # Basin 0 (initial): finds 10.0 -> productive, streak resets.
    assert prod.basin_ended(10.0) is False
    prod.restarted(10.0)
    # Basin 1: no improvement (still 10.0) -> fruitless #1.
    assert prod.basin_ended(10.0) is False
    prod.restarted(10.0)
    # Basin 2: still no improvement -> fruitless #2 == patience -> converged.
    assert prod.basin_ended(10.0) is True
    assert prod.streak == 2


def test_productive_restart_resets_the_streak():
    prod = _RestartProductivity(patience=2)
    prod.basin_ended(10.0)   # productive
    prod.restarted(10.0)
    assert prod.basin_ended(10.0) is False  # fruitless #1
    prod.restarted(10.0)
    assert prod.basin_ended(20.0) is False  # improved -> streak back to 0
    assert prod.streak == 0
    prod.restarted(20.0)
    assert prod.basin_ended(20.0) is False  # fruitless #1 again (not #2)


def test_no_completed_trial_counts_as_fruitless():
    prod = _RestartProductivity(patience=1)
    # A basin that produced no valid best (None) cannot have improved.
    assert prod.basin_ended(None) is True
    assert prod.streak == 1


# -- env overrides -------------------------------------------------------------

def test_env_overrides_parse_converged_and_patience():
    import os

    from sofaopt.core import envkeys
    from sofaopt.core.orchestrator import _apply_env_overrides

    project = _project(run_until_converged=False, restart_patience=2)
    saved = {k: os.environ.get(k) for k in (envkeys.RUN_UNTIL_CONVERGED, envkeys.RESTART_PATIENCE)}
    try:
        os.environ[envkeys.RUN_UNTIL_CONVERGED] = "true"
        os.environ[envkeys.RESTART_PATIENCE] = "4"
        out = _apply_env_overrides(project)
        assert out.run_until_converged is True
        assert out.restart_patience == 4
        assert project.run_until_converged is False  # original untouched (frozen copy)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -- progress payload in converged mode ----------------------------------------

def test_write_progress_converged_reports_restart_position():
    import json

    from sofaopt.core.runconfig import RunConfig
    from sofaopt.core.scoring import write_progress

    cfg = RunConfig.from_project(_project())
    cfg.project.trials_dir.mkdir(parents=True, exist_ok=True)
    rs = {
        "restart_index": 2, "restarts_max": 5, "stall_count": 1, "stall_limit": 3,
        "current_popsize": 16, "fruitless_streak": 1, "restart_patience": 2,
        "run_until_converged": True,
    }
    write_progress(cfg, 7, 0, [50.0], restart_state=rs)
    payload = json.loads(cfg.project.progress_file.read_text(encoding="utf-8"))
    # Self-sizing run: the generation-fraction percentage is meaningless.
    assert payload["gen_total"] is None
    assert payload["pct"] is None
    assert payload["restart_progress"] == "restart 2, 1/2 fruitless"
