"""Pure-function tests for the scoring/aggregation/gating pipeline.

This is the optimizer's correctness core (dev_guidelines §7/§11): it must be
testable without a live optimization or a SOFA install.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sofaopt.core.algorithm import _aggregate_per_test, _gate_and_weight
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.scoring import (
    aggregate_repeats,
    combine_weighted,
    normalize_test_score,
)
from sofaopt.core.trialprep import params_from_trial
from sofaopt.project import ParamSpec, SofaOptProject, TestSpec


def _project(**kw) -> SofaOptProject:
    tmp = Path(tempfile.mkdtemp())
    base = dict(
        name="scoring-tests",
        work_dir=tmp,
        params=[
            ParamSpec("a", "float", 0.0, 1.0, default=0.5),
            ParamSpec("frozen", "float", 2.0, 2.0, default=2.0),
            ParamSpec("flag", "bool", 0, 1, default=False),
        ],
        tests=[
            TestSpec("cheap", scene_file=tmp / "s.py", max_score=10.0, weight=1.0),
            TestSpec(
                "expensive",
                scene_file=tmp / "s.py",
                max_score=100.0,
                weight=3.0,
                gated=True,
            ),
        ],
        sampler="random",  # avoid the CMA-ES n_parallel >= 4 constraint
    )
    base.update(kw)
    return SofaOptProject(**base)


def _cfg(**kw) -> RunConfig:
    return RunConfig.from_project(_project(**kw))


# ---------------------------------------------------------------------------
# normalize_test_score
# ---------------------------------------------------------------------------

def test_normalize_clamps_to_one():
    assert normalize_test_score(150.0, 100.0) == 1.0


def test_normalize_zero_or_negative_max_is_zero():
    assert normalize_test_score(5.0, 0.0) == 0.0
    assert normalize_test_score(5.0, -1.0) == 0.0


def test_normalize_plain_ratio():
    assert normalize_test_score(25.0, 100.0) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# aggregate_repeats / combine_weighted — the score pipeline
# ---------------------------------------------------------------------------

def test_aggregate_empty_scores():
    assert aggregate_repeats([]) == 0.0


@pytest.mark.parametrize(
    "mode,scores,expected",
    [
        ("mean", [1.0, 2.0, 6.0], 3.0),
        ("median", [1.0, 2.0, 6.0], 2.0),
        ("sum", [1.0, 2.0, 6.0], 9.0),
    ],
)
def test_aggregate_repeat_modes(mode, scores, expected):
    assert aggregate_repeats(scores, mode) == pytest.approx(expected)


def test_aggregate_exponential_coverage_rewards_multiple_positives():
    # 2 positive runs -> sum * 1.5^(2-1)
    assert aggregate_repeats([1.0, 2.0], "exponential_coverage") == pytest.approx(
        3.0 * 1.5
    )
    # no positive runs -> 0.0 multiplier
    assert aggregate_repeats([-1.0, 0.0], "exponential_coverage") == 0.0


def test_combine_weighted_cross_test_combination():
    # Two tests: 5/10 -> 0.5 normalized, 50/100 -> 0.5 normalized.
    # Weights 25% / 75% -> 0.5*25 + 0.5*75 = 50 (out of 100).
    final = combine_weighted(
        [5.0, 50.0],
        ["t1", "t2"],
        {"t1": 0.25, "t2": 0.75},
        {"t1": 10.0, "t2": 100.0},
    )
    assert final == pytest.approx(50.0)


def test_combine_weighted_clamps_overachievers():
    # 20/10 clamps to 1.0 before weighting.
    final = combine_weighted([20.0], ["t1"], {"t1": 1.0}, {"t1": 10.0})
    assert final == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# _aggregate_per_test — crashed-run isolation
# ---------------------------------------------------------------------------

def test_per_test_crashed_runs_score_zero_not_dropped():
    cfg = _cfg()
    run_results = [
        ("cheap", 8.0, 2),
        ("cheap", float("-inf"), 2),  # crashed repeat
    ]
    names, details = _aggregate_per_test(cfg, run_results)
    assert names == ["cheap"]
    d = details["cheap"]
    assert d["crashed_run_count"] == 1
    # crashed run counts as 0.0 in the mean: (8 + 0) / 2
    assert d["aggregate_score"] == pytest.approx(4.0)
    assert d["normalized_score"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# _gate_and_weight — the gating rule
# ---------------------------------------------------------------------------

def test_gate_closed_when_no_ungated_positive():
    cfg = _cfg()
    details = {
        "cheap": {"aggregate_score": 0.0},
        "expensive": {"aggregate_score": 50.0},
    }
    counted, weights, gate_open = _gate_and_weight(cfg, ["cheap", "expensive"], details)
    assert gate_open is False
    assert counted == ["cheap"]
    assert weights["cheap"] == pytest.approx(1.0)  # renormalized over counted


def test_gate_open_on_ungated_positive():
    cfg = _cfg()
    details = {
        "cheap": {"aggregate_score": 1.0},
        "expensive": {"aggregate_score": 50.0},
    }
    counted, weights, gate_open = _gate_and_weight(cfg, ["cheap", "expensive"], details)
    assert gate_open is True
    assert set(counted) == {"cheap", "expensive"}
    # declared weights 1:3 -> 0.25 / 0.75
    assert weights["cheap"] == pytest.approx(0.25)
    assert weights["expensive"] == pytest.approx(0.75)


def test_gate_open_when_nothing_gated():
    cfg = RunConfig.from_project(_project(), gated_names=[])
    details = {"cheap": {"aggregate_score": 0.0}, "expensive": {"aggregate_score": 0.0}}
    _, _, gate_open = _gate_and_weight(cfg, ["cheap", "expensive"], details)
    assert gate_open is True


# ---------------------------------------------------------------------------
# RunConfig derivation
# ---------------------------------------------------------------------------

def test_runconfig_normalizes_weights():
    cfg = _cfg()
    assert sum(cfg.test_weights.values()) == pytest.approx(1.0)
    assert cfg.test_weights["expensive"] == pytest.approx(0.75)


def test_runconfig_gated_defaults_from_specs():
    cfg = _cfg()
    assert cfg.gated_test_names == ("expensive",)


def test_runconfig_run_plan_repeats():
    cfg = RunConfig.from_project(_project())
    assert cfg.run_plan == (("cheap", 1, 1), ("expensive", 1, 1))
    assert cfg.n_repeats == 2


def test_runconfig_zero_weights_fall_back_to_uniform():
    cfg = RunConfig.from_project(
        _project(), weights={"cheap": 0.0, "expensive": 0.0}
    )
    assert cfg.test_weights["cheap"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# params_from_trial — frozen params and types
# ---------------------------------------------------------------------------

def test_params_from_trial_frozen_uses_default_and_is_not_suggested():
    import optuna

    project = _project()
    trial = optuna.trial.FixedTrial({"a": 0.7, "flag": True})
    params = params_from_trial(trial, project)
    assert params["frozen"] == 2.0
    assert params["a"] == pytest.approx(0.7)
    assert params["flag"] is True
    # the frozen param never reached Optuna
    assert "frozen" not in trial.params


# ---------------------------------------------------------------------------
# project JSON round-trip (subprocess handoff, replaces pickle)
# ---------------------------------------------------------------------------

def test_project_json_roundtrip_drops_hooks_and_keeps_fields():
    import json

    from sofaopt.project import project_from_jsonable, project_to_jsonable

    project = _project(
        prepare_trial=lambda params, d: None,
        record_frames=True,
        record_frame_size=(320, 240),
    )
    payload = json.loads(json.dumps(project_to_jsonable(project)))  # real JSON trip
    rebuilt = project_from_jsonable(payload)

    assert rebuilt.prepare_trial is None  # hooks cannot cross the boundary
    assert rebuilt.name == project.name
    assert rebuilt.work_dir == project.work_dir
    assert rebuilt.record_frame_size == (320, 240)
    assert [t.name for t in rebuilt.tests] == ["cheap", "expensive"]
    assert rebuilt.tests[1].gated is True
    assert [p.name for p in rebuilt.params] == ["a", "frozen", "flag"]
    assert rebuilt.params[0].default == 0.5


def test_params_from_trial_constrain_hook_applies():
    import optuna

    def constrain(p):
        p = dict(p)
        p["a"] = min(p["a"], 0.2)
        return p

    project = _project(constrain_params=constrain)
    trial = optuna.trial.FixedTrial({"a": 0.9, "flag": False})
    params = params_from_trial(trial, project)
    assert params["a"] == pytest.approx(0.2)
