"""Racing (adaptive re-evaluation) decision logic — no SOFA needed.

Covers the confidence half-width, the launch-time deferral predicate, the
keep/stop verdicts against a study incumbent (including max-score clamping),
and the TestSpec validation contract.
"""

from __future__ import annotations

import math

import optuna
import pytest

from sofaopt.core.generation.racing import (
    confidence_half_width,
    deferred_by_racing,
    evaluate_race,
    incumbent_score,
)
from sofaopt.core.runconfig import RunConfig
from sofaopt.project import ParamSpec, SofaOptProject, TestSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

_T2 = 4.303  # two-sided 95% t critical value, df=2


def _cfg(tmp_path, tests=None, **project_kw) -> RunConfig:
    project = SofaOptProject(
        name="race",
        work_dir=tmp_path,
        params=[ParamSpec("a", "float", 0.0, 1.0, default=0.5)],
        tests=tests
        or [
            TestSpec(
                "t", scene_file=tmp_path / "s.py",
                run_count=5, run_count_min=2, max_score=100.0,
            )
        ],
        runner="python",
        **project_kw,
    )
    return RunConfig.from_project(project)


def _study(best: float | None = None) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    if best is not None:
        study.add_trial(
            optuna.trial.create_trial(
                params={"a": 0.5},
                distributions={"a": optuna.distributions.FloatDistribution(0.0, 1.0)},
                value=best,
            )
        )
    return study


def _results(*scores: float, run_total: int = 5) -> list[tuple]:
    return [("t", s, run_total) for s in scores]


def test_confidence_half_width():
    assert confidence_half_width([]) is None
    assert confidence_half_width([1.0]) is None  # variance unknown at n=1
    assert confidence_half_width([5.0, 5.0, 5.0]) == 0.0
    # stdev([1,2,3]) = 1 -> hw = t95(df=2) / sqrt(3)
    hw = confidence_half_width([1.0, 2.0, 3.0])
    assert math.isclose(hw, _T2 / math.sqrt(3), rel_tol=1e-9)


def test_deferred_by_racing(tmp_path):
    cfg = _cfg(tmp_path)  # run_count=5, run_count_min=2
    assert not deferred_by_racing(cfg, "t", 1)
    assert not deferred_by_racing(cfg, "t", 2)
    assert deferred_by_racing(cfg, "t", 3)
    assert deferred_by_racing(cfg, "t", 5)


def test_no_racing_without_run_count_min(tmp_path):
    cfg = _cfg(tmp_path, tests=[
        TestSpec("t", scene_file=tmp_path / "s.py", run_count=5, max_score=100.0)
    ])
    assert not any(deferred_by_racing(cfg, "t", i) for i in range(1, 6))


def test_racing_disabled_for_multi_objective(tmp_path):
    cfg = _cfg(
        tmp_path,
        tests=[
            TestSpec("t", scene_file=tmp_path / "s.py",
                     run_count=5, run_count_min=2, max_score=100.0),
            TestSpec("u", scene_file=tmp_path / "s.py", max_score=100.0),
        ],
        multi_objective=True,
    )
    assert not deferred_by_racing(cfg, "t", 5)


def test_incumbent_score():
    assert incumbent_score(_study()) is None
    assert incumbent_score(_study(best=42.0)) == 42.0


def test_keep_running_without_incumbent(tmp_path):
    verdict = evaluate_race(_cfg(tmp_path), _study(), _results(10.0, 11.0), {"t"})
    assert verdict.keep_running


def test_keep_running_with_single_repeat(tmp_path):
    """n=1 has no variance estimate: always take a second repeat."""
    verdict = evaluate_race(_cfg(tmp_path), _study(best=80.0), _results(79.0), {"t"})
    assert verdict.keep_running


def test_stop_when_ci_below_incumbent(tmp_path):
    # mean 11, hw ~2.48 -> optimistic 13.5/100 -> cannot beat 80.
    verdict = evaluate_race(
        _cfg(tmp_path), _study(best=80.0), _results(10.0, 11.0, 12.0), {"t"}
    )
    assert not verdict.keep_running
    assert "incumbent 80.00" in verdict.reason


def test_keep_running_when_ci_overlaps_incumbent(tmp_path):
    # n=2, sd ~2.83 -> hw ~25.4 -> optimistic well above 80.
    verdict = evaluate_race(
        _cfg(tmp_path), _study(best=80.0), _results(78.0, 82.0), {"t"}
    )
    assert verdict.keep_running


def test_optimistic_bound_is_clamped_by_max_score(tmp_path):
    # mean 100 with spread: the raw upper bound exceeds max_score but the
    # final-score pipeline clamps at 100 — equal to the incumbent is a stop
    # (it cannot *beat* it).
    scores = _results(99.0, 100.0, 101.0)
    at_max = evaluate_race(_cfg(tmp_path), _study(best=100.0), scores, {"t"})
    assert not at_max.keep_running
    below_max = evaluate_race(_cfg(tmp_path), _study(best=99.99), scores, {"t"})
    assert below_max.keep_running


def test_crashed_repeats_count_as_zero(tmp_path):
    # Crashed runs score -inf and aggregate as 0.0, same as final scoring:
    # a crash-riddled candidate races out quickly.
    verdict = evaluate_race(
        _cfg(tmp_path), _study(best=50.0),
        _results(float("-inf"), float("-inf"), 3.0), {"t"},
    )
    assert not verdict.keep_running


def test_run_count_min_requires_mean_aggregation(tmp_path):
    with pytest.raises(ValueError, match="score_aggregation"):
        TestSpec(
            "t", scene_file=tmp_path / "s.py",
            run_count=5, run_count_min=2, score_aggregation="sum",
        )


def test_run_count_min_range_validated(tmp_path):
    with pytest.raises(ValueError, match="run_count_min"):
        TestSpec("t", scene_file=tmp_path / "s.py", run_count=5, run_count_min=0)
    with pytest.raises(ValueError, match="run_count_min"):
        TestSpec("t", scene_file=tmp_path / "s.py", run_count=5, run_count_min=6)
    # min == run_count is a no-op but legal.
    TestSpec("t", scene_file=tmp_path / "s.py", run_count=5, run_count_min=5)
