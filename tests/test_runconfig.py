"""Tests for RunConfig: test selection, weight normalization, gating, env parsing.

These carry over the behavioral guarantees of the old lab-side
``_parse_test_weights`` / ``_parse_gated_test_names`` tests, now expressed
against the RunConfig API that replaced them.
"""

import json
from pathlib import Path

import pytest

from sofaopt import ParamSpec, SofaOptProject, TestSpec
from sofaopt.core import envkeys
from sofaopt.core.runconfig import RunConfig


def make_project(tests=None) -> SofaOptProject:
    return SofaOptProject(
        name="proj",
        work_dir=Path("."),
        params=[ParamSpec("k", "float", 0.0, 1.0, 0.5)],
        tests=tests
        or [
            TestSpec("a", scene_file="a.py", weight=1.0),
            TestSpec("b", scene_file="b.py", weight=3.0),
            TestSpec("c", scene_file="c.py", weight=0.0, gated=True),
        ],
        runsofa_exe=Path("runSofa"),
    )


class TestSelection:
    def test_default_selects_all_tests(self):
        cfg = RunConfig.from_project(make_project())
        assert cfg.selected_names == ("a", "b", "c")

    def test_subset_selection_keeps_order(self):
        cfg = RunConfig.from_project(make_project(), selected_names=["b", "a"])
        assert cfg.selected_names == ("b", "a")

    def test_unknown_selection_raises(self):
        with pytest.raises(KeyError):
            RunConfig.from_project(make_project(), selected_names=["nope"])


class TestWeights:
    def test_declared_weights_normalised_to_fractions(self):
        cfg = RunConfig.from_project(make_project(), selected_names=["a", "b"])
        assert cfg.test_weights == {"a": 0.25, "b": 0.75}

    def test_weights_always_sum_to_one(self):
        cfg = RunConfig.from_project(
            make_project(), weights={"a": 20, "b": 30, "c": 50}
        )
        assert sum(cfg.test_weights.values()) == pytest.approx(1.0)

    def test_zero_total_falls_back_to_equal(self):
        cfg = RunConfig.from_project(make_project(), weights={"a": 0, "b": 0, "c": 0})
        assert cfg.test_weights == {name: 1 / 3 for name in ("a", "b", "c")}

    def test_missing_test_key_gets_zero_weight(self):
        cfg = RunConfig.from_project(
            make_project(), selected_names=["a", "b"], weights={"a": 100}
        )
        assert cfg.test_weights == {"a": 1.0, "b": 0.0}


class TestGating:
    def test_gated_defaults_from_spec_flags(self):
        cfg = RunConfig.from_project(make_project())
        assert cfg.gated_test_names == ("c",)

    def test_explicit_gated_filters_to_selected(self):
        cfg = RunConfig.from_project(
            make_project(), selected_names=["a", "b"], gated_names=["b", "c"]
        )
        assert cfg.gated_test_names == ("b",)


class TestRunPlan:
    def test_plan_flattens_selected_repeats(self):
        project = make_project(
            tests=[
                TestSpec("a", scene_file="a.py", run_count=2),
                TestSpec("b", scene_file="b.py", run_count=1),
            ]
        )
        cfg = RunConfig.from_project(project)
        assert cfg.run_plan == (("a", 1, 2), ("a", 2, 2), ("b", 1, 1))
        assert cfg.n_repeats == 3


class TestFromEnv:
    def test_no_env_selects_everything(self, monkeypatch):
        for key in (envkeys.SELECTED_TESTS, envkeys.TEST_WEIGHTS, envkeys.GATED_TESTS):
            monkeypatch.delenv(key, raising=False)
        cfg = RunConfig.from_env(make_project())
        assert cfg.selected_names == ("a", "b", "c")

    def test_env_selection_and_weights_applied(self, monkeypatch):
        monkeypatch.setenv(envkeys.SELECTED_TESTS, "a,b")
        monkeypatch.setenv(envkeys.TEST_WEIGHTS, json.dumps({"a": 40, "b": 60}))
        monkeypatch.setenv(envkeys.GATED_TESTS, "b")
        cfg = RunConfig.from_env(make_project())
        assert cfg.selected_names == ("a", "b")
        assert cfg.test_weights == {"a": 0.4, "b": 0.6}
        assert cfg.gated_test_names == ("b",)

    def test_malformed_weights_json_falls_back_to_declared(self, monkeypatch):
        monkeypatch.setenv(envkeys.SELECTED_TESTS, "a,b")
        monkeypatch.setenv(envkeys.TEST_WEIGHTS, "{not json")
        monkeypatch.delenv(envkeys.GATED_TESTS, raising=False)
        cfg = RunConfig.from_env(make_project())
        assert cfg.test_weights == {"a": 0.25, "b": 0.75}

    def test_base_scene_env_forwards_selection(self, monkeypatch):
        for key in (envkeys.SELECTED_TESTS, envkeys.TEST_WEIGHTS, envkeys.GATED_TESTS):
            monkeypatch.delenv(key, raising=False)
        cfg = RunConfig.from_project(make_project(), selected_names=["a", "b"])
        env = cfg.base_scene_env()
        assert env[envkeys.SELECTED_TESTS] == "a,b"
        assert json.loads(env[envkeys.TEST_WEIGHTS]) == {"a": 25, "b": 75}
