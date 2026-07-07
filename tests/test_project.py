"""Tests for the adapter contract: ParamSpec, TestSpec, SofaOptProject."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sofaopt import ParamSpec, SofaOptProject, TestSpec, param_specs_from_dataclass


def make_project(**overrides) -> SofaOptProject:
    """A minimal valid project; keyword overrides tweak individual fields."""
    defaults = dict(
        name="proj",
        work_dir=Path("."),
        params=[ParamSpec("k", "float", 0.0, 1.0, 0.5)],
        tests=[TestSpec("t1", scene_file="scenes/t1.py")],
        runsofa_exe=Path("runSofa"),
    )
    defaults.update(overrides)
    return SofaOptProject(**defaults)


class TestParamSpec:
    def test_float_with_range_is_active(self):
        assert not ParamSpec("k", "float", 0.0, 1.0, 0.5).is_frozen

    def test_equal_bounds_freeze_the_param(self):
        assert ParamSpec("k", "float", 2.0, 2.0, 2.0).is_frozen

    def test_bool_is_never_frozen(self):
        assert not ParamSpec("flag", "bool", default=True).is_frozen

    def test_to_dict_uses_min_max_keys(self):
        d = ParamSpec("k", "int", 1, 6, 3).to_dict()
        assert d == {"name": "k", "type": "int", "min": 1, "max": 6, "default": 3}


class TestParamSpecsFromDataclass:
    def test_only_annotated_fields_become_specs(self):
        @dataclass
        class Params:
            stiffness: float = field(
                default=10.0, metadata={"opt": {"type": "float", "min": 1.0, "max": 100.0}}
            )
            name: str = "not tunable"

        specs = param_specs_from_dataclass(Params())
        assert [s.name for s in specs] == ["stiffness"]
        assert specs[0].default == 10.0
        assert (specs[0].low, specs[0].high) == (1.0, 100.0)


class TestTestSpec:
    def test_label_defaults_to_name(self):
        assert TestSpec("grasp", scene_file="s.py").label == "grasp"

    def test_display_label_includes_description(self):
        spec = TestSpec("grasp", scene_file="s.py", label="Grasp", description="hold it")
        assert spec.display_label == "Grasp — hold it"


class TestSofaOptProject:
    def test_rejects_small_popsize(self):
        with pytest.raises(ValueError, match="n_parallel"):
            make_project(n_parallel=3)

    def test_rejects_empty_params(self):
        with pytest.raises(ValueError, match="params"):
            make_project(params=[])

    def test_rejects_empty_tests(self):
        with pytest.raises(ValueError, match="tests"):
            make_project(tests=[])

    def test_runtime_paths_hang_off_work_dir(self, tmp_path):
        p = make_project(work_dir=tmp_path)
        assert p.runtime_dir == tmp_path / "runtime"
        assert p.trials_dir == tmp_path / "runtime" / "trials"
        assert p.db_path == tmp_path / "runtime" / "study.db"

    def test_unknown_test_lookup_raises(self):
        with pytest.raises(KeyError, match="t2"):
            make_project().test("t2")

    def test_run_plan_flattens_repeats(self):
        p = make_project(
            tests=[
                TestSpec("a", scene_file="a.py", run_count=2),
                TestSpec("b", scene_file="b.py"),
            ]
        )
        assert p.run_plan == (("a", 1, 2), ("a", 2, 2), ("b", 1, 1))

    def test_scene_env_merges_project_env_over_os_env(self, monkeypatch):
        monkeypatch.setenv("SOFA_ROOT", "old")
        p = make_project(sofa_env={"SOFA_ROOT": "new", "EXTRA": 1})
        env = p.scene_env()
        assert env["SOFA_ROOT"] == "new"
        assert env["EXTRA"] == "1"
