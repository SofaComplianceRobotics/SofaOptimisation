"""Tests for trial_state.json CRUD — the optimizer/scene shared file."""

from sofaopt.core.trial_state import (
    init_trial_state,
    read_trial_run,
    read_trial_state,
    update_trial_run,
    update_trial_summary,
)

RUN_PLAN = [("grasp", 1, 2), ("grasp", 2, 2), ("tilt", 1, 1)]


def make_state(tmp_path):
    path = tmp_path / "trial_state.json"
    init_trial_state(
        path,
        gen_index=3,
        trial_index=7,
        run_plan=RUN_PLAN,
        params={"k": 1.5},
        test_weights={"grasp": 60, "tilt": 40},
        test_max_scores={"grasp": 10.0, "tilt": 5.0},
    )
    return path


class TestInit:
    def test_creates_one_slot_per_planned_run(self, tmp_path):
        data = read_trial_state(make_state(tmp_path))
        assert data["gen"] == 3
        assert data["trial"] == 7
        assert data["params"] == {"k": 1.5}
        assert len(data["runs"]) == 3
        assert [r["test_name"] for r in data["runs"]] == ["grasp", "grasp", "tilt"]

    def test_slots_start_unscored(self, tmp_path):
        data = read_trial_state(make_state(tmp_path))
        assert all(r["score"] is None for r in data["runs"])


class TestRunUpdates:
    def test_patch_touches_only_its_slot(self, tmp_path):
        path = make_state(tmp_path)
        update_trial_run(path, 2, {"state": "done", "score": 4.2})
        assert read_trial_run(path, 2)["score"] == 4.2
        assert read_trial_run(path, 1)["score"] is None

    def test_read_missing_slot_returns_none(self, tmp_path):
        assert read_trial_run(make_state(tmp_path), 99) is None


class TestSummaryUpdates:
    def test_summary_patch_preserves_runs(self, tmp_path):
        path = make_state(tmp_path)
        update_trial_run(path, 1, {"score": 1.0})
        update_trial_summary(path, {"state": "done", "final_score": 55.0})
        data = read_trial_state(path)
        assert data["state"] == "done"
        assert data["final_score"] == 55.0
        assert data["runs"][0]["score"] == 1.0
