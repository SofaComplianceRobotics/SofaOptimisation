"""Tests for the scene-side API: open_trial and score writing."""

import json

from sofaopt.core import envkeys
from sofaopt.core.trial_state import init_trial_state, read_trial_run
from sofaopt.scene import open_trial

ALL_KEYS = (
    envkeys.TRIAL_STATE_PATH,
    envkeys.PARAMS_PATH,
    envkeys.RUN_SLOT,
    envkeys.TEST_NAME,
    envkeys.TEST_RUN_INDEX,
    envkeys.TEST_RUN_TOTAL,
    envkeys.GEN,
    envkeys.TRIAL,
    envkeys.RUN,
)


def clear_env(monkeypatch):
    for key in ALL_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestOutsideOptimizer:
    def test_manual_launch_is_not_optimizing(self, monkeypatch):
        clear_env(monkeypatch)
        trial = open_trial()
        assert not trial.is_optimizing
        assert trial.params == {}

    def test_run_slot_read_independently_of_trial_state(self, monkeypatch):
        # A manual launch may still select a slot (e.g. cube size) without the
        # optimizer's trial-state file being present.
        clear_env(monkeypatch)
        monkeypatch.setenv(envkeys.RUN_SLOT, "2")
        trial = open_trial()
        assert trial.run_slot == 2
        assert not trial.is_optimizing


class TestUnderOptimizer:
    def test_reads_params_and_run_identity(self, tmp_path, monkeypatch):
        clear_env(monkeypatch)
        params_path = tmp_path / "params.json"
        params_path.write_text(json.dumps({"stiffness": 2.5}), encoding="utf-8")
        state_path = tmp_path / "trial_state.json"
        init_trial_state(state_path, gen_index=1, trial_index=1, run_plan=[("grasp", 1, 1)])

        monkeypatch.setenv(envkeys.TRIAL_STATE_PATH, str(state_path))
        monkeypatch.setenv(envkeys.PARAMS_PATH, str(params_path))
        monkeypatch.setenv(envkeys.RUN_SLOT, "1")
        monkeypatch.setenv(envkeys.TEST_NAME, "grasp")
        monkeypatch.setenv(envkeys.GEN, "4")

        trial = open_trial()
        assert trial.is_optimizing
        assert trial.params == {"stiffness": 2.5}
        assert trial.test_name == "grasp"
        assert trial.gen == 4

    def test_write_status_lands_in_the_right_slot(self, tmp_path, monkeypatch):
        clear_env(monkeypatch)
        state_path = tmp_path / "trial_state.json"
        init_trial_state(
            state_path, gen_index=1, trial_index=1, run_plan=[("grasp", 1, 2), ("grasp", 2, 2)]
        )
        monkeypatch.setenv(envkeys.TRIAL_STATE_PATH, str(state_path))
        monkeypatch.setenv(envkeys.RUN_SLOT, "2")

        trial = open_trial()
        trial.write_status({"state": "running", "current_frame": 10})

        assert read_trial_run(state_path, 2)["current_frame"] == 10
        assert read_trial_run(state_path, 1)["current_frame"] == 0

    def test_finished_flips_after_status_free_score_write(self, tmp_path, monkeypatch):
        clear_env(monkeypatch)
        state_path = tmp_path / "trial_state.json"
        init_trial_state(state_path, gen_index=1, trial_index=1, run_plan=[("grasp", 1, 1)])
        monkeypatch.setenv(envkeys.TRIAL_STATE_PATH, str(state_path))
        monkeypatch.setenv(envkeys.RUN_SLOT, "1")

        trial = open_trial()
        assert not trial.finished
        # write_status alone must not mark the run finished
        trial.write_status({"state": "running"})
        assert not trial.finished

    def test_interactive_score_write_marks_finished_without_exiting(self, monkeypatch):
        # No trial_state_path: scoring stops the sim but must not kill the
        # process (hand-launched scenes), and finished flips to True.
        clear_env(monkeypatch)
        trial = open_trial()
        trial.write_score(1.0, reason="done")
        assert trial.finished

    def test_carry_round_trip(self, tmp_path, monkeypatch):
        clear_env(monkeypatch)
        state_path = tmp_path / "trial_state.json"
        init_trial_state(state_path, gen_index=1, trial_index=1, run_plan=[("probe", 1, 1)])
        monkeypatch.setenv(envkeys.TRIAL_STATE_PATH, str(state_path))
        monkeypatch.setenv(envkeys.RUN_SLOT, "1")

        first = open_trial()
        assert first.load_carry() == {}
        first.save_carry({"weight_g": 30})

        relaunched = open_trial()
        assert relaunched.load_carry() == {"weight_g": 30}
