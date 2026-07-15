"""Scene-side contract tests (§5) + process-timeout backstops (§10).

All SOFA-free: Trial/ScoreWriter don't import Sofa, and stuck processes are
faked. Covers the documented guarantees:

- a scene opened OUTSIDE the optimizer (plain runSofa) sees params == {},
  is_optimizing == False, and nothing raises or kills the process;
- write_score is idempotent and terminates only under the optimizer;
- the run_slot clamp for standalone launches (OPT_RUN_SLOT unset -> slot 1);
- wait_or_kill / wait_for_slot never hang on a wedged process.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from sofaopt.core import envkeys
from sofaopt.core.sofa_runner import wait_for_slot, wait_or_kill
from sofaopt.core.trial_state import init_trial_state, read_trial_run, read_trial_state
from sofaopt.scene import ScoreWriter, open_trial

_SCENE_KEYS = [
    envkeys.TRIAL_STATE_PATH, envkeys.PARAMS_PATH, envkeys.RUN_SLOT,
    envkeys.GEN, envkeys.TRIAL, envkeys.RUN, envkeys.TEST_NAME,
    envkeys.TEST_RUN_INDEX, envkeys.TEST_RUN_TOTAL,
]


@pytest.fixture()
def clean_env(monkeypatch):
    for key in _SCENE_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def kill_recorder(monkeypatch):
    """Intercept os.kill so a scored optimizer-mode trial can't kill pytest."""
    calls: list[tuple] = []
    monkeypatch.setattr(os, "kill", lambda *a: calls.append(a))
    return calls


# ---------------------------------------------------------------------------
# Standalone mode (plain `runSofa scene.py`, no optimizer env)
# ---------------------------------------------------------------------------

def test_standalone_trial_is_inert(clean_env, kill_recorder):
    trial = open_trial()
    assert trial.params == {}
    assert trial.is_optimizing is False
    # The full scoring API must be safe to call outside the optimizer:
    trial.write_status({"current_frame": 3})
    trial.write_score(1.23, reason="landed")
    trial.prune("ignored")  # after write_score: no-op via idempotency
    assert kill_recorder == []  # standalone never kills the process


def test_optimizer_env_is_parsed(clean_env, monkeypatch, tmp_path):
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps({"k": 2.5}), encoding="utf-8")
    monkeypatch.setenv(envkeys.TRIAL_STATE_PATH, str(tmp_path / "trial_state.json"))
    monkeypatch.setenv(envkeys.PARAMS_PATH, str(params_file))
    monkeypatch.setenv(envkeys.RUN_SLOT, "2")
    monkeypatch.setenv(envkeys.GEN, "3")
    monkeypatch.setenv(envkeys.TRIAL, "4")
    monkeypatch.setenv(envkeys.TEST_NAME, "cheap")

    trial = open_trial()
    assert trial.is_optimizing is True
    assert trial.params == {"k": 2.5}
    assert (trial.run_slot, trial.gen, trial.trial) == (2, 3, 4)
    assert trial.test_name == "cheap"


# ---------------------------------------------------------------------------
# ScoreWriter: idempotency, kill-on-score, slot clamp
# ---------------------------------------------------------------------------

def _state_file(tmp_path, n_slots=2):
    p = tmp_path / "trial_state.json"
    init_trial_state(
        p, gen_index=1, trial_index=1,
        run_plan=[("cheap", i + 1, n_slots) for i in range(n_slots)],
    )
    return p


def test_write_score_records_and_kills_under_optimizer(tmp_path, kill_recorder):
    state = _state_file(tmp_path)
    writer = ScoreWriter(None, {"gen": 1, "trial": 1, "run": 2}, str(state), run_slot=2)
    writer.write_score_and_stop(7.5, "done")
    slot = read_trial_run(state, 2)
    assert slot["state"] == "done" and slot["score"] == 7.5 and slot["reason"] == "done"
    assert len(kill_recorder) == 1  # optimizer mode: process is terminated


def test_write_score_is_idempotent(tmp_path, kill_recorder):
    state = _state_file(tmp_path)
    writer = ScoreWriter(None, {"gen": 1, "trial": 1, "run": 1}, str(state), run_slot=1)
    writer.write_score_and_stop(5.0, "first")
    writer.write_score_and_stop(99.0, "second")  # must be ignored
    writer.write_pruned_and_stop("also ignored")
    slot = read_trial_run(state, 1)
    assert slot["score"] == 5.0 and slot["reason"] == "first"
    assert len(kill_recorder) == 1


def test_run_slot_zero_clamps_to_one(tmp_path, kill_recorder):
    """OPT_RUN_SLOT unset (standalone-launched writer with a state file)."""
    state = _state_file(tmp_path)
    writer = ScoreWriter(None, {"gen": 0, "trial": 0, "run": 0}, str(state), run_slot=0)
    writer.write_score_and_stop(3.0, "clamped")
    assert read_trial_run(state, 1)["score"] == 3.0


def test_prune_marks_slot_unscored(tmp_path, kill_recorder):
    state = _state_file(tmp_path)
    writer = ScoreWriter(None, {"gen": 1, "trial": 1, "run": 1}, str(state), run_slot=1)
    writer.write_pruned_and_stop("too slow")
    slot = read_trial_run(state, 1)
    assert slot["state"] == "pruned" and slot["score"] is None
    assert read_trial_state(state)["state"] == "running"  # trial-level untouched


# ---------------------------------------------------------------------------
# Timeout backstops with a fake stuck process (§10)
# ---------------------------------------------------------------------------

class _StuckProc:
    """A process that never exits on its own."""

    def __init__(self):
        self.killed = False

    def poll(self):
        return 1 if self.killed else None

    def kill(self):
        self.killed = True


def test_wait_or_kill_kills_a_wedged_process():
    proc = _StuckProc()
    start = time.time()
    exited = wait_or_kill(proc, timeout_s=0.3)
    assert exited is False
    assert proc.killed is True
    assert time.time() - start < 5.0  # returned promptly after the deadline


def test_wait_or_kill_returns_true_on_clean_exit():
    class _DoneProc:
        def poll(self):
            return 0

    assert wait_or_kill(_DoneProc(), timeout_s=1.0) is True


def test_wait_for_slot_backstop_prevents_deadlock():
    """All slots wedged: the throttle must give up after its timeout, not hang."""

    class _Entry:
        def __init__(self):
            self.runs = [(_StuckProc(), None, 1)]

    launched = [_Entry(), _Entry()]
    start = time.time()
    wait_for_slot(launched, limit=1, gen_index=1, trial_index=1, timeout_s=0.4)
    elapsed = time.time() - start
    assert 0.3 < elapsed < 5.0  # waited for the backstop, then proceeded
