"""Tests for the shared trial_state.json IPC: locking, atomicity, slot patching.

Covers the §10 contract: concurrent writers never lose an update, a killed
lock-holder never wedges the file, and readers only ever see valid JSON.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from sofaopt.core import io
from sofaopt.core.trial_state import (
    init_trial_state,
    patch_run_slot,
    read_trial_run,
    read_trial_state,
    update_trial_run,
    update_trial_summary,
)


@pytest.fixture()
def state_path(tmp_path):
    p = tmp_path / "trial_state.json"
    init_trial_state(
        p,
        gen_index=1,
        trial_index=1,
        run_plan=[("cheap", 1, 1), ("expensive", 1, 1)],
        params={"a": 0.5},
    )
    return p


def test_init_prepopulates_slots(state_path):
    state = read_trial_state(state_path)
    assert state["gen"] == 1 and state["state"] == "running"
    assert [r["test_name"] for r in state["runs"]] == ["cheap", "expensive"]
    assert all(r["state"] == "not-started" for r in state["runs"])


def test_update_and_read_roundtrip(state_path):
    update_trial_run(state_path, 2, {"state": "done", "score": 3.5})
    slot = read_trial_run(state_path, 2)
    assert slot["score"] == 3.5 and slot["state"] == "done"
    # slot 1 untouched
    assert read_trial_run(state_path, 1)["state"] == "not-started"


def test_update_summary_preserves_runs(state_path):
    update_trial_summary(state_path, {"state": "done", "final_score": 42.0})
    state = read_trial_state(state_path)
    assert state["final_score"] == 42.0
    assert len(state["runs"]) == 2


def test_patch_run_slot_backfills_missing_slots():
    data: dict = {}
    patch_run_slot(data, 3, {"state": "done", "score": 1.0}, now=123.0)
    assert len(data["runs"]) == 3
    assert data["runs"][0] == {"run": 1}
    assert data["runs"][2]["score"] == 1.0
    assert data["runs"][2]["updated_at"] == 123.0


def test_stale_lock_is_broken(state_path):
    lock = state_path.with_suffix(state_path.suffix + ".lock")
    lock.write_text("")
    os.utime(lock, (0, 0))  # ancient lock: holder long dead
    update_trial_run(state_path, 1, {"state": "done", "score": 7.0})
    assert read_trial_run(state_path, 1)["score"] == 7.0
    assert not lock.exists()


def test_fresh_lock_blocks_until_released(state_path, monkeypatch):
    """A *live* lock is respected: the update waits for the holder."""
    lock = state_path.with_suffix(state_path.suffix + ".lock")
    lock.write_text("")

    def _release_soon():
        time.sleep(0.3)
        lock.unlink()

    t = threading.Thread(target=_release_soon)
    t.start()
    start = time.time()
    update_trial_run(state_path, 1, {"state": "done", "score": 9.0})
    t.join()
    assert time.time() - start >= 0.25  # actually waited for the holder
    assert read_trial_run(state_path, 1)["score"] == 9.0


def test_concurrent_writers_lose_no_update(state_path):
    """N threads patching distinct slots + the summary: every write survives."""
    n_writers = 8
    # grow the file to n_writers slots first
    update_trial_run(state_path, n_writers, {"state": "not-started"})

    errors: list[Exception] = []

    def _write(slot: int) -> None:
        try:
            update_trial_run(state_path, slot, {"state": "done", "score": float(slot)})
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(1, n_writers + 1)]
    threads.append(
        threading.Thread(
            target=update_trial_summary, args=(state_path, {"final_score": 99.0})
        )
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    state = read_trial_state(state_path)
    assert state["final_score"] == 99.0
    for slot in range(1, n_writers + 1):
        assert state["runs"][slot - 1]["score"] == float(slot), f"slot {slot} lost"


def test_file_is_always_valid_json_under_concurrent_readers(state_path):
    """Readers polling during writes must never see torn/partial JSON."""
    stop = threading.Event()
    bad: list[str] = []

    def _reader() -> None:
        while not stop.is_set():
            try:
                raw = state_path.read_text(encoding="utf-8")
                json.loads(raw)
            except json.JSONDecodeError:
                bad.append(raw[:80])
                return
            except OSError:
                pass  # transient share violation on Windows replace: retry

    r = threading.Thread(target=_reader)
    r.start()
    for i in range(50):
        update_trial_run(state_path, 1 + (i % 2), {"state": "running", "current_frame": i})
    stop.set()
    r.join()
    assert not bad, f"torn read observed: {bad[0]!r}"


def test_write_json_replaces_atomically(tmp_path):
    p = tmp_path / "x.json"
    io.write_json(p, {"v": 1})
    io.write_json(p, {"v": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 2}
    assert not p.with_suffix(p.suffix + ".tmp").exists()

# ---------------------------------------------------------------------------
# Slot-isolation contract (ported from the upstream engine test suite)
# ---------------------------------------------------------------------------

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
