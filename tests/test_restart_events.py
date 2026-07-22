"""Restart-event display log — schema + tolerant file I/O (no SOFA needed)."""

from __future__ import annotations

from sofaopt.core.restart_events import (
    RESTARTS_FILENAME,
    RestartEvent,
    load_restart_events,
    record_restart_event,
    restart_events_path,
)


def _event(index: int, **kw) -> RestartEvent:
    base = dict(
        restart_index=index, gen=index * 3, trial_chron=index * 12,
        old_popsize=4, new_popsize=8, sigma0=0.2,
        incumbent_score=42.0, timestamp=1.0, incumbent_params={"a": 0.5},
    )
    base.update(kw)
    return RestartEvent(**base)


def test_load_missing_file_returns_empty(tmp_path):
    assert load_restart_events(tmp_path) == []


def test_load_corrupt_file_returns_empty(tmp_path):
    restart_events_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert load_restart_events(tmp_path) == []


def test_round_trip(tmp_path):
    record_restart_event(tmp_path, _event(1))
    record_restart_event(tmp_path, _event(2, new_popsize=16))
    events = load_restart_events(tmp_path)
    assert [e["restart_index"] for e in events] == [1, 2]
    assert events[1]["new_popsize"] == 16
    assert events[0]["incumbent_params"] == {"a": 0.5}
    assert restart_events_path(tmp_path).name == RESTARTS_FILENAME


def test_dedup_by_restart_index(tmp_path):
    record_restart_event(tmp_path, _event(1, incumbent_score=42.0))
    record_restart_event(tmp_path, _event(1, incumbent_score=99.0))  # same index
    events = load_restart_events(tmp_path)
    assert len(events) == 1
    assert events[0]["incumbent_score"] == 42.0  # first write wins, no duplicate


def test_records_into_missing_trials_dir(tmp_path):
    # trials_dir need not exist yet — recording creates it (best-effort).
    target = tmp_path / "runtime" / "trials"
    record_restart_event(target, _event(1))
    assert restart_events_path(target).exists()
    assert load_restart_events(target)[0]["restart_index"] == 1
