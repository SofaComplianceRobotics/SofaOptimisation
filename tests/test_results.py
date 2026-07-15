"""Tests for sofaopt.core.results — the canonical score read path (§11)."""

from __future__ import annotations

import json

import pytest

from sofaopt.core.results import load_trial_records, rank_completed


def _write_state(tmp_path, gen, trial, payload):
    d = tmp_path / f"gen_{gen:04d}" / f"trial_{trial:02d}"
    d.mkdir(parents=True)
    (d / "trial_state.json").write_text(json.dumps(payload), encoding="utf-8")


def test_recorded_final_score_is_never_recomputed(tmp_path):
    """The record must expose the study's score even when a naive recompute
    from the per-test breakdown would disagree (e.g. gate-closed trials)."""
    _write_state(
        tmp_path, 1, 1,
        {
            "state": "done",
            "final_score": 25.0,  # gate closed: only 'cheap' counted at 100%
            "active_test_weights": {"cheap": 100.0},
            "test_scores": {
                "cheap": {
                    "aggregate_score": 2.5, "max_score": 10.0,
                    "normalized_score": 0.25, "weight_pct": 25.0,
                },
                "expensive": {
                    "aggregate_score": 90.0, "max_score": 100.0,
                    "normalized_score": 0.9, "weight_pct": 75.0,
                },
            },
            "runs": [{"run": 1, "test_name": "cheap", "score": 2.5, "state": "done"}],
        },
    )
    (rec,) = load_trial_records(tmp_path)
    assert rec["final_score"] == 25.0
    # contributions use the gate-renormalized weights and sum to the score
    assert rec["contributions"]["cheap"] == pytest.approx(25.0)
    assert rec["contributions"]["expensive"] == 0.0
    assert sum(rec["contributions"].values()) == pytest.approx(rec["final_score"])


def test_legacy_state_without_final_score_reconstructs(tmp_path):
    _write_state(
        tmp_path, 1, 1,
        {
            "state": "done",
            "test_weights": {"t": 1.0},
            "test_max_scores": {"t": 10.0},
            "runs": [
                {"run": 1, "test_name": "t", "score": 4.0, "state": "done"},
                {"run": 2, "test_name": "t", "score": 6.0, "state": "done"},
            ],
        },
    )
    (rec,) = load_trial_records(tmp_path)
    # mean(4, 6) = 5 → 5/10 normalized → 50/100 weighted
    assert rec["final_score"] == pytest.approx(50.0)


def test_rank_completed_sorts_and_filters(tmp_path):
    _write_state(tmp_path, 1, 1, {"state": "done", "final_score": 10.0, "runs": []})
    _write_state(tmp_path, 1, 2, {"state": "done", "final_score": 30.0, "runs": []})
    _write_state(tmp_path, 1, 3, {"state": "running", "runs": []})  # not terminal
    ranked = rank_completed(load_trial_records(tmp_path))
    assert [r["trial_name"] for r in ranked] == ["trial_02", "trial_01"]
