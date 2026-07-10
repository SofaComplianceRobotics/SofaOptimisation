"""Tests for run archiving (move semantics, manifest, restore) and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sofaopt.core.archive import (
    archive_run,
    archives_dir,
    best_so_far_curve,
    comparison_data,
    delete_archive,
    list_archives,
    restore_archive,
    runtime_has_run_data,
)
from sofaopt.project import ParamSpec, SofaOptProject, TestSpec


def _project(tmp_path) -> SofaOptProject:
    return SofaOptProject(
        name="arch-tests",
        work_dir=tmp_path,
        params=[ParamSpec("a", "float", 0.0, 1.0, default=0.5)],
        tests=[TestSpec("t", scene_file=tmp_path / "s.py", max_score=100.0)],
        sampler="random",
    )


def _fake_run(project, scores=(10.0, 30.0, 20.0)) -> None:
    """Lay down a minimal finished run in runtime/."""
    project.db_path.parent.mkdir(parents=True, exist_ok=True)
    project.db_path.write_text("not-a-real-db", encoding="utf-8")
    for i, score in enumerate(scores, 1):
        d = project.trials_dir / "gen_0001" / f"trial_{i:02d}"
        d.mkdir(parents=True)
        (d / "trial_state.json").write_text(
            json.dumps(
                {
                    "state": "done",
                    "final_score": score,
                    "params": {"a": 0.1 * i},
                    "runs": [{"run": 1, "test_name": "t", "score": score, "state": "done"}],
                }
            ),
            encoding="utf-8",
        )


def test_archive_moves_runtime_and_writes_manifest(tmp_path):
    project = _project(tmp_path)
    _fake_run(project)

    dest = archive_run(project, name="baseline run #1", notes="first try")

    assert not project.runtime_dir.exists()  # moved, not copied
    assert dest.parent == archives_dir(project)
    assert "baseline-run" in dest.name  # sanitized: '#'/' ' never hit the fs
    assert (dest / "study.db").exists()
    assert (dest / "trials" / "gen_0001" / "trial_02" / "trial_state.json").exists()

    (info,) = list_archives(project)
    assert info.name == "baseline run #1"
    assert info.notes == "first try"
    assert info.n_trials == 3
    assert info.best_score == 30.0
    assert info.best_params == {"a": 0.2}
    assert info.project_snapshot["name"] == "arch-tests"


def test_archive_requires_run_data(tmp_path):
    project = _project(tmp_path)
    assert runtime_has_run_data(project) is False
    with pytest.raises(FileNotFoundError):
        archive_run(project)


def test_restore_brings_run_back_and_protects_current(tmp_path):
    project = _project(tmp_path)
    _fake_run(project, scores=(10.0,))
    old = archive_run(project, name="old")

    _fake_run(project, scores=(99.0,))  # a new live run exists now
    restore_archive(project, old.name)

    # Old run is live again...
    state = json.loads(
        (project.trials_dir / "gen_0001" / "trial_01" / "trial_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["final_score"] == 10.0
    # ...and the 99.0 run was auto-archived, not destroyed.
    names = [a.name for a in list_archives(project)]
    assert any("auto_before_restore" in n for n in names)
    assert all(Path(n).name != "old" for n in names) or len(names) == 1


def test_delete_refuses_paths_outside_archives(tmp_path):
    project = _project(tmp_path)
    _fake_run(project)
    archive_run(project, name="keep")
    with pytest.raises(FileNotFoundError):
        delete_archive(project, "../runtime")  # traversal must not escape (§9)
    assert len(list_archives(project)) == 1


def test_delete_archive(tmp_path):
    project = _project(tmp_path)
    _fake_run(project)
    dest = archive_run(project)
    delete_archive(project, dest.name)
    assert list_archives(project) == []


def test_best_so_far_curve_monotone_and_skips_failures(tmp_path):
    project = _project(tmp_path)
    _fake_run(project, scores=(10.0, 30.0, 20.0))
    # add a failed trial that must advance x but not the best line
    d = project.trials_dir / "gen_0001" / "trial_04"
    d.mkdir()
    (d / "trial_state.json").write_text(
        json.dumps({"state": "failed", "final_score": -3.0, "runs": []}),
        encoding="utf-8",
    )
    xs, ys = best_so_far_curve(project.trials_dir)
    assert xs == [1, 2, 3, 4]
    assert ys == [10.0, 30.0, 30.0, 30.0]


def test_comparison_data_reads_recorded_scores(tmp_path):
    project = _project(tmp_path)
    _fake_run(project, scores=(10.0, 30.0))
    a1 = archive_run(project, name="run-a")
    _fake_run(project, scores=(50.0,))
    entries = comparison_data(project, [a1.name], include_current=True)

    assert [e["label"] for e in entries] == ["run-a", "current run"]
    assert entries[0]["best_score"] == 30.0
    assert entries[1]["best_score"] == 50.0
    assert entries[0]["curve"] == ([1, 2], [10.0, 30.0])
