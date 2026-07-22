"""Dashboard restart visibility — markers + status panel (no browser/SOFA)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import plotly.graph_objects as go

from sofaopt.core.restart_events import RestartEvent, record_restart_event
from sofaopt.project import ParamSpec, SofaOptProject
from sofaopt.project import TestSpec as _TestSpec


def _project() -> SofaOptProject:
    tmp = Path(tempfile.mkdtemp())
    return SofaOptProject(
        name="dash_restart",
        work_dir=tmp,
        params=[ParamSpec("a", "float", 0.0, 1.0, default=0.5)],
        tests=[_TestSpec("t", scene_file=tmp / "s.py", max_score=100.0)],
        runner="python",
        n_parallel=4,
    )


def _event(index: int, chron: int) -> RestartEvent:
    return RestartEvent(
        restart_index=index, gen=index * 2, trial_chron=chron,
        old_popsize=4, new_popsize=8, sigma0=0.2,
        incumbent_score=55.0, timestamp=1.0, incumbent_params={"a": 0.5},
    )


# -- convergence-curve markers -------------------------------------------------

def test_add_restart_markers_draws_a_vline_per_event():
    from sofaopt.dashboard import context
    from sofaopt.dashboard.plotting.performance import _add_restart_markers

    project = _project()
    context.set_project(project)
    project.trials_dir.mkdir(parents=True, exist_ok=True)
    record_restart_event(project.trials_dir, _event(1, 8))
    record_restart_event(project.trials_dir, _event(2, 20))

    fig = go.Figure()
    _add_restart_markers(fig)
    # add_vline appends a shape per restart.
    assert len(fig.layout.shapes) == 2


def test_add_restart_markers_noop_without_events():
    from sofaopt.dashboard import context
    from sofaopt.dashboard.plotting.performance import _add_restart_markers

    context.set_project(_project())  # fresh work_dir, no restarts.json
    fig = go.Figure()
    _add_restart_markers(fig)
    assert len(fig.layout.shapes) == 0


# -- archives comparison markers (single run only) -----------------------------

def _entry(label: str, with_restart: bool = True) -> dict:
    return {
        "label": label,
        "curve": ([1, 2, 3], [10.0, 20.0, 30.0]),
        "restarts": [{"trial_chron": 2, "restart_index": 1}] if with_restart else [],
    }


def test_comparison_markers_only_for_single_run():
    from sofaopt.dashboard.plotting.archives import build_comparison_figure

    one = build_comparison_figure([_entry("run A")], {"run A": 0})
    assert len(one.layout.shapes) == 1

    two = build_comparison_figure(
        [_entry("run A"), _entry("run B")], {"run A": 0, "run B": 1}
    )
    assert len(two.layout.shapes) == 0  # multi-run: markers would mislead


# -- live restart-status panel -------------------------------------------------

def test_restart_status_hidden_when_restarts_off():
    from sofaopt.dashboard.ui.progress import _build_restart_status

    panel = _build_restart_status([], {"restart": {"restarts_max": 0}})
    assert panel.children is None  # empty div -> nothing rendered


def test_restart_status_shows_latest_event():
    from sofaopt.dashboard.ui.progress import _build_restart_status

    events = [
        {"restart_index": 1, "old_popsize": 4, "new_popsize": 8, "incumbent_score": 55.0},
        {"restart_index": 2, "old_popsize": 8, "new_popsize": 16, "incumbent_score": 61.0},
    ]
    progress = {"restart": {"restarts_max": 3, "stall_count": 4, "stall_limit": 10}}
    text = str(_build_restart_status(events, progress))
    assert "8 → 16" in text  # latest event's population step
    assert "61.00" in text   # best-at-restart
    assert "4 / 10 gens" in text  # patience countdown


def test_restart_status_countdown_before_first_restart():
    from sofaopt.dashboard.ui.progress import _build_restart_status

    progress = {"restart": {"restarts_max": 3, "stall_count": 2, "stall_limit": 8,
                            "current_popsize": 4}}
    text = str(_build_restart_status([], progress))
    assert "0 / 3" in text        # no restart yet
    assert "2 / 8 gens" in text   # generations since improvement


def test_restart_status_converged_mode_shows_fruitless_streak():
    from sofaopt.dashboard.ui.progress import _build_restart_status

    events = [{"restart_index": 1, "old_popsize": 4, "new_popsize": 8, "incumbent_score": 50.0}]
    progress = {"restart": {"restarts_max": 5, "stall_count": 1, "stall_limit": 6,
                            "run_until_converged": True, "fruitless_streak": 1,
                            "restart_patience": 2}}
    text = str(_build_restart_status(events, progress))
    assert "run until converged" in text
    assert "1 / 2" in text  # fruitless restarts toward patience
