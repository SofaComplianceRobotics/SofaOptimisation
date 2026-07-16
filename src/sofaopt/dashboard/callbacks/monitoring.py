"""Callbacks for the Performance, Progress, Bounds and Pareto tabs."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, State, ctx, html

from sofaopt.dashboard import context
from sofaopt.dashboard.data.cache import (
    _current_generation_records,
    _load_data,
    _read_json,
)
from sofaopt.dashboard.plotting.bounds import _build_param_bounds_graph
from sofaopt.dashboard.plotting.pareto import build_pareto_layout
from sofaopt.dashboard.plotting.health import build_health_panel
from sofaopt.dashboard.plotting.performance import (
    _build_leaderboard_html,
    _build_performance_graph,
)
from sofaopt.dashboard.plotting.search_space import build_search_space_report
from sofaopt.core.restart_events import load_restart_events
from sofaopt.dashboard.ui.progress import (
    _build_progress_grid,
    _build_progress_stats,
    _build_restart_status,
    _build_trial_detail,
    _find_earliest_not_done,
)


def _trial_detail_children(click_data):
    """Build the per-trial detail panel from a performance-graph click."""
    if not click_data:
        return html.Div()
    try:
        point = click_data["points"][0]
        cd = point.get("customdata")
        if not cd or len(cd) < 3:
            return html.Div()
        gen_name, trial_name = cd[1], cd[2]
        if not gen_name or not trial_name:
            return html.Div()
        state = _read_json(context.trials_dir() / gen_name / trial_name / "trial_state.json")
        if not state:
            return html.Div("No detail available for this trial.", className="text-muted")
        return _build_trial_detail(state, gen_name, trial_name)
    except Exception as exc:
        return html.Div(f"Could not load trial: {exc}", className="text-muted")


def _progress_children():
    """Restart-status panel + generation stats + trial grid for the Progress tab."""
    records, _summaries = _load_data()
    current_records = _current_generation_records(records)
    events = load_restart_events(context.trials_dir())
    progress = _read_json(context.progress_file())
    return (
        _build_restart_status(events, progress),
        _build_progress_stats(current_records, records),
        _build_progress_grid(current_records),
    )


def register_pareto_callbacks(app) -> None:
    """Register Pareto front tab callback (only wired when multi_objective=True)."""

    @app.callback(
        Output("pareto-graphs", "children"),
        Input("pareto-interval", "n_intervals"),
    )
    def update_pareto(_):
        project = context.project()
        test_names = [t.name for t in project.tests]
        directions = [t.direction for t in project.tests]
        records, _ = _load_data()
        done = [r for r in records if str(r.get("state", "")).lower() == "done"]
        return build_pareto_layout(done, test_names, directions)


def register_monitoring_callbacks(app) -> None:
    """Register performance graph, progress grid, bounds, and jump controls."""

    @app.callback(
        Output("trial-detail-panel", "children"),
        Input("performance-graph", "clickData"),
    )
    def on_trial_click(click_data):
        return _trial_detail_children(click_data)

    @app.callback(
        [
            Output("performance-graph", "figure"),
            Output("leaderboard-table", "children"),
            Output("optimization-health-panel", "children"),
        ],
        Input("tabs", "value"),
        Input("performance-interval", "n_intervals"),
    )
    def update_performance(tab, _):
        records, summaries = _load_data()
        if tab != "performance":
            return go.Figure(), html.Div(), html.Div()
        progress = _read_json(context.progress_file())
        health = build_health_panel(records, context.project(), progress)
        return _build_performance_graph(records, summaries), _build_leaderboard_html(records), health

    @app.callback(
        Output("param-bounds-graph", "figure"),
        Input("bounds-interval", "n_intervals"),
    )
    def update_bounds(_):
        return _build_param_bounds_graph(show_heatmap=True)

    @app.callback(
        Output("search-space-report-panel", "children"),
        Input("search-space-report-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_generate_report(n_clicks):
        if not n_clicks:
            return html.Div()
        records, _summaries = _load_data()
        return build_search_space_report(records)

    @app.callback(
        [
            Output("restart-status", "children"),
            Output("progress-stats", "children"),
            Output("progress-grid", "children"),
        ],
        Input("progress-interval", "n_intervals"),
    )
    def update_progress(_):
        return _progress_children()

    @app.callback(
        Output("jump-running-target-store", "data"),
        Input("jump-running-trial", "n_clicks"),
        Input("progress-interval", "n_intervals"),
        State("jump-auto-enabled", "data"),
    )
    def update_jump_target(_clicks, _intervals, auto_enabled):
        triggered = getattr(ctx, "triggered_id", None)
        is_auto = triggered == "progress-interval"
        if is_auto and not bool(auto_enabled):
            return {"target_id": None, "auto": True}
        records, _summaries = _load_data()
        current_records = _current_generation_records(records)
        return {"target_id": _find_earliest_not_done(current_records), "auto": is_auto}

    app.clientside_callback(
        """
        function(target, auto_enabled) {
            if (!target || !target.target_id) { return window.dash_clientside.no_update; }
            if (target.auto && !auto_enabled) { return window.dash_clientside.no_update; }
            const el = document.getElementById(target.target_id);
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'center'}); }
            return window.dash_clientside.no_update;
        }
        """,
        Output("jump-running-target-output", "children"),
        Input("jump-running-target-store", "data"),
        State("jump-auto-enabled", "data"),
    )

    app.clientside_callback(
        """
        function(n_intervals, auto_enabled) {
            if (!window._ajScrollListenerReady) {
                window._ajUserScrolled = false;
                var mark = function() { window._ajUserScrolled = true; };
                window.addEventListener('wheel',     mark, {passive: true});
                window.addEventListener('touchmove', mark, {passive: true});
                window.addEventListener('keydown', function(e) {
                    if ([' ','ArrowUp','ArrowDown','PageUp','PageDown','Home','End'].includes(e.key)) {
                        window._ajUserScrolled = true;
                    }
                }, {passive: true});
                window._ajScrollListenerReady = true;
            }
            if (!auto_enabled) { window._ajUserScrolled = false; return window.dash_clientside.no_update; }
            if (window._ajUserScrolled) { window._ajUserScrolled = false; return false; }
            return window.dash_clientside.no_update;
        }
        """,
        Output("jump-auto-enabled", "data", allow_duplicate=True),
        Input("progress-interval", "n_intervals"),
        State("jump-auto-enabled", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(n) {
            if (!n) { return window.dash_clientside.no_update; }
            window.scrollTo({top: 0, behavior: 'smooth'});
            return false;
        }
        """,
        Output("jump-auto-enabled", "data", allow_duplicate=True),
        Input("jump-top-button", "n_clicks"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        "function(n, cur) { if (!n) return window.dash_clientside.no_update; return !cur; }",
        Output("jump-auto-enabled", "data", allow_duplicate=True),
        Input("jump-auto-toggle", "n_clicks"),
        State("jump-auto-enabled", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        'function(on) { return on ? "Auto-jump: On" : "Auto-jump: Off"; }',
        Output("jump-auto-toggle", "children"),
        Input("jump-auto-enabled", "data"),
    )
