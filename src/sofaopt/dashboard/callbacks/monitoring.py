"""Callbacks for the Performance, Progress and Bounds tabs."""

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
from sofaopt.dashboard.plotting.performance import (
    _build_leaderboard_html,
    _build_performance_graph,
)
from sofaopt.dashboard.ui.progress import (
    _build_progress_grid,
    _build_progress_stats,
    _build_trial_detail,
    _find_earliest_not_done,
)


def register_monitoring_callbacks(app) -> None:
    """Register performance graph, progress grid, bounds, and jump controls."""

    @app.callback(
        Output("trial-detail-panel", "children"),
        Input("performance-graph", "clickData"),
    )
    def on_trial_click(click_data):
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

    @app.callback(
        [Output("performance-graph", "figure"), Output("leaderboard-table", "children")],
        Input("tabs", "value"),
        Input("performance-interval", "n_intervals"),
    )
    def update_performance(tab, _):
        records, summaries = _load_data()
        if tab != "performance":
            return go.Figure(), html.Div()
        return _build_performance_graph(records, summaries), _build_leaderboard_html(records)

    @app.callback(
        Output("param-bounds-graph", "figure"),
        Input("bounds-interval", "n_intervals"),
    )
    def update_bounds(_):
        return _build_param_bounds_graph(show_heatmap=True)

    @app.callback(
        [Output("progress-stats", "children"), Output("progress-grid", "children")],
        Input("progress-interval", "n_intervals"),
    )
    def update_progress(_):
        records, _summaries = _load_data()
        current_records = _current_generation_records(records)
        return (
            _build_progress_stats(current_records, records),
            _build_progress_grid(current_records),
        )

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
        "function(n) { return window.dash_clientside.no_update; }",
        Output("jump-top-output", "children"),
        Input("jump-top-button", "n_clicks"),
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
