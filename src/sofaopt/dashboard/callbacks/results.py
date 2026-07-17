"""Callbacks for the Results tab: score graph, trial detail, leaderboard,
health panel, and the on-demand search-space report."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Input, Output, html

from sofaopt.dashboard import context
from sofaopt.dashboard.data.cache import _load_data, _read_json
from sofaopt.dashboard.plotting.health import build_health_panel
from sofaopt.dashboard.plotting.performance import (
    _build_leaderboard_html,
    _build_performance_graph,
)
from sofaopt.dashboard.plotting.search_space import build_search_space_report
from sofaopt.dashboard.ui.progress import _build_trial_detail


def _trial_detail_children(click_data):
    """Build the per-trial detail panel from a score-graph click."""
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


def register_results_callbacks(app) -> None:
    """Register score graph, trial detail, leaderboard, health, report."""

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
    def update_results(tab, _):
        records, summaries = _load_data()
        if tab != "results":
            return go.Figure(), html.Div(), html.Div()
        progress = _read_json(context.progress_file())
        health = build_health_panel(records, context.project(), progress)
        return _build_performance_graph(records, summaries), _build_leaderboard_html(records), health

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
