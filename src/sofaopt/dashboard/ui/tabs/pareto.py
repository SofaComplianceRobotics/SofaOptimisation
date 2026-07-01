"""Pareto front tab for multi-objective optimization runs."""

from dash import dcc, html

from sofaopt.dashboard.context import LIVE_REFRESH_SECONDS


def build_pareto_tab() -> html.Div:
    return html.Div(
        [
            html.H3("Pareto Front", className="mb-1"),
            html.P(
                "Each point is one completed trial. "
                "Highlighted points are Pareto-optimal (non-dominated).",
                className="text-muted mb-3",
            ),
            html.Div(id="pareto-graphs"),
            dcc.Interval(
                id="pareto-interval",
                interval=int(max(1.0, LIVE_REFRESH_SECONDS) * 1000),
                n_intervals=0,
            ),
        ],
        className="p-3",
    )
