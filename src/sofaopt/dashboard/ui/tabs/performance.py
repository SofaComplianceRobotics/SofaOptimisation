"""Performance tab — score graph + leaderboard."""

from dash import dcc, html

from sofaopt.dashboard.context import LIVE_REFRESH_SECONDS


def build_performance_tab() -> html.Div:
    return html.Div(
        [
            html.H3("Performance", className="mb-3"),
            dcc.Graph(id="performance-graph", style={"height": "600px"}),
            html.Div(id="trial-detail-panel", className="my-3"),
            html.Hr(),
            html.Div(id="leaderboard-table", className="mt-4"),
            dcc.Interval(
                id="performance-interval",
                interval=int(max(1.0, LIVE_REFRESH_SECONDS) * 1000),
                n_intervals=0,
            ),
        ],
        className="p-3",
    )
