"""Parameter Bounds tab — where the latest trial sits within each range."""

from dash import dcc, html

from sofaopt.dashboard.context import LIVE_REFRESH_SECONDS


def build_param_bounds_tab() -> html.Div:
    return html.Div(
        [
            html.H3("Parameter Bounds Monitor", className="mb-3"),
            html.P("Live tracking of parameter values within optimization bounds."),
            dcc.Graph(id="param-bounds-graph"),
            dcc.Interval(
                id="bounds-interval",
                interval=int(max(1.0, LIVE_REFRESH_SECONDS) * 1000),
                n_intervals=0,
            ),
        ],
        className="p-3",
    )
