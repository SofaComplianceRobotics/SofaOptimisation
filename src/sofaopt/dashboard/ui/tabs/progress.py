"""Progress tab — live per-trial monitoring grid."""

from dash import dcc, html

from sofaopt.dashboard.context import LIVE_REFRESH_SECONDS


def build_progress_tab() -> html.Div:
    return html.Div(
        [
            html.H3("Optimization Progress", className="mb-3"),
            html.Div(
                [
                    html.Button(
                        "Jump to earliest unfinished trial",
                        id="jump-running-trial",
                        n_clicks=0,
                        className="btn btn-primary btn-sm",
                    ),
                ],
                className="d-flex align-items-center gap-2 mb-3",
                style={"position": "sticky", "top": "0", "zIndex": 20, "background": "#ffffff", "padding": "8px 0"},
            ),
            dcc.Store(id="jump-auto-enabled", data=False),
            html.Div(id="progress-stats", className="mb-3"),
            html.Div(id="progress-grid"),
            dcc.Store(id="jump-running-target-store"),
            html.Div(id="jump-running-target-output", style={"display": "none"}),
            html.Div(id="jump-top-output", style={"display": "none"}),
            html.Div(
                [
                    html.Button("Top", id="jump-top-button", n_clicks=0, className="btn btn-sm btn-secondary me-2"),
                    html.Button("Auto-jump: Off", id="jump-auto-toggle", n_clicks=0, className="btn btn-sm btn-outline-primary"),
                ],
                style={
                    "position": "fixed",
                    "right": "16px",
                    "bottom": "16px",
                    "zIndex": 9999,
                    "boxShadow": "0 6px 18px rgba(0,0,0,0.12)",
                    "padding": "8px",
                    "borderRadius": "8px",
                    "background": "#ffffff",
                },
            ),
            dcc.Interval(
                id="progress-interval",
                interval=int(max(1.0, LIVE_REFRESH_SECONDS) * 1000),
                n_intervals=0,
            ),
        ],
        className="p-3",
    )
