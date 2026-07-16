"""Performance tab — score graph + leaderboard."""

from dash import dcc, html

from sofaopt.dashboard.context import LIVE_REFRESH_SECONDS


def build_performance_tab() -> html.Div:
    return html.Div(
        [
            # Shared state stores
            dcc.Store(id="selected-trial-store", data={}),

            html.H3("Performance", className="mb-3"),
            html.Div(id="optimization-health-panel", className="mb-3"),
            dcc.Graph(id="performance-graph", style={"height": "600px"}),
            html.Div(id="trial-detail-panel", className="my-3"),

            # Video panel — shown once a trial is selected
            html.Div(
                [
                    html.Hr(className="my-2"),
                    html.Div(
                        [
                            html.Span("Video", className="fw-semibold small text-muted me-3"),
                            html.Span(id="video-status-text", className="small text-muted me-3"),
                            html.Button(
                                "Test it",
                                id="test-it-btn",
                                n_clicks=0,
                                className="btn btn-sm btn-outline-primary",
                            ),
                        ],
                        className="d-flex align-items-center flex-wrap gap-2",
                    ),
                ],
                id="video-controls-panel",
                className="px-1 mb-3",
                style={"display": "none"},
            ),

            # Summary video panel — always visible, shows status / view link
            html.Div(
                [
                    html.Hr(className="my-2"),
                    html.Div(
                        [
                            html.Span("Summary Video", className="fw-semibold small text-muted me-3"),
                            html.Span(id="summary-status-text", className="small text-muted me-3"),
                            html.Button(
                                "Generate Summary",
                                id="summary-gen-btn",
                                n_clicks=0,
                                className="btn btn-sm btn-outline-secondary",
                            ),
                        ],
                        className="d-flex align-items-center flex-wrap gap-2",
                    ),
                ],
                className="px-1 mb-3",
            ),

            html.Hr(),
            html.Div(id="leaderboard-table", className="mt-4"),
            dcc.Store(id="summary-job-store", data={}),
            dcc.Interval(
                id="performance-interval",
                interval=int(max(1.0, LIVE_REFRESH_SECONDS) * 1000),
                n_intervals=0,
            ),
            dcc.Interval(
                id="video-poll-interval",
                interval=2000,
                n_intervals=0,
            ),
        ],
        className="p-3",
    )
