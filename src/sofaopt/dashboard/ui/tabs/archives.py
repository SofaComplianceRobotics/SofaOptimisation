"""Archives tab: archive the current run, manage archives, compare runs."""

from dash import dcc, html


def build_archives_tab() -> html.Div:
    return html.Div(
        [
            html.H3("Archives", className="mb-1"),
            html.P(
                "Archiving moves the current runtime/ into work_dir/archives/ "
                "(instant, no copy) and resets the workspace. Starting a fresh "
                "run auto-archives the previous one.",
                className="text-muted mb-3",
            ),

            # -- archive the current run -------------------------------------
            html.Div(
                [
                    dcc.Input(
                        id="archive-name-input",
                        type="text",
                        placeholder="archive name (optional)",
                        className="form-control form-control-sm me-2",
                        style={"maxWidth": "240px"},
                    ),
                    dcc.Input(
                        id="archive-notes-input",
                        type="text",
                        placeholder="notes (optional)",
                        className="form-control form-control-sm me-2",
                        style={"maxWidth": "360px"},
                    ),
                    html.Button(
                        "Stop & archive current run",
                        id="archive-now-btn",
                        n_clicks=0,
                        title="Stops the run first if one is active (clean pause), then moves runtime/ into archives/",
                        className="btn btn-sm btn-primary",
                    ),
                    html.Span(id="archive-action-status", className="ms-3 small"),
                ],
                className="d-flex align-items-center mb-4",
            ),

            # -- archive list --------------------------------------------------
            html.Div(id="archives-table", className="mb-2"),

            # -- per-archive search-space report (on demand) -------------------
            html.Div(id="archive-report-panel", className="mb-4"),

            # -- comparison ----------------------------------------------------
            html.H5("Compare runs", className="mb-2"),
            html.Div(
                [
                    dcc.Checklist(
                        id="archive-compare-select",
                        options=[],
                        value=[],
                        inline=True,
                        className="me-3",
                        inputClassName="form-check-input me-1",
                        labelClassName="form-check-label me-3",
                    ),
                    html.Button(
                        "Compare",
                        id="archive-compare-btn",
                        n_clicks=0,
                        className="btn btn-sm btn-outline-primary",
                    ),
                ],
                className="d-flex align-items-center flex-wrap mb-2",
            ),
            dcc.Graph(id="archive-compare-graph", figure={}),
            html.Div(id="archive-compare-summary", className="mt-3"),

            # refresh driver + confirm dialogs
            dcc.Store(id="archives-dirty", data=0),
            dcc.Interval(id="archives-interval", interval=5000, n_intervals=0),
            dcc.ConfirmDialog(
                id="archive-delete-confirm",
                message="Permanently delete this archive? This cannot be undone.",
            ),
            dcc.Store(id="archive-delete-target", data=""),
        ],
        className="p-3",
    )
