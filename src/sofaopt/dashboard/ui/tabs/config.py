"""Config tab — edit the project's optional config file."""

from dash import dcc, html


def build_config_tab() -> html.Div:
    """Editor for ``project.config_file`` (only shown when one is set)."""
    from sofaopt.dashboard.process.process_manager import load_config_text

    return html.Div(
        [
            html.H3("Project Configuration", className="mb-2"),
            html.P("Edit the project config file. Click Save to write it to disk.", className="text-muted mb-3"),
            dcc.Textarea(
                id="config-textarea",
                value=load_config_text(),
                style={
                    "width": "100%",
                    "height": "520px",
                    "fontFamily": "monospace",
                    "fontSize": "0.88rem",
                    "border": "1px solid #ced4da",
                    "borderRadius": "6px",
                    "padding": "10px",
                },
            ),
            html.Div(
                [
                    html.Button("Save", id="config-save-btn", n_clicks=0, className="btn btn-primary me-3"),
                    html.Span(id="config-save-status", className="align-middle"),
                ],
                className="mt-2 d-flex align-items-center",
            ),
        ],
        className="p-3",
    )
