"""Scenes tab — open a test's scene in an interactive SOFA viewer."""

from dash import dcc, html


def build_scenes_tab(catalog: dict) -> html.Div:
    """Pick a test and launch its scene in a GUI ``runSofa`` window.

    Generic: lists the project's tests; launching opens the scene without
    optimizer env, so the scene falls back to interactive defaults
    (``open_trial().is_optimizing == False``).
    """
    options = [{"label": spec.display_label, "value": name} for name, spec in catalog.items()]
    return html.Div(
        [
            html.H3("Scenes", className="mb-2"),
            html.P("Open a test's scene in an interactive SOFA window for inspection.", className="text-muted mb-3"),
            dcc.Dropdown(
                id="scene-select",
                options=options,
                value=(options[0]["value"] if options else None),
                clearable=False,
                style={"maxWidth": "420px"},
                className="mb-3",
            ),
            html.Button("Launch in viewer", id="scene-launch-btn", n_clicks=0, className="btn btn-primary"),
            html.Div(id="scene-status", className="mt-3 fw-semibold"),
        ],
        className="p-3",
    )
