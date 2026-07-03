"""Optimise tab — test selection, weight management, sampler choice, Run/Stop."""

from dash import dcc, html

from sofaopt.dashboard import context
from .styles import LOG_STYLE

PIE_PALETTE = [
    "#4c8bf5",
    "#e84393",
    "#34a853",
    "#fa7b17",
    "#9c27b0",
    "#00bcd4",
    "#ff5722",
    "#8bc34a",
]


def _equal_split(n: int) -> list[int]:
    """Split 100 into ``n`` integer parts as evenly as possible."""
    if n == 0:
        return []
    base = 100 // n
    rem = 100 - base * n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _build_sampler_controls() -> html.Div:
    """Optimizer-settings row: sampler, initial design, margin, parallelism.

    Defaults are read from the project; the values are forwarded as ``OPT_*``
    env overrides when the Run button launches the headless study, so the UI
    drives the same knobs as the ``run.py`` CLI flags.
    """
    project = context.project()
    return html.Div(
        [
            html.H5("Optimizer", className="mb-2"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Sampler", className="form-label mb-1 small text-muted"),
                            dcc.Dropdown(
                                id="opt-sampler",
                                options=[
                                    {"label": "CMA-ES (covariance, coupled params)", "value": "cmaes"},
                                    {"label": "GP-BO (sample-efficient, expensive sims)", "value": "gp"},
                                    {"label": "TPE (Bayesian)", "value": "tpe"},
                                    {"label": "Random (baseline)", "value": "random"},
                                ],
                                value=project.sampler,
                                clearable=False,
                            ),
                        ],
                        className="col-12 col-md-4",
                    ),
                    html.Div(
                        [
                            html.Label("Initial design", className="form-label mb-1 small text-muted"),
                            dcc.Dropdown(
                                id="opt-seed-sampler",
                                options=[
                                    {"label": "Sobol' (space-filling DOE)", "value": "sobol"},
                                    {"label": "Random", "value": "random"},
                                ],
                                value=project.seed_sampler,
                                clearable=False,
                            ),
                        ],
                        className="col-12 col-md-3",
                    ),
                    html.Div(
                        [
                            html.Label("Parallel / Gens", className="form-label mb-1 small text-muted"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="opt-n-parallel", type="number", min=1, step=1,
                                        value=project.n_parallel, className="form-control form-control-sm",
                                        style={"width": "80px"},
                                    ),
                                    dcc.Input(
                                        id="opt-n-generations", type="number", min=1, step=1,
                                        value=project.n_generations, className="form-control form-control-sm ms-2",
                                        style={"width": "80px"},
                                    ),
                                ],
                                className="d-flex",
                            ),
                        ],
                        className="col-12 col-md-3",
                    ),
                    html.Div(
                        dcc.Checklist(
                            id="opt-cmaes-margin",
                            options=[{"label": " CMA-ES with Margin", "value": "margin"}],
                            value=["margin"] if project.cmaes_with_margin else [],
                            className="mt-4",
                        ),
                        className="col-12 col-md-2",
                    ),
                ],
                className="row g-2 align-items-start",
            ),
            html.Small(
                "GP-BO is most sample-efficient with a small batch — try Parallel ≈ 4. "
                "Margin only applies to CMA-ES and fixes low-cardinality integer stagnation.",
                className="text-muted",
            ),
        ],
        className="mb-3 p-3 border rounded bg-light",
    )


def build_optimise_tab(catalog: dict) -> html.Div:
    """Build the Optimise tab from the project's test catalog."""
    names = list(catalog.keys())
    any_default = any(spec.default_selected for spec in catalog.values())

    selected_names = [
        name for name, spec in catalog.items() if spec.default_selected or not any_default
    ]
    weights = _equal_split(len(selected_names))
    initial_store: dict[str, int] = {}
    wi = 0
    for name in names:
        if name in selected_names:
            initial_store[name] = weights[wi]
            wi += 1
        else:
            initial_store[name] = 0

    test_rows = []
    for name, spec in catalog.items():
        pre_selected = spec.default_selected or not any_default
        test_rows.append(
            html.Div(
                [
                    dcc.Checklist(
                        id={"type": "test-check", "test": name},
                        options=[{"label": f" {spec.label}", "value": name}],
                        value=[name] if pre_selected else [],
                        style={"display": "inline-flex", "alignItems": "center", "minWidth": "200px", "flexShrink": 0},
                        className="me-2",
                    ),
                    dcc.Checklist(
                        id={"type": "gate-check", "test": name},
                        options=[{"label": " Gate", "value": name}],
                        value=[],
                        style={"display": "inline-flex", "alignItems": "center", "minWidth": "90px", "flexShrink": 0},
                        className="me-2 text-muted",
                    ),
                    html.Div(
                        dcc.Slider(
                            id={"type": "weight-slider", "test": name},
                            min=0,
                            max=100,
                            step=1,
                            value=initial_store[name],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": True},
                            updatemode="drag",
                        ),
                        style={"flexGrow": 1},
                    ),
                ],
                className="d-flex align-items-center mb-3",
                style={"gap": "8px"},
            )
        )

    return html.Div(
        [
            html.H3("Optimisation", className="mb-2"),
            _build_sampler_controls(),
            dcc.Store(id="opt-weights-store", data=initial_store),
            html.Div(
                [
                    html.Div(
                        [
                            html.P("Drag a slider — the others adjust so the total stays at 100%.", className="text-muted mb-3"),
                            html.P("Use Gate to delay a test until one of the ungated tests succeeds.", className="text-muted mb-3"),
                            html.Div(test_rows, className="mb-2"),
                            html.Div(
                                [
                                    html.Button("Equal split", id="opt-equal-btn", n_clicks=0, className="btn btn-outline-secondary btn-sm me-2"),
                                    html.Button("Normalize", id="opt-normalize-btn", n_clicks=0, className="btn btn-outline-secondary btn-sm"),
                                ],
                                className="mb-3",
                            ),
                        ],
                        className="col-12 col-md-7",
                    ),
                    html.Div(
                        dcc.Graph(id="opt-pie", config={"displayModeBar": False}, style={"height": "320px"}),
                        className="col-12 col-md-5",
                    ),
                ],
                className="row g-3 mb-3",
            ),
            html.Div(id="opt-weight-status", className="mb-3"),
            html.Div(
                [
                    html.Button("Start Optimisation", id="opt-start-btn", n_clicks=0, className="btn btn-success me-2"),
                    html.Button("Pause", id="opt-stop-btn", n_clicks=0, disabled=True, className="btn btn-danger"),
                ],
                className="mb-3",
            ),
            html.Div(id="opt-status", className="mb-2 fw-semibold"),
            html.Pre(id="opt-log", style=LOG_STYLE),
            dcc.Interval(id="opt-interval", interval=1000, n_intervals=0),
        ],
        className="p-3",
    )
