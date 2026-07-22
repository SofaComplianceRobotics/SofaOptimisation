"""Playground tab — interactive optimizer teaching playground.

Layout only (no callbacks). Controls sit in compact bars above two large graphs:
a landscape heatmap with the animated search path, and a panel that accumulates
best-so-far convergence curves. Wired up in dashboard/callbacks/playground.py.
"""

from dash import dcc, html

from .objectives import axis_profiles, make_landscape, score_grid
from .optimizers import ALGO_LABELS

ANIM_INTERVAL_MS = 160  # playback tick; server redraws each frame, so keep it relaxed


def _initial_map() -> dict:
    """Default landscape: a single centred global optimum (2D)."""
    peaks = make_landscape("default", 1, 0.0, seed=0, dim=2)
    xs, ys, z = score_grid(peaks)
    axis_xs, axis_z = axis_profiles(peaks, 2)
    return {"peaks": peaks, "dim": 2, "xs": xs, "ys": ys, "z": z,
            "axis_xs": axis_xs, "axis_z": axis_z}


def _field(label: str, component, width: str = "150px") -> html.Div:
    """A compact label-over-control cell for a horizontal control bar."""
    return html.Div(
        [html.Label(label, style={"fontSize": "0.75rem", "fontWeight": 600,
                                   "display": "block", "marginBottom": "0.1rem"}),
         component],
        style={"width": width, "marginRight": "0.9rem", "marginBottom": "0.4rem"},
    )


def _num(component_id: str, value, **kw):
    return dcc.Input(id=component_id, type="number", value=value, debounce=True,
                     style={"width": "100%"}, **kw)


def build_playground_tab() -> html.Div:
    """Build the Playground tab layout."""
    initial = _initial_map()
    return html.Div(
        [
            dcc.Store(id="pg-map-store", data=initial),
            dcc.Store(id="pg-runs-store", data=[]),
            dcc.Store(id="pg-playing", data=False),
            dcc.Interval(id="pg-anim-timer", interval=ANIM_INTERVAL_MS, disabled=False),
            dcc.ConfirmDialog(
                id="pg-confirm-regen",
                message="Generate a new map? This erases all runs on the current map.",
            ),
            html.H3("Optimizer Playground", className="mb-1"),
            html.P(
                "Pick an algorithm, tune it, and watch it search the landscape. "
                "Curves accumulate so you can compare runs on the same map.",
                style={"color": "#555", "marginBottom": "0.5rem"},
            ),

            # ── Control bar 1: algorithm + knobs + run ───────────────
            html.Div(
                [
                    _field("Algorithm", dcc.Dropdown(
                        id="pg-algo",
                        options=[{"label": v, "value": k} for k, v in ALGO_LABELS.items()],
                        value="cmaes", clearable=False), width="180px"),
                    _field("Budget", _num("pg-budget", 60, min=4, max=400, step=1), width="100px"),
                    _field("Seed", _num("pg-seed", 0, min=0, step=1), width="90px"),
                    html.Div(_field("Grid pts/axis", _num("pg-resolution", 8, min=2, max=20, step=1),
                                    width="110px"), id="pg-row-resolution"),
                    html.Div(_field("Warm-up", _num("pg-n-startup", 5, min=1, max=50, step=1),
                                    width="100px"), id="pg-row-startup"),
                    html.Div(_field("σ₀ step", _num("pg-sigma0", 0.2, min=0.01, max=1.0, step=0.01),
                                    width="100px"), id="pg-row-sigma0"),
                    html.Div(_field("Learn rate", _num("pg-lr", 0.1, min=0.01, max=1.0, step=0.01),
                                    width="100px"), id="pg-row-lr"),
                    html.Div(_field("Pop size", _num("pg-popsize", 6, min=3, max=30, step=1),
                                    width="100px"), id="pg-row-popsize"),
                    html.Div(
                        [html.Button("▶ Run", id="pg-run-btn", n_clicks=0,
                                     className="btn btn-primary btn-sm", style={"marginRight": "0.35rem"}),
                         html.Button("▶▶ Run all", id="pg-run-all-btn", n_clicks=0,
                                     className="btn btn-success btn-sm", style={"marginRight": "0.35rem"}),
                         html.Button("Clear runs", id="pg-clear-btn", n_clicks=0,
                                     className="btn btn-outline-secondary btn-sm")],
                        style={"alignSelf": "flex-end", "marginBottom": "0.4rem"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-start",
                       "padding": "0.5rem", "background": "#f6f7f9", "borderRadius": "6px"},
            ),

            # ── Control bar 2: landscape ─────────────────────────────
            html.Div(
                [
                    _field("Map mode", dcc.RadioItems(
                        id="pg-map-mode",
                        options=[{"label": " Default", "value": "default"},
                                 {"label": " Ridge ↗", "value": "ridge"},
                                 {"label": " Random", "value": "random"}],
                        value="default", labelStyle={"marginRight": "0.6rem"}), width="210px"),
                    _field("Dimensions", _num("pg-dim", 2, min=1, max=10, step=1), width="100px"),
                    html.Div(_field("Local optima", dcc.Slider(
                        id="pg-n-optima", min=1, max=8, step=1, value=4,
                        marks={i: str(i) for i in range(1, 9)}), width="200px"),
                        id="pg-cell-optima"),
                    html.Div(_field("Height spread (low=deceptive · high=easy)", dcc.Slider(
                        id="pg-height-spread", min=0.0, max=1.0, step=0.05, value=0.5,
                        marks={0: "0", 0.5: "0.5", 1: "1"}), width="260px"),
                        id="pg-cell-spread"),
                    html.Div(_field("Map seed (auto)", _num("pg-map-seed", 0, min=0, step=1),
                                    width="110px"), id="pg-cell-mapseed"),
                    html.Div(
                        html.Button("⟳ Generate map", id="pg-regen-btn", n_clicks=0,
                                    className="btn btn-outline-secondary btn-sm"),
                        style={"alignSelf": "flex-end", "marginBottom": "0.4rem"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-start",
                       "padding": "0.5rem", "marginTop": "0.4rem",
                       "background": "#f6f7f9", "borderRadius": "6px"},
            ),

            # ── The two main graphs ──────────────────────────────────
            html.Div(
                [
                    dcc.Graph(id="pg-heatmap", style={"flex": "1 1 0", "minWidth": 0, "height": "600px"},
                              config={"displayModeBar": False}),
                    dcc.Graph(id="pg-curve", style={"flex": "1 1 0", "minWidth": 0, "height": "600px"},
                              config={"displayModeBar": False}),
                ],
                style={"display": "flex", "gap": "0.5rem", "width": "100%", "marginTop": "0.6rem"},
            ),

            # ── Per-axis score strips ────────────────────────────────
            # One row per dimension: colour is the best score reachable at each
            # coordinate (others optimal), with the run's visited points overlaid
            # so you watch each axis converge. Sits under the heatmap for ≤3D and
            # stands in for it past 3D, where the landscape can't be drawn.
            dcc.Graph(id="pg-strips", style={"width": "100%", "marginTop": "0.4rem"},
                      config={"displayModeBar": False}),

            # ── Playback transport ───────────────────────────────────
            html.Div(
                [
                    html.Button("▶ Play", id="pg-play-btn", n_clicks=0,
                                className="btn btn-sm btn-outline-secondary"),
                    html.Button("⏸ Pause", id="pg-pause-btn", n_clicks=0,
                                className="btn btn-sm btn-outline-secondary", style={"marginLeft": "0.35rem"}),
                    html.Button("⏭ Step", id="pg-step-btn", n_clicks=0,
                                className="btn btn-sm btn-outline-secondary", style={"marginLeft": "0.35rem"}),
                    html.Button("⟲ Replay", id="pg-replay-btn", n_clicks=0,
                                className="btn btn-sm btn-outline-secondary", style={"marginLeft": "0.35rem"}),
                    html.Span("Show path for:", style={"marginLeft": "1.2rem", "marginRight": "0.35rem"}),
                    dcc.Dropdown(id="pg-path-run", options=[], value=None, clearable=False,
                                 style={"display": "inline-block", "width": "280px", "verticalAlign": "middle"}),
                ],
                style={"display": "flex", "alignItems": "center", "marginTop": "0.4rem"},
            ),
            dcc.Slider(id="pg-frame", min=0, max=0, step=1, value=0,
                       updatemode="drag", tooltip={"placement": "bottom"}),
        ],
        className="p-3",
    )
