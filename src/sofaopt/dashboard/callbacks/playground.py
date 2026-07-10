"""Callbacks for the Playground tab.

The transport (play / pause / step / replay / timer) runs clientside so the
controls are instant and the timer never floods the server. A single server
callback then redraws both figures for the current frame; the landscape z is
precomputed in the map store, so each redraw is cheap.
"""

from __future__ import annotations

import random

import plotly.graph_objects as go
from dash import Input, Output, State

from sofaopt.dashboard.playground.objectives import (
    axis_profiles, make_landscape, score_at, score_grid, score_line)
from sofaopt.dashboard.playground.optimizers import ALGO_ORDER, ALGO_PARAMS, run_optimization

# Distinct colours for accumulated runs, cycled by run index.
RUN_COLORS = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]


def _visible_indices(run: dict, frame: int) -> list[int]:
    """Indices to display at a frame: current generation (windowed) or cumulative."""
    n = len(run["xs"])
    if n == 0:
        return []
    f = min(int(frame or 0), n - 1)
    if run.get("windowed"):
        gen_size = max(1, int(run.get("gen_size", 1)))
        lo = (f // gen_size) * gen_size
        return list(range(lo, min(lo + gen_size, n)))
    return list(range(0, f + 1))


def _run_points(run: dict, frame: int):
    """2D points + best-marker for a run at a frame."""
    idxs = _visible_indices(run, frame)
    if not idxs:
        return [], [], [], []
    px = [run["xs"][i] for i in idxs]
    py = [run["ys"][i] for i in idxs]
    best_i = max(idxs, key=lambda i: run["scores"][i])
    return px, py, [run["xs"][best_i]], [run["ys"][best_i]]


def _run_ellipse(run: dict, frame: int):
    """The CMA-ES Gaussian outline for the current generation (empty otherwise)."""
    if not run.get("windowed") or not run.get("ellipses_xy"):
        return [], []
    n = len(run["xs"])
    if n == 0:
        return [], []
    gen_size = max(1, int(run.get("gen_size", 1)))
    gen = min(int(frame or 0), n - 1) // gen_size
    if 0 <= gen < len(run["ellipses_xy"]):
        e = run["ellipses_xy"][gen]
        return e["x"], e["y"]
    return [], []


def _heatmap_figure(map_data: dict, runs: list[dict], path_idx, frame: int) -> go.Figure:
    """Full heatmap. Trace order is fixed so clientside restyle can target it:
    [0] landscape, [1] points, [2] best, [3] CMA-ES Gaussian."""
    fig = go.Figure(
        go.Heatmap(x=map_data["xs"], y=map_data["ys"], z=map_data["z"],
                   colorscale="Viridis", colorbar={"title": "score"})
    )
    px, py, bx, by, ex, ey = [], [], [], [], [], []
    color = RUN_COLORS[0]
    if runs and path_idx is not None and 0 <= path_idx < len(runs):
        px, py, bx, by = _run_points(runs[path_idx], frame)
        ex, ey = _run_ellipse(runs[path_idx], frame)
        color = RUN_COLORS[path_idx % len(RUN_COLORS)]
    fig.add_trace(go.Scatter(
        x=px, y=py, mode="markers", name="points", showlegend=False,
        marker={"color": color, "size": 9, "line": {"color": "white", "width": 1}}))
    fig.add_trace(go.Scatter(
        x=bx, y=by, mode="markers", name="best", showlegend=False,
        marker={"color": "white", "size": 16, "symbol": "star",
                "line": {"color": "black", "width": 1}}))
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines", name="gaussian", showlegend=False,
        line={"color": "white", "width": 2, "dash": "dot"}))
    fig.update_layout(
        title="Landscape & search points", margin={"l": 40, "r": 10, "t": 40, "b": 40},
        xaxis_title="x", yaxis_title="y", height=580, uirevision="pg")
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    return fig


def _line_figure(map_data: dict, runs: list[dict], path_idx, frame: int) -> go.Figure:
    """1D landscape as a curve with the selected run's points sitting on it."""
    fig = go.Figure()
    lx, ly = map_data.get("line_x"), map_data.get("line_y")
    if lx:
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", name="landscape",
                                 line={"color": "#888", "width": 2}, showlegend=False))
    if runs and path_idx is not None and 0 <= path_idx < len(runs):
        run = runs[path_idx]
        idxs = _visible_indices(run, frame)
        color = RUN_COLORS[path_idx % len(RUN_COLORS)]
        fig.add_trace(go.Scatter(
            x=[run["xs"][i] for i in idxs], y=[run["scores"][i] for i in idxs],
            mode="markers", name="points", showlegend=False,
            marker={"color": color, "size": 10, "line": {"color": "white", "width": 1}}))
        if idxs:
            bi = max(idxs, key=lambda i: run["scores"][i])
            fig.add_trace(go.Scatter(
                x=[run["xs"][bi]], y=[run["scores"][bi]], mode="markers", showlegend=False,
                marker={"color": "white", "size": 16, "symbol": "star",
                        "line": {"color": "black", "width": 1}}))
    fig.update_layout(
        title="Landscape (1D)", height=580, xaxis_title="x", yaxis_title="score",
        margin={"l": 50, "r": 10, "t": 40, "b": 40}, uirevision="pg")
    fig.update_xaxes(range=[0, 1])
    return fig


def _scatter3d_figure(map_data: dict, runs: list[dict], path_idx, frame: int) -> go.Figure:
    """3D search: sampled points in the cube, colored by score; optima marked."""
    fig = go.Figure()
    peaks = map_data.get("peaks", [])
    if peaks:
        fig.add_trace(go.Scatter3d(
            x=[p["center"][0] for p in peaks], y=[p["center"][1] for p in peaks],
            z=[p["center"][2] for p in peaks], mode="markers", name="optima",
            marker={"color": "black", "size": 4, "symbol": "diamond"}))
    if runs and path_idx is not None and 0 <= path_idx < len(runs):
        run = runs[path_idx]
        idxs = _visible_indices(run, frame)
        fig.add_trace(go.Scatter3d(
            x=[run["xs"][i] for i in idxs], y=[run["ys"][i] for i in idxs],
            z=[run["zs"][i] for i in idxs], mode="markers", name="points",
            marker={"color": [run["scores"][i] for i in idxs], "colorscale": "Viridis",
                    "size": 5, "cmin": 0, "cmax": 1, "colorbar": {"title": "score"}}))
    fig.update_layout(
        title="Search points (3D, colored by score)", height=580,
        margin={"l": 0, "r": 0, "t": 40, "b": 0}, uirevision="pg",
        scene={"xaxis": {"range": [0, 1], "title": "x0"},
               "yaxis": {"range": [0, 1], "title": "x1"},
               "zaxis": {"range": [0, 1], "title": "x2"}})
    return fig


def _axis_strips_figure(map_data: dict, runs: list[dict], path_idx, frame: int,
                        height: int | None = None) -> go.Figure:
    """Per-axis score strips: one row per dimension, colour = best score reachable
    at each coordinate (others optimal). The selected run's points-so-far are
    overlaid cumulatively so each axis is seen converging, with a white diamond at
    the current position. Optimistic projection — read it against the true score.
    """
    dim = int(map_data.get("dim", 2))
    labels = [f"x{i}" for i in range(dim)]
    axis_xs, axis_z = map_data.get("axis_xs"), map_data.get("axis_z")
    fig = go.Figure()
    if axis_xs and axis_z:
        fig.add_trace(go.Heatmap(
            x=axis_xs, y=labels, z=axis_z, colorscale="Viridis", zmin=0, zmax=1,
            colorbar={"title": "max score"},
            hovertemplate="%{y}<br>value: %{x:.2f}<br>best score here: %{z:.2f}<extra></extra>"))

    if runs and path_idx is not None and 0 <= path_idx < len(runs):
        pts = runs[path_idx].get("points")
        if pts:
            color = RUN_COLORS[path_idx % len(RUN_COLORS)]
            f = min(int(frame or 0), len(pts) - 1)
            mx, my = [], []
            for k in range(0, f + 1):
                for i in range(dim):
                    if i < len(pts[k]):
                        mx.append(pts[k][i])
                        my.append(labels[i])
            fig.add_trace(go.Scatter(
                x=mx, y=my, mode="markers", name="points", showlegend=False,
                marker={"color": color, "size": 7, "opacity": 0.45,
                        "line": {"color": "white", "width": 0.5}}, hoverinfo="skip"))
            last = pts[f]
            fig.add_trace(go.Scatter(
                x=[last[i] for i in range(dim) if i < len(last)],
                y=labels[:len(last)], mode="markers", name="current", showlegend=False,
                marker={"symbol": "diamond", "size": 12, "color": "white",
                        "line": {"color": "#212121", "width": 2}},
                hovertemplate="%{y}<br>current: %{x:.3f}<extra></extra>"))

    fig.update_layout(
        title="Per-axis score profile (best reachable at each coordinate)",
        height=height or max(260, 80 + dim * 46),
        margin={"l": 60, "r": 70, "t": 40, "b": 40}, uirevision="pg",
        xaxis_title="coordinate value (0 → 1)")
    fig.update_xaxes(range=[0, 1], fixedrange=True)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)),
                     fixedrange=True)
    return fig


def _empty_strips_figure() -> go.Figure:
    """Collapsed placeholder for the strip slot when the strips fill the heatmap
    slot instead (the >3D case)."""
    fig = go.Figure()
    fig.update_layout(height=10, margin={"l": 0, "r": 0, "t": 0, "b": 0},
                      xaxis={"visible": False}, yaxis={"visible": False},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _curve_figure(runs: list[dict], frame: int) -> go.Figure:
    """Best-so-far convergence curve per run, drawn up to the current frame."""
    fig = go.Figure()
    f = int(frame or 0)
    for idx, run in enumerate(runs):
        k = min(f + 1, len(run["best"]))
        fig.add_trace(go.Scatter(
            x=list(range(1, k + 1)), y=run["best"][:k], mode="lines",
            name=run["label"], line={"color": RUN_COLORS[idx % len(RUN_COLORS)], "width": 2}))
    fig.update_layout(
        title="Best score so far", margin={"l": 50, "r": 10, "t": 40, "b": 40},
        xaxis_title="evaluation", yaxis_title="best score", height=580, uirevision="pg",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.3})
    return fig


def _read_hp(resolution, n_startup, sigma0, popsize, lr) -> dict:
    return {"resolution": resolution, "n_startup_trials": n_startup,
            "sigma0": sigma0, "popsize": popsize, "learning_rate": lr}


def _run_outputs(runs: list[dict]):
    """Build the path-run dropdown options, selected index, and frame max."""
    options = [{"label": f"#{i + 1} {r['label']}", "value": i} for i, r in enumerate(runs)]
    new_idx = len(runs) - 1
    max_frame = max((len(r["xs"]) for r in runs), default=1) - 1
    return options, new_idx, max_frame


def register_playground_callbacks(app) -> None:  # noqa: C901  # Dash registrar: total is the sum of its small nested callbacks; the flat registration list reads best in one place
    """Register all Playground-tab callbacks."""

    # Show only the hyperparameter rows relevant to the chosen algorithm.
    @app.callback(
        [Output("pg-row-resolution", "style"), Output("pg-row-startup", "style"),
         Output("pg-row-sigma0", "style"), Output("pg-row-lr", "style"),
         Output("pg-row-popsize", "style"), Output("pg-budget", "disabled")],
        Input("pg-algo", "value"),
    )
    def toggle_param_rows(algo):
        params = ALGO_PARAMS.get(algo, ())
        show, hide = {}, {"display": "none"}
        return (
            show if "resolution" in params else hide,
            show if "n_startup_trials" in params else hide,
            show if "sigma0" in params else hide,
            show if "learning_rate" in params else hide,
            show if "popsize" in params else hide,
            algo == "grid",
        )

    # Show random-map controls only in random mode.
    @app.callback(
        [Output("pg-cell-optima", "style"), Output("pg-cell-spread", "style"),
         Output("pg-cell-mapseed", "style")],
        Input("pg-map-mode", "value"),
    )
    def toggle_map_rows(mode):
        hide = {"display": "none"}
        return ({}, {}, {}) if mode == "random" else (hide, hide, hide)

    # Confirm before regenerating (it wipes the current map's runs).
    @app.callback(
        Output("pg-confirm-regen", "displayed"),
        Input("pg-regen-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def ask_regen(_n):
        return True

    # Regenerate the map (fresh random seed each time) and clear all runs.
    @app.callback(
        [Output("pg-map-store", "data"), Output("pg-runs-store", "data", allow_duplicate=True),
         Output("pg-map-seed", "value"),
         Output("pg-frame", "value", allow_duplicate=True), Output("pg-playing", "data", allow_duplicate=True)],
        Input("pg-confirm-regen", "submit_n_clicks"),
        [State("pg-map-mode", "value"), State("pg-n-optima", "value"),
         State("pg-height-spread", "value"), State("pg-dim", "value")],
        prevent_initial_call=True,
    )
    def regen_map(_submit, mode, n_optima, spread, dim):
        seed = random.randrange(1_000_000)  # noqa: S311  # toy-landscape seed, not security
        dim = max(1, int(dim or 2))
        peaks = make_landscape(mode, int(n_optima or 1), float(spread or 0.0),
                               seed=seed, dim=dim)
        map_data = {"peaks": peaks, "dim": dim}
        map_data["axis_xs"], map_data["axis_z"] = axis_profiles(peaks, dim)
        if dim == 1:
            map_data["line_x"], map_data["line_y"] = score_line(peaks)
        elif dim == 2:
            map_data["xs"], map_data["ys"], map_data["z"] = score_grid(peaks)
        return map_data, [], seed, 0, False

    # Run the selected algorithm, append it, and auto-play.
    @app.callback(
        [Output("pg-runs-store", "data", allow_duplicate=True),
         Output("pg-path-run", "options"), Output("pg-path-run", "value"),
         Output("pg-frame", "max"), Output("pg-frame", "value"),
         Output("pg-playing", "data", allow_duplicate=True)],
        Input("pg-run-btn", "n_clicks"),
        [State("pg-algo", "value"), State("pg-budget", "value"), State("pg-seed", "value"),
         State("pg-resolution", "value"), State("pg-n-startup", "value"),
         State("pg-sigma0", "value"), State("pg-popsize", "value"), State("pg-lr", "value"),
         State("pg-map-store", "data"), State("pg-runs-store", "data")],
        prevent_initial_call=True,
    )
    def do_run(_n, algo, budget, seed, resolution, n_startup, sigma0, popsize, lr, map_data, runs):
        peaks = map_data["peaks"]
        dim = int(map_data.get("dim", 2))
        result = run_optimization(algo, _read_hp(resolution, n_startup, sigma0, popsize, lr),
                                  lambda p: score_at(peaks, p), int(budget or 60),
                                  int(seed or 0), dim=dim)
        runs = (runs or []) + [result]
        options, new_idx, max_frame = _run_outputs(runs)
        return runs, options, new_idx, max_frame, 0, True

    # Run every algorithm on the current map for a side-by-side comparison.
    @app.callback(
        [Output("pg-runs-store", "data", allow_duplicate=True),
         Output("pg-path-run", "options", allow_duplicate=True), Output("pg-path-run", "value", allow_duplicate=True),
         Output("pg-frame", "max", allow_duplicate=True), Output("pg-frame", "value", allow_duplicate=True),
         Output("pg-playing", "data", allow_duplicate=True)],
        Input("pg-run-all-btn", "n_clicks"),
        [State("pg-budget", "value"), State("pg-seed", "value"),
         State("pg-resolution", "value"), State("pg-n-startup", "value"),
         State("pg-sigma0", "value"), State("pg-popsize", "value"), State("pg-lr", "value"),
         State("pg-map-store", "data")],
        prevent_initial_call=True,
    )
    def do_run_all(_n, budget, seed, resolution, n_startup, sigma0, popsize, lr, map_data):
        peaks = map_data["peaks"]
        dim = int(map_data.get("dim", 2))
        hp = _read_hp(resolution, n_startup, sigma0, popsize, lr)
        def fn(p):
            return score_at(peaks, p)
        runs = [run_optimization(algo, hp, fn, int(budget or 60), int(seed or 0), dim=dim)
                for algo in ALGO_ORDER]
        options, new_idx, max_frame = _run_outputs(runs)
        return runs, options, new_idx, max_frame, 0, True

    # Clear all runs and reset the transport.
    @app.callback(
        [Output("pg-runs-store", "data", allow_duplicate=True),
         Output("pg-path-run", "options", allow_duplicate=True),
         Output("pg-path-run", "value", allow_duplicate=True),
         Output("pg-frame", "value", allow_duplicate=True),
         Output("pg-frame", "max", allow_duplicate=True),
         Output("pg-playing", "data", allow_duplicate=True)],
        Input("pg-clear-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_runs(_n):
        return [], [], None, 0, 0, False

    # Transport (clientside): play / pause / step / replay / timer -> [frame, playing].
    # Clientside so the controls are instant and the timer never floods the server.
    app.clientside_callback(
        """
        function(nTick, playN, pauseN, stepN, replayN, frame, maxFrame, playing) {
            const c = window.dash_clientside.callback_context;
            const trig = (c.triggered[0] || {}).prop_id || "";
            const NU = window.dash_clientside.no_update;
            frame = frame || 0; maxFrame = maxFrame || 0; playing = !!playing;
            if (trig.indexOf("pg-play-btn") === 0) {
                if (maxFrame <= 0) return [NU, false];
                return [frame >= maxFrame ? 0 : frame, true];
            }
            if (trig.indexOf("pg-pause-btn") === 0) return [NU, false];
            if (trig.indexOf("pg-step-btn") === 0) return [Math.min(frame + 1, maxFrame), false];
            if (trig.indexOf("pg-replay-btn") === 0) {
                if (maxFrame <= 0) return [NU, NU];
                return [0, true];
            }
            if (!playing) return [NU, NU];                       // timer tick, paused
            if (frame >= maxFrame) return [maxFrame, false];     // reached the end
            return [frame + 1, NU];
        }
        """,
        [Output("pg-frame", "value", allow_duplicate=True), Output("pg-playing", "data")],
        [Input("pg-anim-timer", "n_intervals"), Input("pg-play-btn", "n_clicks"),
         Input("pg-pause-btn", "n_clicks"), Input("pg-step-btn", "n_clicks"),
         Input("pg-replay-btn", "n_clicks")],
        [State("pg-frame", "value"), State("pg-frame", "max"), State("pg-playing", "data")],
        prevent_initial_call=True,
    )

    # Server: redraw both figures for the current frame and selection. The
    # landscape z is precomputed in the map store, so each redraw only rebuilds
    # the figure scaffolding plus the small point overlays.
    @app.callback(
        [Output("pg-heatmap", "figure"), Output("pg-curve", "figure"),
         Output("pg-strips", "figure")],
        [Input("pg-frame", "value"), Input("pg-runs-store", "data"),
         Input("pg-path-run", "value"), Input("pg-map-store", "data")],
    )
    def draw(frame, runs, path_idx, map_data):
        runs = runs or []
        f = int(frame or 0)
        dim = int(map_data.get("dim", 2))
        if dim == 1:
            heat = _line_figure(map_data, runs, path_idx, f)
        elif dim == 2 and map_data.get("z"):
            heat = _heatmap_figure(map_data, runs, path_idx, f)
        elif dim == 3:
            heat = _scatter3d_figure(map_data, runs, path_idx, f)
        else:
            # No drawable landscape past 3D — the strips stand in for it.
            heat = _axis_strips_figure(map_data, runs, path_idx, f, height=580)
        # Strips sit under the heatmap for ≤3D; for >3D they already fill the
        # heatmap slot, so collapse the companion to avoid drawing them twice.
        strips = (_axis_strips_figure(map_data, runs, path_idx, f)
                  if dim <= 3 else _empty_strips_figure())
        return heat, _curve_figure(runs, f), strips
