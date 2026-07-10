"""Pareto front scatter plots for multi-objective optimization."""

from __future__ import annotations

import itertools
from typing import Sequence

import plotly.graph_objects as go
from dash import html


def _is_dominated(values: list[float], directions: list[str], candidates: list[list[float]]) -> bool:
    """Return True if ``values`` is dominated by any candidate."""
    for other in candidates:
        if other is values:
            continue
        dominated = True
        for v, o, d in zip(values, other, directions, strict=True):
            if d == "maximize":
                if v > o:
                    dominated = False
                    break
            else:
                if v < o:
                    dominated = False
                    break
        if dominated and other != values:
            return True
    return False


def _pareto_mask(
    all_values: list[list[float]], directions: list[str]
) -> list[bool]:
    """Return a boolean mask: True if the corresponding trial is non-dominated."""
    result = []
    for values in all_values:
        dominated = False
        for other in all_values:
            if other is values:
                continue
            # 'other' dominates 'values' if other is at least as good in all objectives
            # and strictly better in at least one.
            at_least_as_good = all(
                (o >= v if d == "maximize" else o <= v)
                for v, o, d in zip(values, other, directions, strict=True)
            )
            strictly_better = any(
                (o > v if d == "maximize" else o < v)
                for v, o, d in zip(values, other, directions, strict=True)
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        result.append(not dominated)
    return result


def build_pareto_figures(
    records: list[dict],
    test_names: Sequence[str],
    directions: Sequence[str],
) -> list[go.Figure]:
    """Build one 2-D scatter per pair of objectives.

    Args:
        records: Trial records from ``_load_data()`` (must be complete/done).
        test_names: Ordered objective names matching ``directions``.
        directions: ``"maximize"`` or ``"minimize"`` per objective.

    Returns:
        List of Plotly figures, one per pair (or a single figure for 2 objectives).
    """
    test_names = list(test_names)
    directions = list(directions)
    n = len(test_names)
    if n < 2:
        return []

    # Extract objective values from trial records.
    trial_data: list[dict] = []
    for r in records:
        state = r.get("state", "")
        if str(state).lower() not in ("done",):
            continue
        obj = r.get("objective_values")
        if not isinstance(obj, list) or len(obj) < n:
            ts = r.get("test_scores", {})
            obj = [
                ts.get(name, {}).get("aggregate_score", None)
                for name in test_names
            ]
        if any(v is None for v in obj):
            continue
        trial_data.append({
            "trial": r.get("trial_index", "?"),
            "gen": r.get("gen_index", 0),
            "values": [float(v) for v in obj[:n]],
            "params": r.get("params", {}),
        })

    if not trial_data:
        return []

    all_values = [d["values"] for d in trial_data]
    is_optimal = _pareto_mask(all_values, directions)
    gen_max = max((d["gen"] for d in trial_data), default=1) or 1

    pairs = list(itertools.combinations(range(n), 2))
    figures = []
    for ix, iy in pairs:
        x_name = test_names[ix]
        y_name = test_names[iy]
        x_dir = directions[ix]
        y_dir = directions[iy]

        dominated_x = [d["values"][ix] for d, opt in zip(trial_data, is_optimal, strict=True) if not opt]
        dominated_y = [d["values"][iy] for d, opt in zip(trial_data, is_optimal, strict=True) if not opt]
        dominated_gen = [d["gen"] for d, opt in zip(trial_data, is_optimal, strict=True) if not opt]
        dominated_text = [
            f"Trial {d['trial']} (gen {d['gen']})<br>"
            + "<br>".join(f"{k}: {v}" for k, v in d.get("params", {}).items())
            for d, opt in zip(trial_data, is_optimal, strict=True) if not opt
        ]

        optimal_x = [d["values"][ix] for d, opt in zip(trial_data, is_optimal, strict=True) if opt]
        optimal_y = [d["values"][iy] for d, opt in zip(trial_data, is_optimal, strict=True) if opt]
        optimal_gen = [d["gen"] for d, opt in zip(trial_data, is_optimal, strict=True) if opt]
        optimal_text = [
            f"Trial {d['trial']} (gen {d['gen']})<br>"
            + "<br>".join(f"{k}: {v}" for k, v in d.get("params", {}).items())
            for d, opt in zip(trial_data, is_optimal, strict=True) if opt
        ]

        fig = go.Figure()
        if dominated_x:
            fig.add_trace(go.Scatter(
                x=dominated_x, y=dominated_y,
                mode="markers",
                name="Dominated",
                text=dominated_text,
                hoverinfo="text",
                marker=dict(
                    color=dominated_gen,
                    colorscale="Blues",
                    cmin=1, cmax=gen_max,
                    size=7, opacity=0.5,
                    colorbar=dict(title="Gen", x=-0.15),
                ),
            ))
        if optimal_x:
            fig.add_trace(go.Scatter(
                x=optimal_x, y=optimal_y,
                mode="markers",
                name="Pareto-optimal",
                text=optimal_text,
                hoverinfo="text",
                marker=dict(
                    color=optimal_gen,
                    colorscale="Reds",
                    cmin=1, cmax=gen_max,
                    size=12, symbol="star",
                    line=dict(width=1, color="black"),
                ),
            ))

        x_arrow = "↑ better" if x_dir == "maximize" else "↓ better"
        y_arrow = "↑ better" if y_dir == "maximize" else "↓ better"
        fig.update_layout(
            title=f"{x_name} vs {y_name}",
            xaxis_title=f"{x_name}  ({x_arrow})",
            yaxis_title=f"{y_name}  ({y_arrow})",
            height=500,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=60, r=20, t=50, b=60),
        )
        figures.append(fig)

    return figures


def build_pareto_layout(records: list[dict], test_names: list[str], directions: list[str]) -> html.Div:
    """Render all Pareto scatter plots into a Div."""
    from dash import dcc as _dcc

    figs = build_pareto_figures(records, test_names, directions)
    if not figs:
        return html.Div(
            "No completed multi-objective trials yet.",
            className="text-muted mt-4",
        )
    return html.Div([
        _dcc.Graph(figure=fig, style={"height": "520px", "marginBottom": "24px"})
        for fig in figs
    ])
