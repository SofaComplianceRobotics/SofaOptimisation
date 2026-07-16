"""Search-space convergence report: where the population started, where it
ended up, and what drove the score. Project-agnostic — works off the generic
trial records (``params``, ``final_score``, ``gen_index``) and the project's
own :class:`~sofaopt.project.ParamSpec` list, no project-specific parsing.

User-triggered from a dashboard button (not auto-refreshed like the health
panel): building this report walks every trial and renders several figures,
which is too heavy to recompute on every polling tick.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sofaopt.dashboard import context

from .colors import C_AVG, C_BEST, C_BG

_MAX_PARAMS_SHOWN = 9  # keep the funnel grid readable; rare to have more free dims


def _active_specs() -> list[dict]:
    """Searchable (non-frozen) param specs as plain dicts (name/min/max)."""
    return [p.to_dict() for p in context.project().params if not p.is_frozen]


def specs_from_snapshot(snapshot: dict) -> list[dict]:
    """Searchable param specs (name/min/max) from an archived project snapshot.

    A snapshot stores each ParamSpec's raw fields (``low``/``high``, not
    ``min``/``max``); this converts to the report's expected shape and drops
    frozen params (``low == high``) — so the report works off an archive's own
    parameter space, not the currently-loaded project's.
    """
    out = []
    for p in snapshot.get("params", []):
        low, high = p.get("low"), p.get("high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low != high:
            out.append({"name": p.get("name"), "type": p.get("type"), "min": low, "max": high})
    return out


def _usable_records(records: list[dict]) -> list[dict]:
    return [
        r for r in records
        if not r.get("failed") and r.get("final_score") is not None and r.get("gen_index") is not None
    ]


def _gen_series(records: list[dict]):
    """Per-generation (gens, best, mean, std) sorted by generation index."""
    by_gen: dict[int, list[float]] = {}
    for r in records:
        by_gen.setdefault(r["gen_index"], []).append(r["final_score"])
    gens = sorted(by_gen)
    best = [max(by_gen[g]) for g in gens]
    mean = [sum(by_gen[g]) / len(by_gen[g]) for g in gens]
    std = [
        (sum((v - m) ** 2 for v in by_gen[g]) / len(by_gen[g])) ** 0.5
        for g, m in zip(gens, mean, strict=True)
    ]
    return gens, best, mean, std


def _running_best(best: list[float]) -> list[float]:
    out, cur = [], float("-inf")
    for b in best:
        cur = max(cur, b)
        out.append(cur)
    return out


def _convergence_figure(records: list[dict]) -> go.Figure:
    gens, best, mean, std = _gen_series(records)
    running = _running_best(best)
    upper = [m + s for m, s in zip(mean, std, strict=True)]
    lower = [m - s for m, s in zip(mean, std, strict=True)]
    fig = go.Figure([
        go.Scatter(x=gens + gens[::-1], y=upper + lower[::-1], fill="toself",
                   fillcolor="rgba(76,114,176,0.15)", line=dict(width=0),
                   name="pop. std", hoverinfo="skip"),
        go.Scatter(x=gens, y=mean, mode="lines", name="gen mean",
                   line=dict(color="#4C72B0", width=2)),
        go.Scatter(x=gens, y=best, mode="lines", name="gen best",
                   line=dict(color=C_AVG, width=1), opacity=0.7),
        go.Scatter(x=gens, y=running, mode="lines", name="best-so-far",
                   line=dict(color=C_BEST, width=2.5)),
    ])
    fig.update_layout(
        title="Convergence — score vs generation",
        xaxis_title="generation", yaxis_title="score",
        hovermode="x unified", height=380,
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        margin={"l": 50, "r": 30, "t": 50, "b": 40},
    )
    return fig


def _add_funnel_subplot(
    fig: go.Figure, row: int, col: int, spec: dict, records: list[dict],
    srange: tuple[float, float], best: dict | None, show_colorbar: bool,
) -> None:
    name = spec["name"]
    numeric = [r for r in records if isinstance(r["params"].get(name), (int, float))]
    smin, smax = srange
    fig.add_trace(
        go.Scatter(
            x=[r["gen_index"] for r in numeric],
            y=[r["params"][name] for r in numeric],
            mode="markers",
            marker=dict(size=5, color=[r["final_score"] for r in numeric], colorscale="Viridis",
                       cmin=smin, cmax=smax, showscale=show_colorbar,
                       colorbar=dict(title="score") if show_colorbar else None, opacity=0.55),
            showlegend=False,
            hovertemplate=f"{name}=%{{y}}<br>gen=%{{x}}<br>score=%{{marker.color:.2f}}<extra></extra>",
        ),
        row=row, col=col,
    )
    if best is not None and isinstance(best["params"].get(name), (int, float)):
        fig.add_trace(
            go.Scatter(
                x=[best["gen_index"]], y=[best["params"][name]], mode="markers",
                marker=dict(symbol="star", size=14, color=C_BEST, line=dict(width=1, color="black")),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=col,
        )
    fig.update_yaxes(range=[spec["min"], spec["max"]], row=row, col=col)


def _funnel_figure(records: list[dict], specs: list[dict]) -> go.Figure:
    specs = specs[:_MAX_PARAMS_SHOWN]
    n = len(specs)
    cols = min(3, n) or 1
    rows = math.ceil(n / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[s["name"] for s in specs])

    scores = [r["final_score"] for r in records]
    srange = (min(scores), max(scores)) if scores else (0.0, 1.0)
    best = max(records, key=lambda r: r["final_score"]) if records else None

    for i, spec in enumerate(specs):
        _add_funnel_subplot(fig, i // cols + 1, i % cols + 1, spec, records, srange, best, show_colorbar=(i == 0))

    fig.update_layout(
        title="Per-parameter convergence funnel (★ = best trial)",
        height=280 * rows, plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        margin={"l": 40, "r": 40, "t": 60, "b": 30},
    )
    return fig


def _pearson(vals: list[float], scores: list[float], mean_s: float, var_s: float) -> float:
    n = len(vals)
    mean_p = sum(vals) / n
    var_p = sum((v - mean_p) ** 2 for v in vals)
    cov = sum((vals[i] - mean_p) * (scores[i] - mean_s) for i in range(n))
    denom = (var_p * var_s) ** 0.5
    return cov / denom if denom > 1e-12 else 0.0


def _score_correlation(records: list[dict], specs: list[dict]) -> dict[str, float]:
    scores = [r["final_score"] for r in records]
    n = len(scores)
    if n < 3:
        return {}
    mean_s = sum(scores) / n
    var_s = sum((v - mean_s) ** 2 for v in scores)
    out = {}
    for spec in specs:
        name = spec["name"]
        vals = [r["params"].get(name) for r in records]
        if all(isinstance(v, (int, float)) for v in vals):
            out[name] = _pearson(vals, scores, mean_s, var_s)
    return out


def _space_map_figure(records: list[dict], corr: dict[str, float]) -> go.Figure:
    ranked = sorted(corr, key=lambda p: -abs(corr[p]))
    if len(ranked) < 2:
        return go.Figure().add_annotation(text="Not enough numeric parameters for a 2D map")
    px, py = ranked[0], ranked[1]
    gens = [r["gen_index"] for r in records]
    xs = [r["params"][px] for r in records]
    ys = [r["params"][py] for r in records]
    best = max(records, key=lambda r: r["final_score"])
    fig = go.Figure([
        go.Scatter(x=xs, y=ys, mode="markers",
                   marker=dict(size=6, color=gens, colorscale="Plasma", showscale=True,
                              colorbar=dict(title="generation"), opacity=0.6),
                   name="trials",
                   hovertemplate=f"{px}=%{{x}}<br>{py}=%{{y}}<br>gen=%{{marker.color}}<extra></extra>"),
        go.Scatter(x=[best["params"][px]], y=[best["params"][py]], mode="markers",
                   marker=dict(symbol="star", size=16, color=C_BEST, line=dict(width=1, color="black")),
                   name="best trial"),
    ])
    fig.update_layout(
        title=f"Search-space map: {px} vs {py} (top-2 score-correlated params)",
        xaxis_title=px, yaxis_title=py, height=420,
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        margin={"l": 50, "r": 30, "t": 50, "b": 40},
    )
    return fig


def _correlation_figure(corr: dict[str, float]) -> go.Figure:
    ranked = sorted(corr.items(), key=lambda kv: kv[1])
    names = [k for k, _ in ranked]
    vals = [v for _, v in ranked]
    colors = [C_BEST if v < 0 else C_AVG for v in vals]
    fig = go.Figure([go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                            hovertemplate="%{y}: r=%{x:.3f}<extra></extra>")])
    fig.update_layout(
        title="Linear score correlation (screening only — see Importance/Interactions for coupling)",
        xaxis_title="Pearson r vs score", height=max(220, 40 * len(names) + 80),
        plot_bgcolor=C_BG, paper_bgcolor=C_BG,
        margin={"l": 140, "r": 30, "t": 50, "b": 40},
    )
    fig.add_vline(x=0, line_color="#888")
    return fig


def build_search_space_report(records: list[dict], specs: list[dict] | None = None):
    """Full report as a Dash ``html.Div``: convergence curve, per-parameter
    funnel, top-2 search-space map, and a linear correlation screen. Returns
    an explanatory error panel instead of raising if there isn't enough data
    yet -- this is user-triggered, so a clear "not ready" beats a stack trace.

    ``specs`` (name/min/max dicts) defaults to the live project's searched
    params; pass an archive's :func:`specs_from_snapshot` to report on an
    archived run's own parameter space.
    """
    from dash import dcc, html

    usable = _usable_records(records)
    if specs is None:
        specs = _active_specs()
    if len(usable) < 10 or not specs:
        return html.Div(
            f"Not enough completed trials yet for a search-space report "
            f"({len(usable)} usable, need >=10, {len(specs)} free parameters).",
            className="text-muted small p-2",
        )

    corr = _score_correlation(usable, specs)
    return html.Div(
        [
            html.P(
                "Convergence over generations, per-parameter narrowing, a 2D map of "
                "where the search ended up, and a linear score-correlation screen. "
                "Generated once on click, not auto-refreshed — press again after "
                "more trials land.",
                className="text-muted small",
            ),
            dcc.Graph(figure=_convergence_figure(usable)),
            dcc.Graph(figure=_funnel_figure(usable, specs)),
            dcc.Graph(figure=_space_map_figure(usable, corr)) if corr else html.Div(),
            dcc.Graph(figure=_correlation_figure(corr)) if corr else html.Div(),
        ]
    )
