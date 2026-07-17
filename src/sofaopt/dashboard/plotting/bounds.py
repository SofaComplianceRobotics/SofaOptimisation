"""Parameter-bounds visualization: where the latest trial sits within each range."""

from __future__ import annotations

import contextlib
import logging
import json

import plotly.graph_objects as go
from dash import html

from sofaopt.dashboard import context

from .colors import C_BG

logger = logging.getLogger(__name__)


def _active_specs() -> list[dict]:
    """Searchable (non-frozen) param specs as plain dicts."""
    return [p.to_dict() for p in context.project().params if not p.is_frozen]


def _fmt(v) -> str:
    """Compact display of a param value (bool stays true/false, not 1/0)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def build_param_table(best_params: dict | None) -> html.Div:
    """Every parameter in one table — including *frozen* ones (``low == high``),
    which the bounds heatmap omits. Columns: name / type / range / default /
    state / current best (from the best completed trial, or the frozen value)."""
    project = context.project()
    header = html.Thead(
        html.Tr([html.Th(h) for h in ("Parameter", "Type", "Range", "Default", "State", "Current best")])
    )
    rows = []
    for p in project.params:
        d = p.to_dict()
        frozen = p.is_frozen
        if p.type == "bool":
            rng = "true / false"
        elif frozen:
            rng = "frozen"
        else:
            rng = f"[{_fmt(d['min'])}, {_fmt(d['max'])}]"
        state = (
            html.Span("frozen", className="badge bg-secondary")
            if frozen
            else html.Span("active", className="badge bg-success")
        )
        if best_params and p.name in best_params:
            current = _fmt(best_params[p.name])
        elif frozen:
            current = _fmt(d["default"])
        else:
            current = "—"
        rows.append(
            html.Tr(
                [
                    html.Td(p.name),
                    html.Td(d["type"]),
                    html.Td(rng),
                    html.Td(_fmt(d["default"])),
                    html.Td(state),
                    html.Td(current),
                ]
            )
        )
    return html.Div(
        html.Table([header, html.Tbody(rows)], className="table table-sm align-middle"),
        className="table-responsive",
    )


def _load_trial_param_values() -> list[dict]:
    """Read the per-trial params.json files written by the optimizer (latest 40)."""
    trials_dir = context.trials_dir()
    paths = []
    # Trial dirs appear/move mid-scan during a live run; a partial list is fine.
    with contextlib.suppress(Exception):
        for gen_dir in sorted(
            trials_dir.glob("gen_*"),
            key=lambda d: int(d.name.split("_")[1]) if len(d.name.split("_")) > 1 else 0,
        ):
            for trial_dir in sorted(
                gen_dir.glob("trial_*"),
                key=lambda d: int(d.name.split("_")[1]) if len(d.name.split("_")) > 1 else 0,
            ):
                p = trial_dir / "params.json"
                if p.exists():
                    paths.append(p)

    configs = []
    for p in paths[-40:]:
        # Mid-write/partial params.json is expected while the optimizer runs.
        with contextlib.suppress(Exception):
            configs.append(json.loads(p.read_text(encoding="utf-8")))
    return configs


def _density_row(spec: dict, trial_configs: list[dict], nbins: int) -> list[float]:
    """One parameter's sampled-value histogram, row-normalized to [0, 1]."""
    name = spec["name"]
    span = (spec["max"] - spec["min"]) or 1.0
    counts = [0] * nbins
    total = 0
    for cfg in trial_configs:
        v = cfg.get(name)
        if isinstance(v, (int, float)):
            norm = max(0.0, min(1.0, (v - spec["min"]) / span))
            idx = min(int(norm * nbins), nbins - 1)
            counts[idx] += 1
            total += 1
    maxc = max(counts) if counts else 0
    return [c / maxc if maxc > 0 else 0.0 for c in counts] if total else [0.0] * nbins


def _density_heatmap(active_specs: list[dict], trial_configs: list[dict]) -> go.Heatmap:
    """Per-parameter histogram of sampled values, row-normalized to [0, 1]."""
    nbins = 64
    bin_centers = [(i + 0.5) / nbins for i in range(nbins)]
    z_rows = [_density_row(spec, trial_configs, nbins) for spec in active_specs]
    return go.Heatmap(
        x=bin_centers,
        y=[spec["name"] for spec in active_specs],
        z=z_rows,
        colorscale="YlOrRd",
        showscale=False,
        hovertemplate="%{y}<br>Value: %{x:.2f}<br>Rel. density: %{z:.2f}<extra></extra>",
        zmin=0,
        zmax=1,
    )


def _marker_values(spec: dict, latest_config: dict | None) -> list[float]:
    """Latest-trial value(s) for a parameter, defaulting to the range midpoint."""
    values: list[float] = []
    if latest_config is not None:
        v = latest_config.get(spec["name"])
        if isinstance(v, (int, float)):
            values = [float(v)]
        elif isinstance(v, (list, tuple)):
            values = [float(x) for x in v if isinstance(x, (int, float))]
    if not values:
        values = [(spec["min"] + spec["max"]) / 2]
    return values


def _add_marker(fig: go.Figure, spec: dict, values: list[float]) -> None:
    """Diamond marker(s) + value annotation for one parameter's latest value."""
    name = spec["name"]
    param_min, param_max = spec["min"], spec["max"]
    span = (param_max - param_min) or 1.0
    marker_xs = [max(0.0, min(1.0, (val - param_min) / span)) for val in values]
    side_texts = [f"{val:.3f}" for val in values]
    if not marker_xs:
        return
    fig.add_trace(
        go.Scatter(
            x=marker_xs,
            y=[name] * len(marker_xs),
            mode="markers",
            marker=dict(symbol="diamond", size=12, color="#ffffff", line=dict(width=2, color="#212121")),
            hovertemplate=(
                f"<b>{name}</b><br>Current: " + ", ".join(side_texts)
                + f"<br>Min: {param_min:.3f} | Max: {param_max:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=1.02, y=name, text="[" + ", ".join(side_texts) + "]",
        showarrow=False, xanchor="left", yanchor="middle",
        font=dict(size=11, color="#111"),
    )


def _add_latest_markers(fig: go.Figure, active_specs: list[dict], latest_config: dict | None) -> None:
    """Diamond marker(s) + value annotation for the latest trial, per parameter."""
    for spec in active_specs:
        _add_marker(fig, spec, _marker_values(spec, latest_config))


def _build_param_bounds_graph(show_heatmap: bool = False) -> go.Figure:
    """Heatmap of sampled values + a marker for the latest trial, per parameter."""
    try:
        active_specs = _active_specs()
        if not active_specs:
            return go.Figure().add_annotation(text="No active parameters to display")

        trial_configs = _load_trial_param_values()
        latest_config = trial_configs[-1] if trial_configs else None
        param_names = [spec["name"] for spec in active_specs]
        fig = go.Figure()

        if trial_configs:
            fig.add_trace(_density_heatmap(active_specs, trial_configs))
        _add_latest_markers(fig, active_specs, latest_config)

        row_h = 70 if show_heatmap else 52
        fig.update_layout(
            title="Parameter Bounds Monitor (Latest Trial)",
            xaxis=dict(title="Position in Range (0 = Min, 1 = Max)", range=[0, 1], fixedrange=True),
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(param_names)), fixedrange=True),
            barmode="overlay",
            height=max(400, 90 + len(active_specs) * row_h),
            showlegend=False,
            margin={"l": 160, "r": 70, "t": 50, "b": 50},
            plot_bgcolor=C_BG,
            paper_bgcolor=C_BG,
        )
        for spec in active_specs:
            fig.add_annotation(x=0.0, y=spec["name"], text=f"{spec['min']:.3f}", xanchor="left", yanchor="top", showarrow=False, yshift=-18, font=dict(size=9, color="#888"))
            fig.add_annotation(x=1.0, y=spec["name"], text=f"{spec['max']:.3f}", xanchor="right", yanchor="top", showarrow=False, yshift=-18, font=dict(size=9, color="#888"))
        return fig
    except Exception as exc:
        logger.warning(f"[warn] Error building param bounds: {exc}")
        return go.Figure().add_annotation(text=f"Error: {exc}")
