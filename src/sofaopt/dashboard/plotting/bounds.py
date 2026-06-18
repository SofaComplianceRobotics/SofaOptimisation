"""Parameter-bounds visualization: where the latest trial sits within each range."""

from __future__ import annotations

import json

import plotly.graph_objects as go

from sofaopt.dashboard import context

from .colors import C_BG


def _active_specs() -> list[dict]:
    """Searchable (non-frozen) param specs as plain dicts."""
    return [p.to_dict() for p in context.project().params if not p.is_frozen]


def _load_trial_param_values() -> list[dict]:
    """Read the per-trial params.json files written by the optimizer (latest 40)."""
    trials_dir = context.trials_dir()
    paths = []
    try:
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
    except Exception:
        pass

    configs = []
    for p in paths[-40:]:
        try:
            configs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return configs


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
            nbins = 64
            bin_centers = [(i + 0.5) / nbins for i in range(nbins)]
            z_rows = []
            for spec in active_specs:
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
                z_rows.append([c / maxc if maxc > 0 else 0.0 for c in counts] if total else [0.0] * nbins)
            fig.add_trace(
                go.Heatmap(
                    x=bin_centers,
                    y=param_names,
                    z=z_rows,
                    colorscale="YlOrRd",
                    showscale=False,
                    hovertemplate="%{y}<br>Value: %{x:.2f}<br>Rel. density: %{z:.2f}<extra></extra>",
                    zmin=0,
                    zmax=1,
                )
            )

        for spec in active_specs:
            name = spec["name"]
            param_min, param_max = spec["min"], spec["max"]
            span = (param_max - param_min) or 1.0
            values: list[float] = []
            if latest_config is not None:
                v = latest_config.get(name)
                if isinstance(v, (int, float)):
                    values = [float(v)]
                elif isinstance(v, (list, tuple)):
                    values = [float(x) for x in v if isinstance(x, (int, float))]
            if not values:
                values = [(param_min + param_max) / 2]

            marker_xs = [max(0.0, min(1.0, (val - param_min) / span)) for val in values]
            side_texts = [f"{val:.3f}" for val in values]
            if marker_xs:
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
        print(f"[warn] Error building param bounds: {exc}")
        return go.Figure().add_annotation(text=f"Error: {exc}")
