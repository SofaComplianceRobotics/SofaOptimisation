"""Comparison figure for archived optimization runs."""

from __future__ import annotations

import plotly.graph_objects as go

from .colors import ARCHIVE_COLORS, C_BG


def build_comparison_figure(
    entries: list[dict], color_index_by_label: dict[str, int]
) -> go.Figure:
    """Overlaid best-so-far convergence curves, one step-line per run.

    ``entries`` come from :func:`sofaopt.core.archive.comparison_data` —
    recorded scores only. ``color_index_by_label`` maps every run to its
    stable palette slot so colors follow runs across reselections.
    """
    fig = go.Figure()
    for entry in entries:
        xs, ys = entry["curve"]
        if not xs:
            continue
        color = ARCHIVE_COLORS[
            color_index_by_label.get(entry["label"], 0) % len(ARCHIVE_COLORS)
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=entry["label"],
                line=dict(color=color, width=2, shape="hv"),
                hovertemplate="trial %{x}: best %{y:.2f}<extra>"
                + entry["label"]
                + "</extra>",
            )
        )
    fig.update_layout(
        title="Best score so far, per evaluation spent",
        xaxis_title="Completed trials",
        yaxis_title="Best recorded score",
        hovermode="x unified",
        height=420,
        margin={"l": 50, "r": 30, "t": 50, "b": 50},
        plot_bgcolor=C_BG,
        paper_bgcolor=C_BG,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uirevision="archive-compare",
    )
    if not fig.data:
        fig.add_annotation(text="No completed trials in the selected runs")
    return fig
