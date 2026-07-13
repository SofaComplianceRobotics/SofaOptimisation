"""Performance graph and leaderboard."""

import contextlib
import logging
import plotly.graph_objects as go

from sofaopt.dashboard import context as _ctx
from .colors import C_BG
from .compute import _collect_all_test_names, compute_plot_data
from .traces import (
    _build_avg_traces,
    _build_bar_traces,
    _build_final_ticks,
    _build_hover_overlay,
)

logger = logging.getLogger(__name__)


def _build_performance_graph(records: list[dict], summaries: list[dict]) -> go.Figure:
    """Plotly figure: per-test contributions + score trends over trials."""
    if not records:
        return go.Figure().add_annotation(text="No data available")
    try:
        all_test_names = _collect_all_test_names(records)
        plot_data = compute_plot_data(records, all_test_names)
        xs = plot_data["xs"]
        bar_width = max(0.4, (max(xs) - min(xs) + 1) / len(xs) * 0.8) if xs else 0.8

        all_traces = (
            _build_bar_traces(records, plot_data, all_test_names)
            + [
                _build_hover_overlay(records, plot_data, all_test_names, bar_width),
                _build_final_ticks(plot_data, bar_width),
            ]
            + _build_avg_traces(plot_data, all_test_names)
            + _build_video_markers(records, plot_data)
        )

        fig = go.Figure(data=all_traces)
        fig.update_layout(
            title="Performance Overview: Per-Test Contributions & Score Trends",
            xaxis_title="Trial",
            yaxis_title="Score / Contribution",
            barmode="relative",
            hovermode="x unified",
            height=600,
            margin={"l": 50, "r": 50, "t": 50, "b": 50},
            plot_bgcolor=C_BG,
            paper_bgcolor=C_BG,
            uirevision="performance-graph",
        )
        # Cosmetic animation only; transition support varies across plotly versions.
        with contextlib.suppress(Exception):
            fig.layout.transition = dict(duration=600, easing="cubic-in-out")
        _add_restart_markers(fig)
        return fig
    except Exception as exc:
        logger.warning(f"[warn] Error building performance graph: {exc}")
        return go.Figure().add_annotation(text=f"Error: {exc}")


def _add_restart_markers(fig: go.Figure) -> None:
    """Dotted vertical line at each IPOP restart (see core/restart_events.py).

    Self-contained and best-effort like ``_build_video_markers``: a failure
    here must degrade to "no markers", never replace the whole graph — so it
    runs after the figure is otherwise complete, inside its own guard.
    """
    # Cosmetic only — a marker failure must never blank the whole graph.
    with contextlib.suppress(Exception):
        from sofaopt.core.restart_events import load_restart_events
        for ev in load_restart_events(_ctx.trials_dir()):
            fig.add_vline(
                x=ev["trial_chron"],
                line_dash="dot",
                line_color="#868e96",
                opacity=0.6,
                annotation_text=(
                    f"restart {ev['restart_index']}: "
                    f"{ev['old_popsize']}→{ev['new_popsize']}"
                ),
                annotation_position="top",
                annotation_font_size=10,
            )


def _build_video_markers(records: list[dict], plot_data: dict) -> list:
    """Scatter trace marking trials that have a cached trial.mp4 recording."""
    try:
        trials_dir = _ctx.trials_dir()
        xs = plot_data["xs"]
        final_scores = plot_data["final_scores"]
        video_xs, video_ys = [], []
        for i, r in enumerate(records):
            if i >= len(xs):
                break
            mp4 = trials_dir / r.get("gen_name", "") / r.get("trial_name", "") / "trial.mp4"
            if mp4.exists():
                video_xs.append(xs[i])
                video_ys.append(final_scores[i])
        if not video_xs:
            return []
        return [
            go.Scatter(
                x=video_xs,
                y=video_ys,
                mode="markers",
                name="Has video",
                uid="video-markers",
                marker=dict(symbol="star", size=10, color="#ff7f0e", opacity=0.9),
                hoverinfo="skip",
                showlegend=True,
            )
        ]
    except Exception:
        return []


def _build_leaderboard_html(records: list[dict]):
    """Top-10 leaderboard as a Dash table."""
    from dash import html

    valid = [r for r in records if not r.get("failed", False)]
    sorted_records = sorted(valid, key=lambda r: r.get("final_score", 0), reverse=True)
    if not sorted_records:
        return html.Div("No valid trials found.", className="text-muted")

    rows = []
    for rank, record in enumerate(sorted_records[:10], 1):
        label = f"{record.get('gen_name', '')} / {record.get('trial_name', '')}"
        score = record.get("final_score", 0.0)
        marker = " ⭐ BEST" if rank == 1 else ""
        rows.append(
            html.Tr(
                [
                    html.Td(str(rank), className="fw-bold"),
                    html.Td(label),
                    html.Td(f"{score:.4f}", className="text-end"),
                    html.Td(marker, className="text-success fw-bold"),
                ]
            )
        )
    return html.Div(
        [
            html.H5("Top 10 Trials", className="mt-3 mb-2"),
            html.Table(
                [
                    html.Thead(
                        html.Tr([html.Th("Rank"), html.Th("Trial"), html.Th("Score"), html.Th("Status")])
                    ),
                    html.Tbody(rows),
                ],
                className="table table-striped table-sm",
            ),
        ]
    )
