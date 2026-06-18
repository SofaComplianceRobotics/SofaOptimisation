"""Plotly trace builders for the performance graph."""

import plotly.graph_objects as go

from .colors import C_AVG, C_BEST, C_FINAL
from .compute import _calculate_smart_ticks, _compute_contributions, _test_color  # noqa: F401


def _build_bar_traces(records, plot_data, all_test_names):
    """Diverging stacked bar traces (one per test)."""
    traces = []
    xs = plot_data["xs"]
    contributions = plot_data["contributions"]
    failed_mask = plot_data["failed_mask"]
    is_complete = plot_data["is_complete"]

    for test_name in all_test_names:
        y_vals = []
        colors_list = []
        opacity_list = []
        for x, contrib, failed, complete in zip(xs, contributions, failed_mask, is_complete):
            y_vals.append(contrib.get(test_name, 0.0))
            colors_list.append(_test_color(test_name, all_test_names))
            alpha = 0.3 if failed else 1.0
            alpha *= 0.5 if not complete else 1.0
            opacity_list.append(alpha)
        traces.append(
            go.Bar(
                x=xs,
                y=y_vals,
                name=test_name,
                uid=f"bar-{test_name}",
                marker=dict(
                    color=colors_list,
                    opacity=opacity_list,
                    line=dict(color=_test_color(test_name, all_test_names), width=0.5),
                ),
                hoverinfo="skip",
                showlegend=True,
            )
        )
    return traces


def _build_hover_overlay(records, plot_data, all_test_names, bar_width):
    """Invisible bar overlay so hover works across the whole trial bar."""
    xs = plot_data["xs"]
    contributions = plot_data["contributions"]
    hover_texts = []
    y_vals = []
    for i, contrib in enumerate(contributions):
        r = records[i]
        gen = r.get("gen_index", "?")
        trial_id = r.get("trial_index", i)
        final_score = sum(contrib.values())
        detail = [
            f"<b>Gen {gen} | Trial {trial_id}</b>",
            f"<b>Total Score: {final_score:.4f}</b>",
            "─────────────────",
        ]
        for tn in all_test_names:
            detail.append(f"{tn}: {contrib.get(tn, 0.0):.4f}")
        hover_texts.append("<br>".join(detail))
        y_vals.append(final_score)

    customdata = [
        [hover_texts[i], records[i].get("gen_name", ""), records[i].get("trial_name", "")]
        for i in range(len(records))
    ]
    return go.Bar(
        x=xs,
        y=y_vals,
        base=0,
        width=[bar_width] * len(xs),
        marker=dict(color="rgba(0,0,0,0)", line=dict(width=0)),
        customdata=customdata,
        hovertemplate="%{customdata[0]}<extra></extra>",
        hoverlabel=dict(bgcolor="#d7ecff", bordercolor="#9ec5fe", font=dict(color="#16324f")),
        showlegend=False,
        name="hover",
        uid="hover-overlay",
    )


def _build_final_ticks(plot_data, bar_width):
    """Short horizontal tick per trial showing its final score."""
    xs = plot_data["xs"]
    final_scores = plot_data["final_scores"]
    x_segments, y_segments = [], []
    halfw = bar_width / 2.0
    for x, s in zip(xs, final_scores):
        x_segments.extend([x - halfw, x + halfw, None])
        y_segments.extend([s, s, None])
    return go.Scatter(
        x=x_segments,
        y=y_segments,
        mode="lines",
        name="Final score",
        uid="final-score",
        line=dict(color=C_FINAL, width=2),
        hoverinfo="skip",
        visible="legendonly",
        showlegend=True,
    )


def _build_avg_traces(plot_data, all_test_names):
    """Rolling-average, best-so-far, and per-test rolling-average traces."""
    from sofaopt.dashboard import context

    traces = []
    avg_x, avg_y = plot_data["avg_x"], plot_data["avg_y"]
    if avg_x:
        traces.append(
            go.Scatter(
                x=avg_x,
                y=avg_y,
                mode="lines",
                name=f"Rolling avg (±{context.CENTERED_AVG_HALF_WINDOW})",
                uid="rolling-avg",
                line=dict(color=C_AVG, width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    best_x, best_y = plot_data["best_x"], plot_data["best_y"]
    if best_x:
        traces.append(
            go.Scatter(
                x=best_x,
                y=best_y,
                mode="lines",
                name="Best so far",
                uid="best-so-far",
                line=dict(color=C_BEST, width=2),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    per_test_avg = plot_data.get("per_test_avg", {})
    for test_name in all_test_names:
        if test_name in per_test_avg:
            t_x, t_y = per_test_avg[test_name]
            if t_x:
                traces.append(
                    go.Scatter(
                        x=t_x,
                        y=t_y,
                        mode="lines",
                        name=f"~{test_name} avg",
                        uid=f"avg-{test_name}",
                        line=dict(color=_test_color(test_name, all_test_names), width=1.5, dash="dashdot"),
                        hoverinfo="skip",
                        visible="legendonly",
                        showlegend=True,
                    )
                )
    return traces
