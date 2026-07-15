"""Figures for the Importance / Interactions tab.

Reuses :mod:`sofaopt.analysis` to turn the completed trials in ``study.db`` into
a fANOVA main-effect bar chart and a pairwise-interaction heatmap. The analysis
(surrogate fit + H-statistic) is moderately expensive, so the result is cached
and only recomputed when the database changes or a throttle window elapses.
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go

from sofaopt.dashboard import context

_CACHE: dict = {"key": None, "report": None, "ts": 0.0, "error": None}
_MIN_RECOMPUTE_S = 20.0  # don't refit the surrogate more often than this


def _placeholder(message: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[{
            "text": message, "showarrow": False,
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
            "font": {"size": 13, "color": "#888"},
        }],
    )
    return fig


def _get_report():
    """Return (report, error_message). Cached by db mtime + throttle window."""
    from sofaopt import analysis

    db = context.project().db_path
    if not db.exists():
        return None, "No study.db yet — start a run to populate the interaction map."

    key = db.stat().st_mtime
    now = time.time()
    if _CACHE["report"] is not None and _CACHE["key"] == key:
        return _CACHE["report"], None
    if _CACHE["report"] is not None and (now - _CACHE["ts"]) < _MIN_RECOMPUTE_S:
        return _CACHE["report"], None  # throttle: serve last good report
    try:
        report = analysis.analyze(db)
        _CACHE.update(key=key, report=report, ts=now, error=None)
        return report, None
    except ImportError as exc:
        # The [analysis] extra (scikit-learn) is not installed — an install
        # gap, not a data gap. Surface analysis.py's own install prompt verbatim
        # instead of framing it as "not enough data yet".
        _CACHE.update(ts=now, error=str(exc))
        return _CACHE["report"], (None if _CACHE["report"] is not None else str(exc))
    except Exception as exc:
        _CACHE.update(ts=now, error=str(exc))
        return _CACHE["report"], (None if _CACHE["report"] is not None else f"Not enough data yet: {exc}")


def build_importance_bar() -> go.Figure:
    """fANOVA main-effect importance per parameter (descending)."""
    report, err = _get_report()
    if report is None:
        return _placeholder(err or "No data.", "Parameter importance (fANOVA)")
    if not report.main_effects:
        return _placeholder("fANOVA needs a few more varied trials.", "Parameter importance (fANOVA)")

    items = sorted(report.main_effects.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker={"color": vals, "colorscale": "Blues"},
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Parameter importance — fANOVA main effects ({report.n_trials} trials)",
        xaxis_title="share of score variance",
        margin={"l": 20, "r": 20, "t": 40, "b": 30},
        height=max(220, 40 * len(names) + 80),
    )
    return fig


def build_interaction_heatmap() -> go.Figure:
    """Pairwise interaction strength between parameters."""
    report, err = _get_report()
    if report is None:
        return _placeholder(err or "No data.", "Pairwise interactions")

    names = report.param_names
    m = np.asarray(report.interaction_matrix, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=m, x=names, y=names, colorscale="Viridis", zmin=0.0,
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    label = "Sobol' S₂" if report.interaction_method == "sobol" else "Friedman H"
    fig.update_layout(
        title=f"Pairwise interactions — {label} (higher = more coupled)",
        margin={"l": 20, "r": 20, "t": 40, "b": 40},
        height=max(320, 46 * len(names) + 120),
        yaxis={"autorange": "reversed"},
    )
    return fig