"""Callbacks for the Importance / Interactions tab."""

from __future__ import annotations

from dash import Input, Output

from sofaopt.dashboard.plotting.interactions import (
    build_importance_bar,
    build_interaction_heatmap,
)


def register_interactions_callbacks(app) -> None:
    @app.callback(
        Output("importance-bar", "figure"),
        Output("interaction-heatmap", "figure"),
        Input("interactions-interval", "n_intervals"),
    )
    def _update_interactions(_):
        return build_importance_bar(), build_interaction_heatmap()
