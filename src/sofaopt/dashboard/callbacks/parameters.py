"""Callbacks for the Parameters tab: param table, bounds heatmap, interactions.

Interactions need a scalar objective, so that callback is registered only for a
single-objective study (matching the layout in ``ui/tabs/parameters.py``).
"""

from __future__ import annotations

from dash import Input, Output

from sofaopt.core.results import rank_completed
from sofaopt.dashboard import context
from sofaopt.dashboard.data.cache import _load_data
from sofaopt.dashboard.plotting.bounds import _build_param_bounds_graph, build_param_table
from sofaopt.dashboard.plotting.interactions import (
    build_importance_bar,
    build_interaction_heatmap,
)


def _best_params(records: list[dict]) -> dict | None:
    """Params of the best completed trial (highest recorded score), or None."""
    ranked = rank_completed(records)
    return ranked[0].get("params") if ranked else None


def register_parameters_callbacks(app) -> None:
    @app.callback(
        Output("param-table", "children"),
        Output("param-bounds-graph", "figure"),
        Input("bounds-interval", "n_intervals"),
    )
    def _update_params(_):
        records, _summaries = _load_data()
        return build_param_table(_best_params(records)), _build_param_bounds_graph(show_heatmap=True)

    if context.project().multi_objective:
        return

    @app.callback(
        Output("importance-bar", "figure"),
        Output("interaction-heatmap", "figure"),
        Input("interactions-interval", "n_intervals"),
    )
    def _update_interactions(_):
        return build_importance_bar(), build_interaction_heatmap()
