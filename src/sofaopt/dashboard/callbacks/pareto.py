"""Callback for the Pareto Front tab (only wired when multi_objective=True)."""

from __future__ import annotations

from dash import Input, Output

from sofaopt.dashboard import context
from sofaopt.dashboard.data.cache import _load_data
from sofaopt.dashboard.plotting.pareto import build_pareto_layout


def register_pareto_callbacks(app) -> None:
    @app.callback(
        Output("pareto-graphs", "children"),
        Input("pareto-interval", "n_intervals"),
    )
    def update_pareto(_):
        project = context.project()
        test_names = [t.name for t in project.tests]
        directions = [t.direction for t in project.tests]
        records, _ = _load_data()
        done = [r for r in records if str(r.get("state", "")).lower() == "done"]
        return build_pareto_layout(done, test_names, directions)
