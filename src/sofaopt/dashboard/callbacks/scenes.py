"""Callbacks for the Scenes tab: launch a test scene in a SOFA viewer."""

from __future__ import annotations

from dash import Input, Output, State

from sofaopt.dashboard import context
from sofaopt.dashboard.process.process_manager import launch_scene


def register_scene_callbacks(app, catalog: dict) -> None:
    @app.callback(
        Output("scene-status", "children"),
        Input("scene-launch-btn", "n_clicks"),
        State("scene-select", "value"),
        prevent_initial_call=True,
    )
    def handle_launch(_, test_name):
        if not test_name:
            return "Select a scene first."
        try:
            spec = context.project().test(test_name)
        except KeyError:
            return f"Unknown test '{test_name}'."
        return launch_scene(spec.scene_file)
