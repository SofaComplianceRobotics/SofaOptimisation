"""Callbacks for the Config tab: save the project's config file."""

from __future__ import annotations

import json
import re

from dash import Input, Output, State, html

from sofaopt.dashboard import context


def register_config_callbacks(app) -> None:
    @app.callback(
        Output("config-save-status", "children"),
        Input("config-save-btn", "n_clicks"),
        State("config-textarea", "value"),
        prevent_initial_call=True,
    )
    def save_config(_, text):
        cfg = context.project().config_file
        if cfg is None:
            return html.Span("No config_file set on the project.", style={"color": "#e03131"})
        if not text:
            return "Nothing to save."
        try:
            clean = re.sub(r"//[^\n]*", "", text)  # strip JSONC // comments before parsing
            data = json.loads(clean)
            cfg.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return html.Span("Saved.", style={"color": "#2f9e44"})
        except json.JSONDecodeError as exc:
            return html.Span(f"Invalid JSON: {exc}", style={"color": "#e03131"})
        except Exception as exc:
            return html.Span(f"Error: {exc}", style={"color": "#e03131"})
