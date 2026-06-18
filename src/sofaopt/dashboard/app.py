"""Dash app factory and server launch for a sofaopt project dashboard."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import webbrowser

try:
    from dash import Dash, dcc, html
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'dash' package is required for the sofaopt dashboard. "
        f"Install it with: {sys.executable} -m pip install dash plotly"
    ) from exc

from sofaopt.dashboard import context
from sofaopt.dashboard.callbacks import (
    register_config_callbacks,
    register_monitoring_callbacks,
    register_optimise_callbacks,
    register_scene_callbacks,
)
from sofaopt.dashboard.ui.tabs import (
    build_config_tab,
    build_optimise_tab,
    build_param_bounds_tab,
    build_performance_tab,
    build_progress_tab,
    build_scenes_tab,
)
from sofaopt.dashboard.ui.tabs.styles import (
    BODY_STYLE,
    HEADER_BAR_STYLE,
    HEADER_INNER_STYLE,
    HEADER_SUBTITLE_STYLE,
    HEADER_TITLE_STYLE,
    PAGE_STYLE,
    TAB_CONTENT_STYLE,
    TAB_SELECTED_STYLE,
    TAB_STYLE,
    TABS_STYLE,
)
from sofaopt.project import SofaOptProject

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("dash").setLevel(logging.ERROR)


def create_app(project: SofaOptProject) -> Dash:
    """Build the Dash app for ``project``."""
    context.set_project(project)
    catalog = context.catalog()
    title = project.title or project.name

    app = Dash(
        __name__,
        title=title,
        update_title=None,
        suppress_callback_exceptions=True,
        external_stylesheets=[
            "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        ],
    )

    tab_defs = []
    if project.config_file is not None:
        tab_defs.append(("Config", "config", build_config_tab()))
    tab_defs += [
        ("Scenes", "scenes", build_scenes_tab(catalog)),
        ("Optimise", "optimise", build_optimise_tab(catalog)),
        ("Performance", "performance", build_performance_tab()),
        ("Progress", "progress", build_progress_tab()),
        ("Parameter Bounds", "bounds", build_param_bounds_tab()),
    ]
    default_tab = tab_defs[0][1]

    app.layout = html.Div(
        [
            html.Header(
                html.Div(
                    [
                        html.Span(title, style=HEADER_TITLE_STYLE),
                        html.Span("Optimise · Analyse · with SOFA", style=HEADER_SUBTITLE_STYLE),
                    ],
                    style=HEADER_INNER_STYLE,
                ),
                style=HEADER_BAR_STYLE,
            ),
            html.Div(
                dcc.Tabs(
                    id="tabs",
                    value=default_tab,
                    style=TABS_STYLE,
                    children=[
                        dcc.Tab(
                            label=label,
                            value=value,
                            children=html.Div(children, style=TAB_CONTENT_STYLE),
                            style=TAB_STYLE,
                            selected_style=TAB_SELECTED_STYLE,
                        )
                        for label, value, children in tab_defs
                    ],
                ),
                style=BODY_STYLE,
            ),
        ],
        style=PAGE_STYLE,
    )

    if project.config_file is not None:
        register_config_callbacks(app)
    register_scene_callbacks(app, catalog)
    register_optimise_callbacks(app)
    register_monitoring_callbacks(app)
    return app


def launch_dashboard(
    project: SofaOptProject, port: int = 8050, open_browser: bool = True
) -> None:
    """Start the dashboard web server for ``project``."""
    print(f"[info] Starting {project.title or project.name} on http://localhost:{port}")
    os.environ["WERKZEUG_RUN_MAIN"] = "false"
    os.environ.pop("WERKZEUG_SERVER_FD", None)

    app = create_app(project)
    launch_url = f"http://localhost:{port}/?v={int(time.time())}"

    if open_browser:
        def _open():
            time.sleep(2)
            webbrowser.open_new_tab(launch_url)

        threading.Thread(target=_open, daemon=True).start()

    app.run(debug=False, use_reloader=False, port=port, host="127.0.0.1")
