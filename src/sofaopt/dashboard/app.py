"""Dash app factory and server launch for a sofaopt project dashboard."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable, Sequence

try:
    from dash import Dash, Input, Output, State, dcc, html
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'dash' package is required for the sofaopt dashboard. "
        f"Install it with: {sys.executable} -m pip install dash plotly"
    ) from exc

from sofaopt.dashboard import context
from sofaopt.dashboard.callbacks import (
    register_archives_callbacks,
    register_config_callbacks,
    register_interactions_callbacks,
    register_monitoring_callbacks,
    register_pareto_callbacks,
    register_run_callbacks,
    register_video_callbacks,
)
from sofaopt.dashboard.ui.tabs import (
    build_archives_tab,
    build_config_tab,
    build_interactions_tab,
    build_param_bounds_tab,
    build_pareto_tab,
    build_performance_tab,
    build_progress_tab,
    build_run_tab,
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

logger = logging.getLogger(__name__)

logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("dash").setLevel(logging.ERROR)


@dataclass(frozen=True)
class DashboardTab:
    """One project-supplied dashboard tab.

    Args:
        label: Tab caption shown in the tab bar.
        value: Unique tab id (must not collide with the built-ins:
            config, run, performance, progress, bounds).
        build: Zero-arg callable returning the tab's Dash layout children.
            Called once at app build time, after the project context is set —
            so it may read :mod:`sofaopt.dashboard.context`.
        register: Optional ``register(app)`` hook for the tab's callbacks.
        before: Insert the tab before the built-in with this id
            (default: append after the built-ins).
    """

    label: str
    value: str
    build: Callable[[], Any]
    register: Callable[[Any], None] | None = None
    before: str | None = None


def create_app(
    project: SofaOptProject,
    extra_tabs: Sequence[DashboardTab] = (),
    hide_tabs: Sequence[str] = (),
) -> Dash:
    """Build the Dash app for ``project``.

    Args:
        project: The project to serve.
        extra_tabs: Project-specific :class:`DashboardTab` additions.
        hide_tabs: Built-in tab ids to omit (e.g. ``("scenes",)`` when a
            project supplies its own replacement via ``extra_tabs``).
    """
    context.set_project(project)
    catalog = context.catalog()
    title = project.title or project.name
    instance_id = uuid.uuid4().hex[:12]

    app = Dash(
        __name__,
        title=title,
        update_title=None,
        suppress_callback_exceptions=True,
        external_stylesheets=[
            "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        ],
    )

    hidden = set(hide_tabs)
    tab_defs = _build_tab_defs(project, catalog, extra_tabs, hidden)
    default_tab = tab_defs[0][1]

    app.layout = html.Div(
        [
            dcc.Store(id="server-instance-id", data=instance_id),
            dcc.Interval(id="stale-check-interval", interval=15000, n_intervals=0),
            html.Div(id="stale-reload-sentinel", style={"display": "none"}),
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

    _register_tab_callbacks(app, project, hidden)
    _register_video_routes(app, project)
    _register_stale_reload(app, instance_id)
    for tab in extra_tabs:
        if tab.register is not None:
            tab.register(app)
    return app


def _build_tab_defs(
    project: SofaOptProject,
    catalog,
    extra_tabs: Sequence[DashboardTab],
    hidden: set,
) -> list[tuple]:
    """Ordered ``(label, value, children)`` triples for the tab bar."""
    tab_defs = []
    if project.config_file is not None:
        tab_defs.append(("Config", "config", build_config_tab()))
    tab_defs += [
        ("Run", "run", build_run_tab(catalog)),
        ("Performance", "performance", build_performance_tab()),
        ("Progress", "progress", build_progress_tab()),
        ("Parameter Bounds", "bounds", build_param_bounds_tab()),
    ]
    # Interaction analysis needs a scalar objective; skip in Pareto mode.
    if not project.multi_objective:
        tab_defs.append(("Importance / Interactions", "interactions", build_interactions_tab()))
    if project.multi_objective:
        tab_defs.append(("Pareto Front", "pareto", build_pareto_tab()))
    tab_defs.append(("Archives", "archives", build_archives_tab()))

    tab_defs = [t for t in tab_defs if t[1] not in hidden]
    return _insert_extra_tabs(tab_defs, extra_tabs)


def _insert_extra_tabs(tab_defs: list[tuple], extra_tabs: Sequence[DashboardTab]) -> list[tuple]:
    for tab in extra_tabs:
        entry = (tab.label, tab.value, tab.build())
        anchor = next((i for i, t in enumerate(tab_defs) if t[1] == tab.before), None)
        if anchor is None:
            tab_defs.append(entry)
        else:
            tab_defs.insert(anchor, entry)
    return tab_defs


def _register_tab_callbacks(app, project: SofaOptProject, hidden: set) -> None:
    """Register the built-in callbacks that match the visible tabs."""
    if project.config_file is not None and "config" not in hidden:
        register_config_callbacks(app)
    if "run" not in hidden:
        register_run_callbacks(app)
    register_monitoring_callbacks(app)
    register_video_callbacks(app)
    if not project.multi_objective and "interactions" not in hidden:
        register_interactions_callbacks(app)
    if project.multi_objective and "pareto" not in hidden:
        register_pareto_callbacks(app)
    if "archives" not in hidden:
        register_archives_callbacks(app)


def _register_video_routes(app, project) -> None:
    """Serve trial recordings and generated videos over HTTP so the browser can play them."""
    from flask import send_from_directory

    @app.server.route("/trial-video/<gen_name>/<trial_name>")
    def serve_trial_video(gen_name, trial_name):
        trial_dir = project.trials_dir / gen_name / trial_name
        return send_from_directory(str(trial_dir), "trial.mp4")

    @app.server.route("/runtime-video/<filename>")
    def serve_runtime_video(filename):
        video_dir = project.runtime_dir / "videos"
        return send_from_directory(str(video_dir), filename)

    @app.server.route("/runtime-summary")
    def serve_runtime_summary():
        return send_from_directory(str(project.runtime_dir), "summary.mp4")


def _register_stale_reload(app, instance_id: str) -> None:
    """Auto-reload a browser tab left open across a dashboard restart.

    ``launch_dashboard`` always runs with ``debug=False`` (see its
    docstring/call site), so Dash's own "server restarted, please refresh"
    banner never activates -- that's a devtools-only feature. Without this, a
    tab open from before a restart silently keeps its stale client state
    (stale component IDs, stale callback wiring) with no visible sign
    anything is wrong -- e.g. the Performance graph rendering empty even
    though the new server's data loads fine. A dashboard restart is routine
    during development (any sofaopt code change needs one), so this isn't a
    one-off: poll a tiny endpoint carrying THIS process's instance id and
    force a reload the moment it stops matching what the page was served
    with, the same "never silently serve stale state" principle as
    ``_check_port_free`` below.
    """

    @app.server.route("/instance-id")
    def serve_instance_id():
        return instance_id

    app.clientside_callback(
        """
        function(n_intervals, page_instance_id) {
            if (!page_instance_id) { return window.dash_clientside.no_update; }
            fetch('/instance-id')
                .then(r => r.text())
                .then(function(current) {
                    if (current && current !== page_instance_id) {
                        window.location.reload();
                    }
                })
                .catch(function() {});
            return window.dash_clientside.no_update;
        }
        """,
        Output("stale-reload-sentinel", "children"),
        Input("stale-check-interval", "n_intervals"),
        State("server-instance-id", "data"),
    )


def _check_port_free(host: str, port: int) -> None:
    """Fail loudly instead of silently serving stale code: a previous
    dashboard.py instance that was never actually killed can leave the port
    bound (or, on some Windows setups, appear to let a second process bind it
    too) -- the browser then talks to whichever process answers first, which
    may be running yesterday's code with no error anywhere. A live TCP
    connect attempt catches this regardless of the platform's bind-sharing
    quirks; checking bind() alone would not."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f"Port {port} is already answering -- a previous dashboard.py "
                f"instance is likely still running and would keep serving stale "
                f"code even after this one starts. Find and stop it first "
                f"(Windows: `netstat -ano | findstr :{port}`, then "
                f"`taskkill /PID <pid> /T /F`), or pass a different port."
            )


def launch_dashboard(
    project: SofaOptProject,
    port: int = 8050,
    open_browser: bool = True,
    extra_tabs: Sequence[DashboardTab] = (),
    hide_tabs: Sequence[str] = (),
) -> None:
    """Start the dashboard web server for ``project``."""
    from sofaopt.core.runtime_dirs import configure_console_logging

    _check_port_free("127.0.0.1", port)
    configure_console_logging()
    for _stream in (sys.stdout, sys.stderr):
        # Non-reconfigurable stream (e.g. pytest capture) — keep the default.
        with contextlib.suppress(Exception):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    logger.info(f"[info] Starting {project.title or project.name} on http://localhost:{port}")
    os.environ["WERKZEUG_RUN_MAIN"] = "false"
    os.environ.pop("WERKZEUG_SERVER_FD", None)

    app = create_app(project, extra_tabs=extra_tabs, hide_tabs=hide_tabs)
    launch_url = f"http://localhost:{port}/?v={int(time.time())}"

    if open_browser:
        def _open():
            time.sleep(2)
            webbrowser.open_new_tab(launch_url)

        threading.Thread(target=_open, daemon=True).start()

    # threaded=True: Werkzeug's dev server otherwise handles one request at a
    # time. During a heavy run (many trial dirs being written concurrently,
    # e.g. a large IPOP-restart popsize) a single slow load_trial_records()
    # rescan could block every other request behind it, including the page's
    # own polling -- symptomatically "the graph goes blank while the
    # optimization is running, comes back once it stops."
    app.run(debug=False, use_reloader=False, port=port, host="127.0.0.1", threaded=True)
