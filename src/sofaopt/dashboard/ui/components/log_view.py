"""Reusable run-log window: a dark terminal ``<pre>`` tailing the optimizer's
``optimize.log`` with an All / Warnings / Errors level filter.

The log is written by ``core.runtime_dirs.attach_run_log_file`` with a
``HH:MM:SS LEVEL message`` line format, so filtering is a cheap line scan. A
continuation line (a traceback body, a wrapped message) carries no level of its
own, so it inherits the level of the line above it — that keeps a multi-line
traceback attached to its ERROR when the view is filtered.

Built parametric on ids so more than one tab (Run, Monitor) can embed it.
"""

from __future__ import annotations

from typing import Callable

from dash import Input, Output, dcc, html

_LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_FILTER_THRESHOLD = {"all": 0, "warn": 30, "error": 40}


def _line_level(line: str, current: int) -> int:
    """Level of a log line, or ``current`` when the line carries none."""
    parts = line.split(None, 2)  # "HH:MM:SS LEVEL message"
    if len(parts) >= 2 and parts[1] in _LEVEL_RANK:
        return _LEVEL_RANK[parts[1]]
    return current


def filter_log_lines(text: str, level: str) -> str:
    """Keep only lines at or above the selected filter level."""
    threshold = _FILTER_THRESHOLD.get(level, 0)
    if threshold == 0 or not text:
        return text
    out = []
    current = 0
    for line in text.splitlines():
        current = _line_level(line, current)
        if current >= threshold:
            out.append(line)
    return "\n".join(out)


def build_log_view(
    *, pre_id: str, interval_id: str, filter_id: str, interval_ms: int = 1000
) -> html.Div:
    """A filterable, auto-refreshing log window."""
    # Lazy: components must not import the tabs package at module load (the tabs
    # __init__ imports the Run tab, which imports this component — a cycle).
    from sofaopt.dashboard.ui.tabs.styles import LOG_STYLE

    return html.Div(
        [
            html.Div(
                [
                    html.Span("Log", className="fw-semibold me-3 small text-muted"),
                    dcc.RadioItems(
                        id=filter_id,
                        options=[
                            {"label": " All", "value": "all"},
                            {"label": " Warnings", "value": "warn"},
                            {"label": " Errors", "value": "error"},
                        ],
                        value="all",
                        inline=True,
                        inputClassName="me-1",
                        labelClassName="me-3 small",
                    ),
                ],
                className="d-flex align-items-center mb-2",
            ),
            html.Pre(id=pre_id, style=LOG_STYLE),
            dcc.Interval(id=interval_id, interval=interval_ms, n_intervals=0),
        ]
    )


def register_log_view(
    app, *, pre_id: str, interval_id: str, filter_id: str, source: Callable[[], str]
) -> None:
    """Wire the interval + filter to the ``<pre>``. ``source`` returns the raw
    log text (e.g. ``lambda: _read_proc_log("optimize")``)."""

    @app.callback(
        Output(pre_id, "children"),
        Input(interval_id, "n_intervals"),
        Input(filter_id, "value"),
    )
    def _update(_n, level):
        return filter_log_lines(source(), level)
