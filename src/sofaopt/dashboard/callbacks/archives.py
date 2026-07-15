"""Callbacks for the Archives tab: archive/restore/delete + run comparison.

Archive and restore MOVE ``runtime/`` — both refuse to act while an
optimization subprocess is running (moving the workspace under a live run
would corrupt it). Delete is two-stage behind a confirm dialog.
"""

from __future__ import annotations

import logging
import time

from dash import ALL, Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate

from sofaopt.core.archive import (
    archive_run,
    comparison_data,
    delete_archive,
    list_archives,
    restore_archive,
    runtime_has_run_data,
)
from sofaopt.dashboard import context
from sofaopt.dashboard.plotting.archives import build_comparison_figure
from sofaopt.dashboard.process.process_manager import _proc_running

logger = logging.getLogger(__name__)

_CURRENT = "__current__"


def _optimize_running() -> bool:
    return _proc_running("optimize")


def _fmt_when(created_at: float) -> str:
    if not created_at:
        return "?"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(created_at))


def _archives_table(infos) -> html.Div:
    if not infos:
        return html.Div("No archives yet.", className="text-muted")
    header = html.Thead(
        html.Tr(
            [html.Th(h) for h in ("Archived", "Name", "Sampler", "Trials", "Best", "Notes", "")]
        )
    )
    rows = []
    for info in infos:
        key = info.path.name
        best = f"{info.best_score:.2f}" if isinstance(info.best_score, (int, float)) else "-"
        rows.append(
            html.Tr(
                [
                    html.Td(_fmt_when(info.created_at), className="text-nowrap"),
                    html.Td(info.name),
                    html.Td(info.sampler),
                    html.Td(str(info.n_trials), className="text-end"),
                    html.Td(best, className="text-end"),
                    html.Td(info.notes, className="text-muted small"),
                    html.Td(
                        [
                            html.Button(
                                "Restore",
                                id={"type": "archive-restore", "name": key},
                                n_clicks=0,
                                className="btn btn-sm btn-outline-secondary me-2",
                                title="Move this archive back to runtime/ (current run is auto-archived first)",
                            ),
                            html.Button(
                                "Delete",
                                id={"type": "archive-delete", "name": key},
                                n_clicks=0,
                                className="btn btn-sm btn-outline-danger",
                            ),
                        ],
                        className="text-nowrap",
                    ),
                ]
            )
        )
    return html.Table(
        [header, html.Tbody(rows)], className="table table-striped table-sm align-middle"
    )


def _compare_options(infos) -> list[dict]:
    options = []
    if runtime_has_run_data(context.project()):
        options.append({"label": "current run", "value": _CURRENT})
    options += [{"label": i.name, "value": i.path.name} for i in infos]
    return options


def _summary_children(entries) -> list:
    """Summary table + best-params diff for the compared runs.

    The table doubles as the accessible 'table view' of the comparison chart.
    """
    head = html.Thead(
        html.Tr([html.Th(h) for h in ("Run", "Sampler", "Trials", "Best score", "Notes")])
    )
    body = html.Tbody(
        [
            html.Tr(
                [
                    html.Td(e["label"]),
                    html.Td(e["sampler"]),
                    html.Td(str(e["n_trials"]), className="text-end"),
                    html.Td(
                        f"{e['best_score']:.2f}"
                        if isinstance(e["best_score"], (int, float))
                        else "-",
                        className="text-end fw-semibold",
                    ),
                    html.Td(e["notes"], className="text-muted small"),
                ]
            )
            for e in entries
        ]
    )
    children = [
        html.H6("Summary"),
        html.Table([head, body], className="table table-sm"),
    ]

    param_names: list[str] = []
    for e in entries:
        for k in e["best_params"]:
            if k not in param_names:
                param_names.append(k)
    if param_names:
        diff_head = html.Thead(
            html.Tr([html.Th("Best params")] + [html.Th(e["label"]) for e in entries])
        )
        diff_rows = []
        for name in param_names:
            values = [e["best_params"].get(name) for e in entries]
            cells = [
                html.Td(f"{v:.4g}" if isinstance(v, float) else ("-" if v is None else str(v)))
                for v in values
            ]
            distinct = len({repr(v) for v in values}) > 1
            diff_rows.append(
                html.Tr(
                    [html.Td(name, className="fw-semibold" if distinct else "")] + cells,
                    className="table-warning" if distinct else "",
                )
            )
        children += [
            html.H6("Best parameters (rows highlighted where runs differ)"),
            html.Table([diff_head, html.Tbody(diff_rows)], className="table table-sm"),
        ]
    return children


def register_archives_callbacks(app) -> None:  # noqa: C901  # Dash registrar: total is the sum of its small nested callbacks; the flat registration list reads best in one place
    @app.callback(
        Output("archives-table", "children"),
        Output("archive-compare-select", "options"),
        Input("archives-interval", "n_intervals"),
        Input("archives-dirty", "data"),
    )
    def refresh_archives(_n, _dirty):
        infos = list_archives(context.project())
        return _archives_table(infos), _compare_options(infos)

    @app.callback(
        Output("archive-action-status", "children"),
        Output("archives-dirty", "data"),
        Input("archive-now-btn", "n_clicks"),
        State("archive-name-input", "value"),
        State("archive-notes-input", "value"),
        State("archives-dirty", "data"),
        prevent_initial_call=True,
    )
    def archive_now(n_clicks, name, notes, dirty):
        if not n_clicks:
            raise PreventUpdate
        stopped = ""
        if _optimize_running():
            # Stop & archive: pause the run (clean — the study resumes if
            # restored later), then move its runtime into the archive.
            from sofaopt.dashboard.process.process_manager import stop_optimize_and_wait

            if not stop_optimize_and_wait():
                return html.Span(
                    "Could not stop the running optimization.", className="text-danger"
                ), dirty
            stopped = " (run stopped first — restore + Resume to continue it)"
        try:
            dest = archive_run(context.project(), name=name or "", notes=notes or "")
            return html.Span(
                f"Archived to {dest.name}{stopped}", className="text-success"
            ), (dirty or 0) + 1
        except FileNotFoundError:
            return html.Span("No run data to archive.", className="text-warning"), dirty
        except Exception as exc:
            logger.warning(f"[archive] Archive failed: {exc}")
            return html.Span(f"Archive failed: {exc}", className="text-danger"), dirty

    @app.callback(
        Output("archive-action-status", "children", allow_duplicate=True),
        Output("archives-dirty", "data", allow_duplicate=True),
        Input({"type": "archive-restore", "name": ALL}, "n_clicks"),
        State("archives-dirty", "data"),
        prevent_initial_call=True,
    )
    def restore_clicked(n_clicks_list, dirty):
        if not ctx.triggered_id or not any(n_clicks_list):
            raise PreventUpdate
        if _optimize_running():
            return html.Span(
                "Stop the running optimization first.", className="text-danger"
            ), dirty
        key = ctx.triggered_id["name"]
        try:
            restore_archive(context.project(), key)
            return html.Span(
                f"Restored {key} — it is the live run again.", className="text-success"
            ), (dirty or 0) + 1
        except Exception as exc:
            logger.warning(f"[archive] Restore failed: {exc}")
            return html.Span(f"Restore failed: {exc}", className="text-danger"), dirty

    @app.callback(
        Output("archive-delete-confirm", "displayed"),
        Output("archive-delete-target", "data"),
        Input({"type": "archive-delete", "name": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def delete_clicked(n_clicks_list):
        if not ctx.triggered_id or not any(n_clicks_list):
            raise PreventUpdate
        return True, ctx.triggered_id["name"]

    @app.callback(
        Output("archive-action-status", "children", allow_duplicate=True),
        Output("archives-dirty", "data", allow_duplicate=True),
        Input("archive-delete-confirm", "submit_n_clicks"),
        State("archive-delete-target", "data"),
        State("archives-dirty", "data"),
        prevent_initial_call=True,
    )
    def delete_confirmed(submit_n_clicks, key, dirty):
        if not submit_n_clicks or not key:
            raise PreventUpdate
        try:
            delete_archive(context.project(), key)
            return html.Span(f"Deleted {key}.", className="text-success"), (dirty or 0) + 1
        except Exception as exc:
            logger.warning(f"[archive] Delete failed: {exc}")
            return html.Span(f"Delete failed: {exc}", className="text-danger"), dirty

    @app.callback(
        Output("archive-compare-graph", "figure"),
        Output("archive-compare-summary", "children"),
        Input("archive-compare-btn", "n_clicks"),
        State("archive-compare-select", "value"),
        prevent_initial_call=True,
    )
    def compare(n_clicks, selected):
        if not n_clicks or not selected:
            raise PreventUpdate
        project = context.project()
        archive_keys = [v for v in selected if v != _CURRENT]
        entries = comparison_data(
            project, archive_keys, include_current=_CURRENT in selected
        )
        # Stable palette slots: 0 is reserved for the live run; archives keep
        # their position in the full (not selected) list across reselections.
        color_index = {"current run": 0}
        for i, info in enumerate(list_archives(project), start=1):
            color_index[info.name] = i
        return build_comparison_figure(entries, color_index), _summary_children(entries)
