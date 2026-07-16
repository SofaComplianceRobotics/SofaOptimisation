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


def _fmt(value, spec: str = "") -> str:
    """Format a possibly-missing numeric stat; '-' for None/non-numeric."""
    if not isinstance(value, (int, float)):
        return "-"
    return format(value, spec) if spec else str(value)


def _convergence_row(e) -> html.Tr:
    s = e.get("summary_stats") or {}
    final = f"{_fmt(s.get('final_mean'), '.1f')}±{_fmt(s.get('final_std'), '.1f')}"
    collapse = _fmt(s.get("diversity_collapse_pct"), ".0f")
    return html.Tr(
        [
            html.Td(e["label"]),
            html.Td(f"{_fmt(s.get('n_completed'))}/{_fmt(s.get('n_failed'))}", className="text-end"),
            html.Td(_fmt(s.get("mean_score"), ".1f"), className="text-end"),
            html.Td(final, className="text-end"),
            html.Td(_fmt(s.get("trials_to_90pct_best")), className="text-end"),
            html.Td("-" if collapse == "-" else f"{collapse}%", className="text-end"),
            html.Td(_fmt(s.get("n_restarts")), className="text-end"),
        ]
    )


def _convergence_table(entries) -> list:
    """Cross-run convergence & diversity table from each run's summary_stats.

    'Diversity collapse' = how much the searched population narrowed from its
    first to its last generations (100% = fully converged onto one design).
    """
    if not any(e.get("summary_stats") for e in entries):
        return []
    cols = ("Run", "Done/Fail", "Mean", "Final gen (mean±sd)", "→90% best", "Diversity collapse", "Restarts")
    head = html.Thead(html.Tr([html.Th(c) for c in cols]))
    rows = [_convergence_row(e) for e in entries]
    return [
        html.H6("Convergence & diversity", className="mt-3"),
        html.Table([head, html.Tbody(rows)], className="table table-sm"),
    ]


def _summary_children(entries) -> list:
    """Summary table + convergence stats + best-params diff for the compared runs.

    The tables double as the accessible 'table view' of the comparison chart.
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
    return [
        html.H6("Summary"),
        html.Table([head, body], className="table table-sm"),
        *_convergence_table(entries),
        *_params_diff_table(entries),
    ]


def _all_param_names(entries) -> list[str]:
    names: list[str] = []
    for e in entries:
        for k in e["best_params"]:
            if k not in names:
                names.append(k)
    return names


def _params_diff_table(entries) -> list:
    """Best-params-per-run table, rows highlighted where the runs differ."""
    param_names = _all_param_names(entries)
    if not param_names:
        return []
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
    return [
        html.H6("Best parameters (rows highlighted where runs differ)"),
        html.Table([diff_head, html.Tbody(diff_rows)], className="table table-sm"),
    ]


def _do_archive_now(name, notes, dirty):
    """Stop-if-running, then archive the current runtime. Returns (status, dirty)."""
    stopped = ""
    if _optimize_running():
        # Stop & archive: pause the run (clean — the study resumes if restored
        # later), then move its runtime into the archive.
        from sofaopt.dashboard.process.process_manager import stop_optimize_and_wait

        if not stop_optimize_and_wait():
            return html.Span("Could not stop the running optimization.", className="text-danger"), dirty
        stopped = " (run stopped first — restore + Resume to continue it)"
    try:
        dest = archive_run(context.project(), name=name or "", notes=notes or "")
        return html.Span(f"Archived to {dest.name}{stopped}", className="text-success"), (dirty or 0) + 1
    except FileNotFoundError:
        return html.Span("No run data to archive.", className="text-warning"), dirty
    except Exception as exc:
        logger.warning(f"[archive] Archive failed: {exc}")
        return html.Span(f"Archive failed: {exc}", className="text-danger"), dirty


def _do_restore(key, dirty):
    if _optimize_running():
        return html.Span("Stop the running optimization first.", className="text-danger"), dirty
    try:
        restore_archive(context.project(), key)
        return html.Span(f"Restored {key} — it is the live run again.", className="text-success"), (dirty or 0) + 1
    except Exception as exc:
        logger.warning(f"[archive] Restore failed: {exc}")
        return html.Span(f"Restore failed: {exc}", className="text-danger"), dirty


def _do_delete(key, dirty):
    try:
        delete_archive(context.project(), key)
        return html.Span(f"Deleted {key}.", className="text-success"), (dirty or 0) + 1
    except Exception as exc:
        logger.warning(f"[archive] Delete failed: {exc}")
        return html.Span(f"Delete failed: {exc}", className="text-danger"), dirty


def _do_compare(selected):
    project = context.project()
    archive_keys = [v for v in selected if v != _CURRENT]
    entries = comparison_data(project, archive_keys, include_current=_CURRENT in selected)
    # Stable palette slots: 0 is reserved for the live run; archives keep their
    # position in the full (not selected) list across reselections.
    color_index = {"current run": 0}
    for i, info in enumerate(list_archives(project), start=1):
        color_index[info.name] = i
    return build_comparison_figure(entries, color_index), _summary_children(entries)


def register_archives_callbacks(app) -> None:  # noqa: C901  # Dash registrar: cyclomatic count is the sum of each thin callback's one guard branch; the flat registration list reads best in one place (heavy bodies already extracted to _do_* helpers)
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
        return _do_archive_now(name, notes, dirty)

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
        return _do_restore(ctx.triggered_id["name"], dirty)

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
        return _do_delete(key, dirty)

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
        return _do_compare(selected)
