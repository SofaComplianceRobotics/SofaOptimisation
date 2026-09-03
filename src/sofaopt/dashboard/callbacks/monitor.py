"""Callbacks for the Monitor tab: the live per-generation trial grid, restart
status, generation stats, and the jump-to-running-trial controls."""

from __future__ import annotations

from dash import Input, Output, State, ctx

from sofaopt.core.restart_events import load_restart_events
from sofaopt.dashboard import context
from sofaopt.dashboard.data.cache import (
    _current_generation_records,
    _load_data,
    _read_json,
)
from sofaopt.dashboard.ui.progress import (
    _build_progress_grid,
    _build_progress_stats,
    _build_restart_status,
    _find_earliest_not_done,
)


def _progress_children():
    """Restart-status panel + generation stats + trial grid for the Monitor tab."""
    records, _summaries = _load_data()
    current_records = _current_generation_records(records)
    events = load_restart_events(context.trials_dir())
    progress = _read_json(context.progress_file())
    return (
        _build_restart_status(events, progress),
        _build_progress_stats(current_records, records),
        _build_progress_grid(current_records),
    )


def register_monitor_callbacks(app) -> None:
    """Register progress grid/stats/restart status and the jump controls."""

    @app.callback(
        [
            Output("restart-status", "children"),
            Output("progress-stats", "children"),
            Output("progress-grid", "children"),
        ],
        Input("progress-interval", "n_intervals"),
    )
    def update_progress(_):
        return _progress_children()

    @app.callback(
        Output("jump-running-target-store", "data"),
        Input("jump-running-trial", "n_clicks"),
        Input("progress-interval", "n_intervals"),
        State("jump-auto-enabled", "data"),
    )
    def update_jump_target(_clicks, _intervals, auto_enabled):
        triggered = getattr(ctx, "triggered_id", None)
        is_auto = triggered == "progress-interval"
        if is_auto and not bool(auto_enabled):
            return {"target_id": None, "auto": True}
        records, _summaries = _load_data()
        current_records = _current_generation_records(records)
        return {"target_id": _find_earliest_not_done(current_records), "auto": is_auto}

    app.clientside_callback(
        """
        function(target, auto_enabled) {
            if (!target || !target.target_id) { return window.dash_clientside.no_update; }
            if (target.auto && !auto_enabled) { return window.dash_clientside.no_update; }
            const el = document.getElementById(target.target_id);
            if (el) { el.scrollIntoView({behavior: 'smooth', block: 'center'}); }
            return window.dash_clientside.no_update;
        }
        """,
        Output("jump-running-target-output", "children"),
        Input("jump-running-target-store", "data"),
        State("jump-auto-enabled", "data"),
    )

    app.clientside_callback(
        """
        function(n_intervals, auto_enabled) {
            if (!window._ajScrollListenerReady) {
                window._ajUserScrolled = false;
                var mark = function() { window._ajUserScrolled = true; };
                window.addEventListener('wheel',     mark, {passive: true});
                window.addEventListener('touchmove', mark, {passive: true});
                window.addEventListener('keydown', function(e) {
                    if ([' ','ArrowUp','ArrowDown','PageUp','PageDown','Home','End'].includes(e.key)) {
                        window._ajUserScrolled = true;
                    }
                }, {passive: true});
                window._ajScrollListenerReady = true;
            }
            if (!auto_enabled) { window._ajUserScrolled = false; return window.dash_clientside.no_update; }
            if (window._ajUserScrolled) { window._ajUserScrolled = false; return false; }
            return window.dash_clientside.no_update;
        }
        """,
        Output("jump-auto-enabled", "data", allow_duplicate=True),
        Input("progress-interval", "n_intervals"),
        State("jump-auto-enabled", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(n) {
            if (!n) { return window.dash_clientside.no_update; }
            window.scrollTo({top: 0, behavior: 'smooth'});
            return false;
        }
        """,
        Output("jump-auto-enabled", "data", allow_duplicate=True),
        Input("jump-top-button", "n_clicks"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        "function(n, cur) { if (!n) return window.dash_clientside.no_update; return !cur; }",
        Output("jump-auto-enabled", "data", allow_duplicate=True),
        Input("jump-auto-toggle", "n_clicks"),
        State("jump-auto-enabled", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        'function(on) { return on ? "Auto-jump: On" : "Auto-jump: Off"; }',
        Output("jump-auto-toggle", "children"),
        Input("jump-auto-enabled", "data"),
    )
