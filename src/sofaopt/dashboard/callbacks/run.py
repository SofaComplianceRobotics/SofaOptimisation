"""Callbacks for the Run tab: scene preview, weight sliders, pie, Run/Pause, log.

Merges the former Optimise and Scenes callbacks. The optimizer machinery keeps
its ``opt-*`` component ids so the (id-coupled) clientside weight callbacks port
over verbatim; the newly added pieces use ``run-*`` ids.
"""

from __future__ import annotations

import json
import os

from dash import ALL, Input, Output, State, ctx
from dash.exceptions import PreventUpdate

from sofaopt.core import envkeys
from sofaopt.dashboard import context
from sofaopt.dashboard.process.process_manager import (
    _read_proc_log,
    external_run_pid,
    launch_scene,
    optimize_running,
    start_optimize,
    stop_optimize,
)
from sofaopt.dashboard.ui.components import register_log_view


def _selected_tests(check_vals, check_ids, store) -> tuple[list[str], dict[str, int]]:
    """Checked test names and their stored weights, in catalog order."""
    test_names: list[str] = []
    test_weights: dict[str, int] = {}
    for checks, cid in zip(check_vals, check_ids, strict=True):
        if checks:
            name = cid["test"]
            test_names.append(name)
            test_weights[name] = int(store.get(name, 0))
    return test_names, test_weights


def _selection_error(
    test_names, gated_names, test_weights, sampler, n_parallel,
    run_until_converged=None, restart_patience=None,
    cmaes_restarts=None, stall_generations=None, restart_flags=None,
) -> str | None:
    """Validation message for the Run request, or None when it can start.

    The authoritative gate is ``SofaOptProject.__post_init__`` in the launched
    subprocess; this only pre-flights the UI-knowable conditions so a
    misconfigured run surfaces a friendly message instead of a subprocess
    traceback in the log. Restart settings are validated against the UI values
    (they override the project fields via the ``OPT_*`` env keys).
    """
    if not test_names:
        return "No tests selected."
    if len(gated_names) == len(test_names):
        return "At least one selected test must stay ungated so the gate can open."
    total = sum(test_weights.values())
    if total != 100:
        return f"Weights must sum to 100% (currently {total}%)."
    if sampler == "cmaes" and int(n_parallel or 0) < 4:
        return "CMA-ES needs Parallel >= 4. Lower it only with a different sampler."
    has_trigger = int(stall_generations or 0) > 0 or "conv" in (restart_flags or [])
    if int(cmaes_restarts or 0) > 0 and not has_trigger:
        return (
            "Restarts need a trigger — set Stall gens > 0 or check "
            "'Restart on convergence'."
        )
    return _converged_error(
        sampler, run_until_converged, restart_patience, cmaes_restarts, has_trigger
    )


def _converged_error(
    sampler, run_until_converged, restart_patience, cmaes_restarts, has_trigger
) -> str | None:
    """Pre-flight the 'Run until converged' toggle against the UI settings.

    Mirrors SofaOptProject's own validation (restart_on_convergence counts as
    a trigger) so a config the validator accepts is never refused here.
    """
    if not (run_until_converged and "converged" in run_until_converged):
        return None
    if sampler != "cmaes":
        return "'Run until converged' needs the CMA-ES sampler."
    if int(restart_patience or 0) < 1:
        return "'Run until converged' needs restart patience >= 1."
    if int(cmaes_restarts or 0) <= 0 or not has_trigger:
        return (
            "'Run until converged' needs Restarts > 0 and a restart trigger "
            "(Stall gens > 0 or 'Restart on convergence')."
        )
    return None


def _apply_restart_env(env: dict, restarts: dict | None) -> None:
    """Restart/convergence-row overrides (a None number = leave the project's
    value alone; the flags checklist is always authoritative)."""
    restarts = restarts or {}
    for field, key in (
        ("cmaes_restarts", envkeys.CMAES_RESTARTS),
        ("stall_generations", envkeys.STALL_GENERATIONS),
        ("inc_popsize", envkeys.CMAES_INC_POPSIZE),
    ):
        value = restarts.get(field)
        if value is not None:
            env[key] = str(int(value))
    flags = restarts.get("flags") or []
    env[envkeys.RESTART_ON_CONVERGENCE] = "1" if "conv" in flags else "0"
    env[envkeys.WARM_RESTARTS] = "1" if "warm" in flags else "0"
    env[envkeys.DEDUP_TRIALS] = "1" if "dedup" in flags else "0"


def _optimizer_env(
    test_names, test_weights, gated_names,
    sampler, seed_sampler, cmaes_margin, n_parallel, n_generations,
    run_until_converged=None, restart_patience=None, prune_mode=None,
    restarts=None,
) -> dict:
    """Environment for the optimizer subprocess: selection + setting overrides."""
    env = os.environ.copy()
    env[envkeys.SELECTED_TESTS] = ",".join(test_names)
    env[envkeys.TEST_WEIGHTS] = json.dumps(test_weights)
    if gated_names:
        env[envkeys.GATED_TESTS] = ",".join(gated_names)
    if prune_mode:
        env[envkeys.PRUNE_MODE] = str(prune_mode)
    _apply_restart_env(env, restarts)

    # Optimizer-setting overrides → honored by run_optimization before build_study.
    if sampler:
        env[envkeys.SAMPLER] = str(sampler)
    if seed_sampler:
        env[envkeys.SEED_SAMPLER] = str(seed_sampler)
    env[envkeys.CMAES_MARGIN] = "1" if (cmaes_margin and "margin" in cmaes_margin) else "0"
    env[envkeys.RUN_UNTIL_CONVERGED] = (
        "1" if (run_until_converged and "converged" in run_until_converged) else "0"
    )
    if n_parallel:
        env[envkeys.N_PARALLEL] = str(int(n_parallel))
    if n_generations:
        env[envkeys.N_GENERATIONS] = str(int(n_generations))
    if restart_patience:
        env[envkeys.RESTART_PATIENCE] = str(int(restart_patience))
    return env


_WEIGHT_SYNC_JS = """
        function(slider_vals, check_vals, eq_clicks, norm_clicks, slider_ids, store) {
            var NO_UPDATE = window.dash_clientside.no_update;
            var ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || ctx.triggered.length === 0) return NO_UPDATE;
            var prop_id = ctx.triggered[0].prop_id;
            var dot = prop_id.lastIndexOf('.');
            var id_part = prop_id.substring(0, dot);
            var tid;
            try { tid = JSON.parse(id_part); } catch(e) { tid = id_part; }
            store = store ? Object.assign({}, store) : {};
            var all_tests = slider_ids.map(function(s) { return s.test; });
            var selected_tests = [];
            slider_ids.forEach(function(s, i) {
                if (check_vals[i] && check_vals[i].length > 0) selected_tests.push(s.test);
            });
            function equal_split(n) {
                if (n === 0) return [];
                var base = Math.floor(100 / n), rem = 100 - base * n;
                return Array.from({length: n}, function(_, i) { return base + (i < rem ? 1 : 0); });
            }
            function normalize_selected() {
                var total = selected_tests.reduce(function(s,t){ return s+(store[t]||0); }, 0);
                all_tests.forEach(function(t){ if (selected_tests.indexOf(t) < 0) store[t] = 0; });
                if (total === 0) {
                    var w = equal_split(selected_tests.length);
                    selected_tests.forEach(function(t,i){ store[t] = w[i]; });
                } else {
                    var scaled = selected_tests.map(function(t){ return Math.round((store[t]||0) / total * 100); });
                    var diff = 100 - scaled.reduce(function(a,b){return a+b;}, 0);
                    if (diff) scaled[0] += diff;
                    selected_tests.forEach(function(t,i){ store[t] = scaled[i]; });
                }
            }
            if (tid === 'opt-equal-btn') {
                if (selected_tests.length === 0) return NO_UPDATE;
                var w = equal_split(selected_tests.length);
                all_tests.forEach(function(t){ store[t] = 0; });
                selected_tests.forEach(function(t,i){ store[t] = w[i]; });
                return store;
            }
            if (tid === 'opt-normalize-btn') {
                if (selected_tests.length === 0) return NO_UPDATE;
                normalize_selected(); return store;
            }
            if (tid && typeof tid === 'object' && tid.type === 'test-check') {
                if (selected_tests.length === 0) { all_tests.forEach(function(t){ store[t] = 0; }); return store; }
                normalize_selected(); return store;
            }
            if (tid && typeof tid === 'object' && tid.type === 'weight-slider') {
                var changed_test = tid.test;
                var changed_value = null;
                slider_ids.forEach(function(sid, i) { if (sid.test === changed_test) changed_value = slider_vals[i]; });
                if (changed_value === null) return NO_UPDATE;
                if (store[changed_test] === changed_value) return NO_UPDATE;
                if (selected_tests.indexOf(changed_test) < 0) return NO_UPDATE;
                var n_sel = selected_tests.length;
                if (n_sel === 1) { store[changed_test] = 100; return store; }
                var old_val = store[changed_test] || 0;
                var new_val = Math.max(0, Math.min(100, Math.round(changed_value)));
                var delta = new_val - old_val;
                if (delta === 0) return NO_UPDATE;
                store[changed_test] = new_val;
                var idx = selected_tests.indexOf(changed_test);
                var remaining = delta;
                for (var step = 1; step < n_sel; step++) {
                    var next_test = selected_tests[(idx + step) % n_sel];
                    var cur = store[next_test] || 0;
                    if (remaining > 0) { var take = Math.min(remaining, cur); store[next_test] = cur - take; remaining -= take; }
                    else { var give = Math.min(-remaining, 100 - cur); store[next_test] = cur + give; remaining += give; }
                    if (remaining === 0) break;
                }
                if (remaining !== 0) store[changed_test] = new_val - remaining;
                all_tests.forEach(function(t) { if (selected_tests.indexOf(t) < 0) store[t] = 0; });
                return store;
            }
            return NO_UPDATE;
        }
        """

_SLIDER_MIRROR_JS = """
        function(store, slider_ids) {
            if (!store) return slider_ids.map(function() { return 0; });
            return slider_ids.map(function(sid) { return store[sid.test] || 0; });
        }
        """

_PIE_JS = """
        function(store, slider_ids, check_vals) {
            var palette = ["#4c8bf5","#e84393","#34a853","#fa7b17","#9c27b0","#00bcd4","#ff5722","#8bc34a"];
            store = store || {};
            var labels = [], values = [], colors = [];
            slider_ids.forEach(function(sid, i) {
                var w = store[sid.test] || 0;
                if (check_vals[i] && check_vals[i].length > 0 && w > 0) {
                    labels.push(sid.test); values.push(w); colors.push(palette[i % palette.length]);
                }
            });
            var layout = {margin: {l:10, r:10, t:30, b:10}, showlegend: false, paper_bgcolor: 'rgba(0,0,0,0)'};
            if (labels.length === 0) {
                return {data: [{type:'pie', labels:['none'], values:[1], marker:{colors:['#dee2e6']}, textinfo:'none', hoverinfo:'none'}], layout: layout};
            }
            return {data: [{type: 'pie', labels: labels, values: values, marker: {colors: colors}, textinfo: 'label+percent', hovertemplate: '%{label}: %{value}%<extra></extra>', hole: 0.35}], layout: layout};
        }
        """

_WEIGHT_STATUS_JS = """
        function(store, check_vals, slider_ids) {
            store = store || {};
            var selected_count = check_vals.filter(function(v) { return v && v.length > 0; }).length;
            var total = 0;
            slider_ids.forEach(function(sid, i) { if (check_vals[i] && check_vals[i].length > 0) total += (store[sid.test] || 0); });
            var mk = function(txt, cls) { return {type:'Span', namespace:'dash_html_components', props:{children: txt, className: cls}}; };
            if (selected_count === 0) return mk('Select at least one test.', 'text-warning');
            if (total !== 100) return mk('Selected weights sum to ' + total + '% — must equal 100%.', 'text-danger fw-semibold');
            return mk(selected_count + ' test(s) selected · weights OK (total 100%).', 'text-success fw-semibold');
        }
        """


def _do_scene_preview() -> str:
    """Launch the clicked row's scene in an interactive viewer (Run tab)."""
    tid = ctx.triggered_id
    if not isinstance(tid, dict) or tid.get("type") != "scene-preview":
        raise PreventUpdate
    name = tid["test"]
    try:
        spec = context.project().test(name)
    except KeyError:
        return f"Unknown test '{name}'."
    return launch_scene(spec.scene_file)


def _do_run_or_pause(
    check_vals, check_ids, gate_vals, gate_ids, store,
    sampler, seed_sampler, cmaes_margin, n_parallel, n_generations,
    run_until_converged, restart_patience, prune_mode, restarts,
) -> str:
    """Start/Resume (validate + launch) or Pause, per the triggering button.

    ``restarts`` bundles the restart/convergence-row values:
    ``{"cmaes_restarts", "stall_generations", "inc_popsize", "flags"}``.
    """
    if ctx.triggered_id == "opt-stop-btn":
        return stop_optimize()

    store = store or {}
    test_names, test_weights = _selected_tests(check_vals, check_ids, store)
    gated_names = [
        cid["test"]
        for checks, cid in zip(gate_vals, gate_ids, strict=True)
        if checks and cid["test"] in test_names
    ]
    error = _selection_error(
        test_names, gated_names, test_weights, sampler, n_parallel,
        run_until_converged, restart_patience,
        restarts.get("cmaes_restarts"), restarts.get("stall_generations"),
        restarts.get("flags"),
    )
    if error:
        return error
    return start_optimize(_optimizer_env(
        test_names, test_weights, gated_names,
        sampler, seed_sampler, cmaes_margin, n_parallel, n_generations,
        run_until_converged, restart_patience, prune_mode, restarts,
    ))


def _current_button_state() -> tuple[str, bool, str, bool]:
    """(start label, start disabled, pause label, pause disabled) for the live
    run state — including a run started outside the dashboard (holds the study
    lock but is not ours to pause)."""
    if external_run_pid() is not None:
        return "Running outside dashboard", True, "Pause", True
    if optimize_running():
        return "Optimising…", True, "Pause", False
    try:
        has_study = context.project().db_path.exists()
    except Exception:
        has_study = False
    start_label = "Resume Optimisation" if has_study else "Start Optimisation"
    return start_label, False, "Pause", True


def register_run_callbacks(app) -> None:
    """Register weight-store, slider/pie sync, scene preview, run/pause, log."""

    app.clientside_callback(
        _WEIGHT_SYNC_JS,
        Output("opt-weights-store", "data"),
        Input({"type": "weight-slider", "test": ALL}, "value"),
        Input({"type": "test-check", "test": ALL}, "value"),
        Input("opt-equal-btn", "n_clicks"),
        Input("opt-normalize-btn", "n_clicks"),
        State({"type": "weight-slider", "test": ALL}, "id"),
        State("opt-weights-store", "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        _SLIDER_MIRROR_JS,
        Output({"type": "weight-slider", "test": ALL}, "value"),
        Input("opt-weights-store", "data"),
        State({"type": "weight-slider", "test": ALL}, "id"),
    )

    app.clientside_callback(
        _PIE_JS,
        Output("opt-pie", "figure"),
        Input("opt-weights-store", "data"),
        State({"type": "weight-slider", "test": ALL}, "id"),
        State({"type": "test-check", "test": ALL}, "value"),
    )

    app.clientside_callback(
        _WEIGHT_STATUS_JS,
        Output("opt-weight-status", "children"),
        Input("opt-weights-store", "data"),
        Input({"type": "test-check", "test": ALL}, "value"),
        State({"type": "weight-slider", "test": ALL}, "id"),
    )

    @app.callback(
        Output("run-scene-status", "children"),
        Input({"type": "scene-preview", "test": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def handle_preview(_n_clicks):
        return _do_scene_preview()

    @app.callback(
        Output("opt-status", "children"),
        Input("opt-start-btn", "n_clicks"),
        Input("opt-stop-btn", "n_clicks"),
        State({"type": "test-check", "test": ALL}, "value"),
        State({"type": "test-check", "test": ALL}, "id"),
        State({"type": "gate-check", "test": ALL}, "value"),
        State({"type": "gate-check", "test": ALL}, "id"),
        State("opt-weights-store", "data"),
        State("opt-sampler", "value"),
        State("opt-seed-sampler", "value"),
        State("opt-cmaes-margin", "value"),
        State("opt-n-parallel", "value"),
        State("opt-n-generations", "value"),
        State("opt-run-until-converged", "value"),
        State("opt-restart-patience", "value"),
        State("opt-prune-mode", "value"),
        State("opt-cmaes-restarts", "value"),
        State("opt-stall-generations", "value"),
        State("opt-inc-popsize", "value"),
        State("opt-restart-flags", "value"),
        prevent_initial_call=True,
    )
    def handle_optimise(
        _, __, check_vals, check_ids, gate_vals, gate_ids, store,
        sampler, seed_sampler, cmaes_margin, n_parallel, n_generations,
        run_until_converged, restart_patience, prune_mode,
        cmaes_restarts, stall_generations, inc_popsize, restart_flags,
    ):
        restarts = {
            "cmaes_restarts": cmaes_restarts,
            "stall_generations": stall_generations,
            "inc_popsize": inc_popsize,
            "flags": restart_flags,
        }
        return _do_run_or_pause(
            check_vals, check_ids, gate_vals, gate_ids, store,
            sampler, seed_sampler, cmaes_margin, n_parallel, n_generations,
            run_until_converged, restart_patience, prune_mode, restarts,
        )

    register_log_view(
        app,
        pre_id="run-log",
        interval_id="opt-interval",
        filter_id="run-log-filter",
        source=lambda: _read_proc_log("optimize"),
    )

    @app.callback(
        Output("opt-start-btn", "children"),
        Output("opt-start-btn", "disabled"),
        Output("opt-stop-btn", "children"),
        Output("opt-stop-btn", "disabled"),
        Input("opt-interval", "n_intervals"),
    )
    def refresh_run_buttons(_):
        return _current_button_state()
