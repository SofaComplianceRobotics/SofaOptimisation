"""Dashboard callbacks for trial inspection and summary video generation."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from sofaopt.dashboard import context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cached_trial_url(gen_name: str, trial_name: str) -> str | None:
    mp4 = context.trials_dir() / gen_name / trial_name / "trial.mp4"
    if mp4.exists():
        return f"/trial-video/{gen_name}/{trial_name}"
    return None


def _test_scene_file(project, trial_dir: Path) -> Path:
    """Scene to open for a trial: the one its first run actually used.

    Falls back to the project's first test when the trial state is unreadable
    (e.g. a legacy runtime dir).
    """
    from sofaopt.core.trial_state import read_trial_run

    run0 = read_trial_run(trial_dir / "trial_state.json", 1) or {}
    test_name = str(run0.get("test_name", ""))
    for t in project.tests:
        if t.name == test_name:
            return t.scene_file
    return project.tests[0].scene_file


def _spawn_test_run(project, trial_dir: Path) -> None:
    """Launch runSofa in GUI mode for a completed trial (fire and forget).

    Calls prepare_trial in-process (if defined) so scene assets (meshes, configs)
    are ready, then starts runSofa with OPT_PARAMS_PATH pointing at the trial's
    params.json.  OPT_TRIAL_STATE_PATH is intentionally absent so the scene sees
    is_optimizing=False and won't call os.kill() after scoring.
    """
    import json as _json
    import shutil as _shutil

    from sofaopt.core import envkeys
    from sofaopt.core.sofa_runner import attach_process_to_sofa_job

    params_path = trial_dir / "params.json"
    extra_env: dict[str, str] = {}
    if project.prepare_trial is not None and params_path.exists():
        params = _json.loads(params_path.read_text(encoding="utf-8"))
        prep_dir = trial_dir / "_test_prep"
        _shutil.rmtree(prep_dir, ignore_errors=True)
        prep_dir.mkdir(parents=True)
        (prep_dir / "params.json").write_text(
            _json.dumps(params, indent=2), encoding="utf-8"
        )
        try:
            prep = project.prepare_trial(params, prep_dir)
            if prep is not None:
                extra_env = {k: str(v) for k, v in prep.env.items()}
        except Exception as exc:
            logger.info(f"[dashboard] prepare_trial for test run failed: {exc}")

    scene_file = _test_scene_file(project, trial_dir)
    plugins = list(project.sofa_plugins)
    if "SofaImGui" not in plugins:
        plugins.append("SofaImGui")
    cmd = [str(project.runsofa_exe)]
    for plugin in plugins:
        cmd += ["-l", plugin]
    cmd += ["-g", "imgui", str(scene_file)]

    env = project.scene_env()  # os.environ + project.sofa_env
    env.update(extra_env)
    env[envkeys.PARAMS_PATH] = str(params_path)

    proc = subprocess.Popen(cmd, env=env)
    # Kill-on-close job: the viewer must not outlive the dashboard (§ no
    # orphaned processes — every spawn path shares the same lifecycle rule).
    attach_process_to_sofa_job(proc)


def _spawn_summary_gen(project, summary_path: Path) -> subprocess.Popen:
    """Spawn summary video generation as a background subprocess.

    The project is handed over as JSON (hooks dropped by serialization) to a
    real entry point — ``python -m sofaopt.video.summary_entry`` — which owns
    the SOFA bootstrap and cleans the args file up itself. No pickle (§9),
    no generated script, no temp-dir leak.
    """
    import json as _json

    from sofaopt.project import project_to_jsonable

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    args_file = summary_path.parent / "summary_gen_args.json"
    args_file.write_text(
        _json.dumps(
            {
                "project": project_to_jsonable(project),
                "summary_path": str(summary_path),
                "top_n": project.record_summary_top_n,
                "bottom_n": project.record_summary_bottom_n,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update({k: str(v) for k, v in project.sofa_env.items()})
    log_path = summary_path.parent / "summary_gen.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115  # handle feeds the child process and must outlive this function
    popen_kwargs: dict = {"stdin": subprocess.DEVNULL}
    if os.name == "nt":
        # CREATE_NEW_PROCESS_GROUP: keep Ctrl+C in the dashboard terminal from
        # propagating into the render child; CREATE_NO_WINDOW: no console flash.
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    proc = subprocess.Popen(
        [sys.executable, "-m", "sofaopt.video.summary_entry", str(args_file)],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        **popen_kwargs,
    )
    log_fh.close()
    return proc


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _summary_status_children(job: dict, summary_path: Path):
    """Build the children for the summary-status-text span."""
    if summary_path.exists():
        return [
            html.A(
                "View Summary",
                href="/runtime-summary",
                target="_blank",
                className="text-success fw-semibold me-2",
            )
        ]
    status = (job or {}).get("status")
    if status == "running":
        return [html.Span("Generating summary...", className="text-warning")]
    if status == "error":
        err = (job or {}).get("error_msg", "")
        msg = f"Failed: {err}" if err else "Generation failed"
        log = str(summary_path.parent / "summary_gen.log")
        return [
            html.Span(msg, className="text-danger fw-semibold"),
            html.Span(f" (log: {log})", className="text-muted small"),
        ]
    return [html.Span("No summary yet", className="text-muted")]


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_video_callbacks(app) -> None:  # noqa: C901  # Dash registrar: total is the sum of its small nested callbacks; the flat registration list reads best in one place
    """Register all video-related dashboard callbacks."""

    # -- 1. Track which trial is selected ------------------------------------
    @app.callback(
        Output("selected-trial-store", "data"),
        Output("video-controls-panel", "style"),
        Output("video-status-text", "children"),
        Input("performance-graph", "clickData"),
    )
    def on_trial_selected(click_data):
        if not click_data:
            return {}, {"display": "none"}, []
        try:
            point = click_data["points"][0]
            cd = point.get("customdata")
            if not cd or len(cd) < 3:
                return {}, {"display": "none"}, []
            gen_name, trial_name = cd[1], cd[2]
            if not gen_name or not trial_name:
                return {}, {"display": "none"}, []
        except Exception:
            return {}, {"display": "none"}, []

        cached_url = _cached_trial_url(gen_name, trial_name)
        status = (
            [html.A("View recording", href=cached_url, target="_blank",
                    className="text-success fw-semibold me-2")]
            if cached_url else []
        )
        return (
            {"gen_name": gen_name, "trial_name": trial_name},
            {"display": "block"},
            status,
        )

    # -- 2. Test it button ---------------------------------------------------
    @app.callback(
        Output("video-status-text", "children", allow_duplicate=True),
        Input("test-it-btn", "n_clicks"),
        State("selected-trial-store", "data"),
        prevent_initial_call=True,
    )
    def trigger_test_it(n_clicks, selected):
        if not n_clicks or not selected:
            raise PreventUpdate
        gen_name = selected.get("gen_name", "")
        trial_name = selected.get("trial_name", "")
        if not gen_name or not trial_name:
            raise PreventUpdate

        project = context.project()
        trial_dir = context.trials_dir() / gen_name / trial_name
        try:
            _spawn_test_run(project, trial_dir)
            return [html.Span(
                "runSofa launched — check desktop for GUI window.",
                className="text-success",
            )]
        except Exception as exc:
            return [html.Span(f"Failed to launch: {exc}", className="text-danger")]

    # -- 3. Trigger summary generation ----------------------------------------
    @app.callback(
        Output("summary-job-store", "data"),
        Output("summary-status-text", "children"),
        Input("summary-gen-btn", "n_clicks"),
        State("summary-job-store", "data"),
        prevent_initial_call=True,
    )
    def trigger_summary_gen(n_clicks, job):
        if not n_clicks:
            raise PreventUpdate
        project = context.project()
        summary_path = project.runtime_dir / "summary.mp4"

        if summary_path.exists():
            return {"status": "done"}, _summary_status_children({}, summary_path)

        job = job or {}
        if job.get("status") == "running" and _process_alive(job.get("pid", 0)):
            return job, _summary_status_children(job, summary_path)

        try:
            proc = _spawn_summary_gen(project, summary_path)
            job = {"status": "running", "pid": proc.pid}
        except Exception as exc:
            job = {"status": "error"}
            return job, [html.Span(f"Failed to start: {exc}", className="text-danger")]

        return job, _summary_status_children(job, summary_path)

    # -- 4. Poll summary job + initial status ---------------------------------
    @app.callback(
        Output("summary-status-text", "children", allow_duplicate=True),
        Output("summary-job-store", "data", allow_duplicate=True),
        Input("video-poll-interval", "n_intervals"),
        State("summary-job-store", "data"),
        prevent_initial_call=True,
    )
    def poll_summary_status(_n, job):
        project = context.project()
        summary_path = project.runtime_dir / "summary.mp4"

        if summary_path.exists():
            new_job = {"status": "done"}
            if (job or {}).get("status") == "done":
                raise PreventUpdate
            return _summary_status_children(new_job, summary_path), new_job

        job = job or {}
        if job.get("status") != "running":
            raise PreventUpdate

        pid = job.get("pid", 0)
        if not _process_alive(pid):
            log_path = summary_path.parent / "summary_gen.log"
            try:
                lines = [ln for ln in log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines() if ln.strip()]
                error_msg = "\n".join(lines[-3:])
            except Exception:
                error_msg = ""
            failed_job = {**job, "status": "error", "error_msg": error_msg}
            return _summary_status_children(failed_job, summary_path), failed_job

        raise PreventUpdate
