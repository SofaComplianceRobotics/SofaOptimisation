"""Dashboard callbacks for trial inspection and summary video generation."""

from __future__ import annotations

import dataclasses
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from sofaopt.dashboard import context


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cached_trial_url(gen_name: str, trial_name: str) -> str | None:
    mp4 = context.trials_dir() / gen_name / trial_name / "trial.mp4"
    if mp4.exists():
        return f"/trial-video/{gen_name}/{trial_name}"
    return None


def _spawn_test_run(project, trial_dir: Path) -> None:
    """Launch runSofa in GUI mode for a completed trial (fire and forget).

    Calls prepare_trial in-process (if defined) so scene assets (meshes, configs)
    are ready, then starts runSofa with OPT_PARAMS_PATH pointing at the trial's
    params.json.  OPT_TRIAL_STATE_PATH is intentionally absent so the scene sees
    is_optimizing=False and won't call os.kill() after scoring.
    """
    import json as _json
    import shutil as _shutil

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
            print(f"[dashboard] prepare_trial for test run failed: {exc}")

    scene_file = project.tests[0].scene_file
    plugins = list(project.sofa_plugins)
    if "SofaImGui" not in plugins:
        plugins.append("SofaImGui")
    cmd = [str(project.runsofa_exe)]
    for plugin in plugins:
        cmd += ["-l", plugin]
    cmd += [str(scene_file)]

    env = project.scene_env()  # os.environ + project.sofa_env
    env.update(extra_env)
    env["OPT_PARAMS_PATH"] = str(params_path)

    subprocess.Popen(cmd, env=env)


def _spawn_summary_gen(project, summary_path: Path) -> subprocess.Popen:
    """Spawn summary video generation as a background subprocess."""
    stripped = dataclasses.replace(
        project,
        prepare_trial=None,
        constrain_params=None,
        on_generation_end=None,
    )
    tmp_dir = Path(tempfile.mkdtemp(prefix="sofaopt_sumgen_"))
    pkl_file = tmp_dir / "project.pkl"
    pkl_file.write_bytes(pickle.dumps(stripped))

    script = tmp_dir / "gen.py"
    script.write_text(
        f"""
import os, sys, pickle, shutil
from pathlib import Path
print("[summary-gen] Script started", flush=True)

try:
    _sofa_root = os.environ.get('SOFA_ROOT', '') or os.environ.get('SOFAPYTHON3_ROOT', '')
    if not _sofa_root:
        for _pp in os.environ.get('PYTHONPATH', '').split(os.pathsep):
            _pp = _pp.strip()
            if not _pp:
                continue
            _pp_path = Path(_pp)
            try:
                _has_sofa = (_pp_path / 'Sofa').is_dir() or bool(list(_pp_path.glob('Sofa*.pyd'))[:1])
            except (OSError, PermissionError):
                _has_sofa = False
            if _has_sofa:
                for _up in [_pp_path.parent.parent.parent, _pp_path.parent.parent]:
                    if (_up / 'bin').is_dir() or (_up / 'bin' / 'Release').is_dir():
                        _sofa_root = str(_up)
                        break
            if _sofa_root:
                break
    if _sofa_root and os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        for _d in [Path(_sofa_root) / 'bin' / 'Release', Path(_sofa_root) / 'bin']:
            if _d.is_dir():
                os.add_dll_directory(str(_d))
                print(f"[summary-gen] DLL dir registered: {{_d}}", flush=True)
                break
    if _sofa_root and not os.environ.get('SOFA_ROOT'):
        os.environ['SOFA_ROOT'] = _sofa_root
        print(f"[summary-gen] Set SOFA_ROOT = {{_sofa_root}}", flush=True)
    import importlib.util as _ilu
    if _ilu.find_spec('Sofa') is None and _sofa_root:
        for _sp in [
            Path(_sofa_root) / 'lib' / 'python3' / 'site-packages',
            Path(_sofa_root) / 'lib' / f'python{{sys.version_info.major}}.{{sys.version_info.minor}}' / 'site-packages',
        ]:
            if _sp.is_dir():
                sys.path.insert(0, str(_sp))
                print(f"[summary-gen] Added to sys.path: {{_sp}}", flush=True)
                break
    print(f"[summary-gen] SOFA root: {{_sofa_root or '(not found)'}}", flush=True)
except Exception as _e:
    print(f"[summary-gen] Bootstrap warning: {{_e}}", flush=True)

project = pickle.loads(Path({str(pkl_file)!r}).read_bytes())
print("[summary-gen] Project loaded, generating summary ...", flush=True)
from sofaopt.video import generate_summary_video
generate_summary_video(
    project,
    {str(summary_path)!r},
    top_n=project.record_summary_top_n,
    bottom_n=project.record_summary_bottom_n,
    text_overlay=True,
)
print("[summary-gen] Done.", flush=True)
shutil.rmtree({str(tmp_dir)!r}, ignore_errors=True)
# Avoid SOFA Py_FinalizeEx crash on Windows (same fix as scene/runner.py).
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update({k: str(v) for k, v in project.sofa_env.items()})
    log_path = summary_path.parent / "summary_gen.log"
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    _popen_kwargs: dict = {"stdin": subprocess.DEVNULL}
    if os.name == "nt":
        _popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        **_popen_kwargs,
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

def register_video_callbacks(app) -> None:
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
                lines = [l for l in log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines() if l.strip()]
                error_msg = "\n".join(lines[-3:])
            except Exception:
                error_msg = ""
            failed_job = {**job, "status": "error", "error_msg": error_msg}
            return _summary_status_children(failed_job, summary_path), failed_job

        raise PreventUpdate
