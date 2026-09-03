"""Subprocess management: start/stop the optimizer and launch scenes for viewing."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

from sofaopt.dashboard import context

# Running subprocesses keyed by role.
_PROCS: dict[str, subprocess.Popen | None] = {"optimize": None}


def _log_dir() -> Path:
    d = context.project().logs_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _proc_running(name: str) -> bool:
    proc = _PROCS.get(name)
    return proc is not None and proc.poll() is None


def _start_proc(name: str, script: Path, env: dict | None = None) -> str:
    """Start a background subprocess for ``name`` running ``script``."""
    if _proc_running(name):
        return f"Already running (PID {_PROCS[name].pid})."
    if script is None:
        return "No run_script configured on the project (dashboard is read-only)."
    try:
        # The optimizer tees its own structured, level-tagged log to
        # {logs_dir}/optimize.log (see core.runtime_dirs.attach_run_log_file),
        # which the dashboard tails. The child's raw stdout/stderr goes to a
        # separate .console.log — a crash / pre-logging fallback only, so the
        # tailed log never carries duplicated lines.
        console_path = _log_dir() / f"{name}.console.log"
        log_file = open(console_path, "w", encoding="utf-8")  # noqa: SIM115  # handle feeds the child process and must outlive this function
        run_env = env if env is not None else os.environ.copy()
        run_env["PYTHONIOENCODING"] = "utf-8"
        python_exe = context.project().run_python_exe
        if python_exe is None or not os.path.isfile(str(python_exe)):
            python_exe = sys.executable
        proc = subprocess.Popen(
            [str(python_exe), str(script)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(script).parent),
            env=run_env,
        )
        _PROCS[name] = proc
        return f"Started (PID {proc.pid})."
    except Exception as exc:
        return f"Error starting process: {exc}"


def _stop_proc(name: str) -> str:
    from sofaopt.core.sofa_runner import kill_process_tree

    proc = _PROCS.get(name)
    if proc is None or proc.poll() is not None:
        return "Not running."
    try:
        kill_process_tree(proc)
        proc.wait(timeout=10)
        _PROCS[name] = None
        return "Stopped."
    except Exception as exc:
        return f"Error stopping process: {exc}"


def _read_proc_log(name: str, tail: int = 150) -> str:
    log_path = _log_dir() / f"{name}.log"
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])
    except Exception:
        return ""


def start_optimize(env: dict | None = None) -> str:
    """Launch the project's headless optimization run (auto-resumes a study)."""
    if not _proc_running("optimize"):
        # A run started OUTSIDE the dashboard (CLI/script) holds the run lock;
        # launching a second orchestrator on the same study corrupts both.
        from sofaopt.core.runlock import lock_holder

        pid = lock_holder(context.project().runtime_dir)
        if pid:
            return (
                f"An optimization for this study is already running outside the "
                f"dashboard (PID {pid}) — stop it first."
            )
    return _start_proc("optimize", context.project().run_script, env)


def stop_optimize() -> str:
    """Stop the run — this is a PAUSE: the study resumes from the last
    completed generation, and interrupted trials are re-enqueued."""
    msg = _stop_proc("optimize")
    if msg == "Stopped.":
        return ("Paused — Resume continues from the last completed generation "
                "(interrupted trials are re-enqueued).")
    return msg


def optimize_running() -> bool:
    """True while the dashboard-launched optimization process is alive."""
    return _proc_running("optimize")


def external_run_pid() -> int | None:
    """PID of an optimizer holding this study's run lock but NOT spawned by the
    dashboard (a ``sofaopt`` CLI / ``run.py`` run), else None. The dashboard can
    monitor such a run (it tails the same log) but cannot pause it."""
    from sofaopt.core.runlock import lock_holder

    if _proc_running("optimize"):
        return None
    return lock_holder(context.project().runtime_dir)


def run_is_active() -> bool:
    """True when any optimizer is running on this study — the dashboard's own
    child or an external CLI/script run (via the run lock). Use this to gate
    actions that are unsafe during a live run (e.g. archiving moves runtime/)."""
    return _proc_running("optimize") or external_run_pid() is not None


def stop_optimize_and_wait(timeout_s: float = 15.0) -> bool:
    """Stop the run and WAIT for the process to exit (for stop-&-archive:
    the runtime dir must not be moved under a live process). True when gone."""
    from sofaopt.core.sofa_runner import kill_process_tree

    proc = _PROCS.get("optimize")
    if proc is None or proc.poll() is not None:
        return True
    # Best-effort stop; the poll() below decides the outcome either way.
    with contextlib.suppress(Exception):
        kill_process_tree(proc)
        proc.wait(timeout=timeout_s)
    if proc.poll() is not None:
        _PROCS["optimize"] = None
        return True
    return False


def _viewer_cmd(project, scene_file: Path, gui: str) -> list[str]:
    """runSofa command for an interactive viewer. ``-g imgui`` only works when
    the plugin registering that GUI is loaded, so SofaImGui is appended when
    missing — same rule as the Results tab's "Test it" launcher."""
    plugins = list(project.sofa_plugins)
    if gui == "imgui" and "SofaImGui" not in plugins:
        plugins.append("SofaImGui")
    cmd = [str(project.runsofa_exe)]
    for plugin in plugins:
        cmd += ["-l", plugin]
    cmd += ["-g", gui, str(scene_file)]
    return cmd


def launch_scene(scene_file: Path, extra_env: dict | None = None, gui: str = "imgui") -> str:
    """Launch one scene in an interactive ``runSofa`` window for viewing."""
    project = context.project()
    if project.runsofa_exe is None:
        return "runSofa not configured (set RUNSOFA_EXE or add runsofa_exe to the project)."
    if not os.path.isfile(str(project.runsofa_exe)):
        return f"runSofa not found at: {project.runsofa_exe}"
    env = project.scene_env()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    try:
        from sofaopt.core.sofa_runner import attach_process_to_sofa_job

        log_path = _log_dir() / "preview.log"
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115  # handle feeds the child process and must outlive this function
        proc = subprocess.Popen(
            _viewer_cmd(project, scene_file, gui),
            env=env,
            cwd=str(project.work_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        # Viewer windows must not outlive the dashboard (kill-on-close job).
        # The headless optimize run is intentionally NOT attached: a long run
        # must survive closing the dashboard; it owns its own job for children.
        attach_process_to_sofa_job(proc)
        # A bad GUI/plugin/scene makes runSofa exit within moments while the
        # PID looked fine — reporting "Launched" then no window ever appearing
        # is a silent failure (observed with -g imgui minus SofaImGui). Give it
        # a moment and surface the log instead.
        time.sleep(1.5)
        if proc.poll() is not None:
            tail = _read_proc_log("preview", tail=8) or "(empty log)"
            return (
                f"SOFA exited immediately (code {proc.returncode}) — "
                f"see {log_path}. Log tail:\n{tail}"
            )
        return f"Launched SOFA (PID {proc.pid})."
    except Exception as exc:
        return f"Failed to launch: {exc}"


def load_config_text() -> str:
    """Return the project's config file text for the Config tab (or '')."""
    cfg = context.project().config_file
    if cfg is None:
        return ""
    try:
        return Path(cfg).read_text(encoding="utf-8")
    except Exception:
        return "{}"
