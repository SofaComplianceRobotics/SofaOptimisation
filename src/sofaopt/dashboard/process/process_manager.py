"""Subprocess management: start/stop the optimizer and launch scenes for viewing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sofaopt.dashboard import context

# Running subprocesses keyed by role.
_PROCS: dict[str, subprocess.Popen | None] = {"optimize": None}


def _log_dir() -> Path:
    d = context.project().runtime_dir / "logs"
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
        log_path = _log_dir() / f"{name}.log"
        log_file = open(log_path, "w", encoding="utf-8")
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


def stop_optimize_and_wait(timeout_s: float = 15.0) -> bool:
    """Stop the run and WAIT for the process to exit (for stop-&-archive:
    the runtime dir must not be moved under a live process). True when gone."""
    from sofaopt.core.sofa_runner import kill_process_tree

    proc = _PROCS.get("optimize")
    if proc is None or proc.poll() is not None:
        return True
    try:
        kill_process_tree(proc)
        proc.wait(timeout=timeout_s)
    except Exception:
        pass
    if proc.poll() is not None:
        _PROCS["optimize"] = None
        return True
    return False


def launch_scene(scene_file: Path, extra_env: dict | None = None, gui: str = "imgui") -> str:
    """Launch one scene in an interactive ``runSofa`` window for viewing."""
    project = context.project()
    if project.runsofa_exe is None:
        return "runSofa not configured (set RUNSOFA_EXE or add runsofa_exe to the project)."
    runsofa = str(project.runsofa_exe)
    if not os.path.isfile(runsofa):
        return f"runSofa not found at: {runsofa}"
    env = project.scene_env()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    cmd = [runsofa]
    for plugin in project.sofa_plugins:
        cmd += ["-l", plugin]
    cmd += ["-g", gui, str(scene_file)]
    try:
        from sofaopt.core.sofa_runner import attach_process_to_sofa_job

        proc = subprocess.Popen(cmd, env=env, cwd=str(project.work_dir))
        # Viewer windows must not outlive the dashboard (kill-on-close job).
        # The headless optimize run is intentionally NOT attached: a long run
        # must survive closing the dashboard; it owns its own job for children.
        attach_process_to_sofa_job(proc)
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
