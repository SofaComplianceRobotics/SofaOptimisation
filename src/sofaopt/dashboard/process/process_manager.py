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
        proc = subprocess.Popen(
            [sys.executable, str(script)],
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
    proc = _PROCS.get(name)
    if proc is None or proc.poll() is not None:
        return "Not running."
    try:
        proc.kill()
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
    """Launch the project's headless optimization run."""
    return _start_proc("optimize", context.project().run_script, env)


def stop_optimize() -> str:
    return _stop_proc("optimize")


def launch_scene(scene_file: Path, extra_env: dict | None = None, gui: str = "imgui") -> str:
    """Launch one scene in an interactive ``runSofa`` window for viewing."""
    project = context.project()
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
        proc = subprocess.Popen(cmd, env=env, cwd=str(project.work_dir))
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
