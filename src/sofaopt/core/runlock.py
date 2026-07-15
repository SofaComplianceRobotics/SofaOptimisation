"""Single-instance run lock: at most one optimizer per study.

Two ``run_optimization`` processes sharing one study.db corrupt each other:
the second one's resume recovery marks the first one's in-flight RUNNING
trials FAILED, and the first then dies on ``study.tell`` ("Cannot tell a
FAIL trial"). The lock makes the second process refuse to start instead.

The lock file sits NEXT TO runtime/ (not inside it): a fresh start archives
runtime/ away, which must not carry a held lock along. A lock whose PID is
no longer alive is stale (crash / hard kill) and is silently replaced.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

from pathlib import Path

logger = logging.getLogger(__name__)


def pid_alive(pid: int) -> bool:
    """True when a process with this PID is currently running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # NOT os.kill(pid, 0): on Windows that TERMINATES the process.
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX: signal 0 = existence probe, no signal sent
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_path_for(runtime_dir: Path) -> Path:
    runtime_dir = Path(runtime_dir)
    return runtime_dir.parent / (runtime_dir.name + ".lock")


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def acquire_run_lock(runtime_dir: Path) -> Path | None:
    """Take the lock, or return ``None`` when another LIVE run holds it."""
    path = lock_path_for(runtime_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        holder = _read(path)
        pid = int(holder.get("pid", 0) or 0)
        if pid and pid != os.getpid() and pid_alive(pid):
            logger.error(
                f"[lock] Another optimization is already running this study "
                f"(PID {pid}, since {holder.get('started', '?')}) — refusing to start.\n"
                f"[lock] Pause/kill it first, or delete {path} if you are sure it is dead."
            )
            return None
    path.write_text(
        json.dumps(
            {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        ),
        encoding="utf-8",
    )
    return path


def release_run_lock(path: Path | None) -> None:
    """Remove the lock if this process is the holder (no-op otherwise)."""
    if path is None:
        return
    try:
        if int(_read(path).get("pid", 0) or 0) == os.getpid():
            path.unlink()
    except OSError:
        pass


def lock_holder(runtime_dir: Path) -> int | None:
    """PID of a live run currently holding the lock, else ``None``."""
    path = lock_path_for(runtime_dir)
    if not path.exists():
        return None
    pid = int(_read(path).get("pid", 0) or 0)
    return pid if pid and pid_alive(pid) else None
