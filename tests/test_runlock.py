"""Single-run lock: acquire/refuse/stale/release (no SOFA needed)."""

import json
import os
import subprocess
import sys

from sofaopt.core.runlock import (
    acquire_run_lock,
    lock_holder,
    lock_path_for,
    pid_alive,
    release_run_lock,
)


def _dead_pid() -> int:
    """PID of a process that certainly exited (spawn one and wait for it)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_lock_sits_next_to_runtime_not_inside(tmp_path):
    # Archiving renames runtime/ away — a held lock must not travel with it.
    runtime = tmp_path / "runtime"
    assert lock_path_for(runtime).parent == tmp_path


def test_acquire_and_release(tmp_path):
    runtime = tmp_path / "runtime"
    lock = acquire_run_lock(runtime)
    assert lock is not None and lock.exists()
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    assert lock_holder(runtime) == os.getpid()
    release_run_lock(lock)
    assert not lock.exists()
    assert lock_holder(runtime) is None


def test_reacquire_by_same_process_is_allowed(tmp_path):
    runtime = tmp_path / "runtime"
    first = acquire_run_lock(runtime)
    second = acquire_run_lock(runtime)
    assert first == second and second is not None


def test_refuses_when_live_process_holds_it(tmp_path):
    runtime = tmp_path / "runtime"
    # The pytest parent process is alive and is not us: a genuine live holder.
    live_other = os.getppid()
    lock_path_for(runtime).parent.mkdir(parents=True, exist_ok=True)
    lock_path_for(runtime).write_text(json.dumps({"pid": live_other}))
    assert acquire_run_lock(runtime) is None
    assert lock_holder(runtime) == live_other


def test_stale_lock_is_replaced(tmp_path):
    runtime = tmp_path / "runtime"
    lock_path_for(runtime).parent.mkdir(parents=True, exist_ok=True)
    lock_path_for(runtime).write_text(json.dumps({"pid": _dead_pid()}))
    assert lock_holder(runtime) is None
    lock = acquire_run_lock(runtime)
    assert lock is not None
    assert json.loads(lock.read_text())["pid"] == os.getpid()


def test_corrupt_lock_file_is_replaced(tmp_path):
    runtime = tmp_path / "runtime"
    lock_path_for(runtime).parent.mkdir(parents=True, exist_ok=True)
    lock_path_for(runtime).write_text("not json {")
    assert acquire_run_lock(runtime) is not None


def test_release_does_not_remove_someone_elses_lock(tmp_path):
    runtime = tmp_path / "runtime"
    path = lock_path_for(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getppid()}))
    release_run_lock(path)
    assert path.exists()


def test_pid_alive():
    assert pid_alive(os.getpid())
    assert not pid_alive(_dead_pid())
    assert not pid_alive(0)
    assert not pid_alive(-5)
