"""Low-level atomic file I/O and cross-process locking for state JSON files.

Fully generic — no project knowledge. Used by the optimizer (writer side) and
may be reused by scenes (reader side) and the dashboard.
"""

import errno
import json
import os
import time
from pathlib import Path


def write_run_status(path: Path, data: dict) -> None:
    """Atomically write one status JSON file, avoiding partial/empty reads."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _replace_with_retry(tmp_path, path)


def _acquire_lock(lock_path: Path, timeout_s: float = 5.0) -> bool:
    """Acquire a simple cross-process file lock using exclusive create."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.01)
        except Exception:
            return False
    return False


def _release_lock(lock_path: Path) -> None:
    """Delete the file lock, silently ignoring errors."""
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def _read_json_safe(path: Path) -> dict:
    """Read a JSON file and return a dict, returning {} on any error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _replace_with_retry(tmp: Path, path: Path, timeout_s: float = 1.0) -> None:
    """Replace ``path`` with ``tmp``, retrying past transient Windows locks.

    On Windows, replacing a file can fail with PermissionError if another
    process briefly holds it open. Retry for a short timeout before raising.
    """
    deadline = time.time() + float(timeout_s)
    last_exc: Exception | None = None
    while True:
        try:
            tmp.replace(path)
            return
        except PermissionError as e:
            last_exc = e
            if time.time() >= deadline:
                break
            time.sleep(0.02)
            continue
        except OSError as e:
            last_exc = e
            if e.errno in (errno.EACCES, errno.EPERM) and time.time() < deadline:
                time.sleep(0.02)
                continue
            break
    try:
        os.replace(str(tmp), str(path))
        return
    except Exception:
        if last_exc:
            raise last_exc
        raise


def write_json(path: Path, data: dict) -> None:
    """Write a dict as pretty JSON."""
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
