"""Low-level atomic file I/O and cross-process locking for state JSON files.

Fully generic — no project knowledge, stdlib-only (it is imported inside the
SofaPython3 scene processes as well as by the optimizer and the dashboard).

All shared-file updates must go through :func:`write_json` (atomic
write-temp-then-rename) or :func:`update_json_locked` (locked
read-modify-write). Scene processes can be SIGKILLed at any moment (that is how
``write_score`` and the timeout prune work), so a holder may die mid-lock:
:func:`_acquire_lock` therefore breaks locks older than ``_STALE_LOCK_S``.
"""

import errno
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A .lock older than this is assumed to belong to a dead process and is broken.
# Writers hold the lock only for one read-modify-replace (milliseconds), so a
# multi-second-old lock means the holder was killed while holding it.
_STALE_LOCK_S = 10.0


def _acquire_lock(lock_path: Path, timeout_s: float = 5.0) -> bool:
    """Acquire a cross-process file lock via exclusive create.

    Breaks stale locks left behind by killed holders. Returns False only on
    sustained live contention or a persistent OS error (both logged by the
    caller via :func:`update_json_locked`).
    """
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            _break_stale_lock(lock_path)
            if time.time() >= deadline:
                return False
            time.sleep(0.01)
        except OSError as e:
            logger.warning("Lock %s: unexpected error: %s", lock_path, e)
            if time.time() >= deadline:
                return False
            time.sleep(0.05)


def _break_stale_lock(lock_path: Path) -> None:
    try:
        age = time.time() - lock_path.stat().st_mtime
        if age > _STALE_LOCK_S:
            lock_path.unlink()
            logger.warning(
                "Broke stale lock %s (age %.1fs - holder likely died)", lock_path, age
            )
    except OSError:
        pass  # already gone, or lost the unlink race — the acquire loop retries


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except OSError as e:
        # Not fatal: the stale-lock breaker will reclaim it after _STALE_LOCK_S.
        logger.warning("Could not release lock %s: %s", lock_path, e)


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
    except Exception as final_exc:
        if last_exc:
            # Surface the retried failure; the final os.replace error stays
            # attached as the explicit cause.
            raise last_exc from final_exc
        raise


def write_json(path: Path, data: dict) -> None:
    """Atomically write a dict as pretty JSON (write-temp-then-rename)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _replace_with_retry(tmp, path)


def update_json_locked(path: Path, mutate: Callable[[dict], Any]) -> None:
    """Locked read-modify-write of a shared JSON file.

    ``mutate`` receives the parsed dict ({} if missing/corrupt) and modifies it
    in place; the result is written back atomically.

    If the lock cannot be acquired even after stale-break, the update proceeds
    *unlocked* with a loud log: certainly dropping this write (e.g. a run's
    final score) is worse than the small concurrent-writer race window — the
    file replace itself stays atomic either way.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    locked = _acquire_lock(lock_path)
    if not locked:
        logger.error(
            "Proceeding WITHOUT lock on %s (sustained contention) - "
            "a concurrent update may be lost",
            path,
        )
    try:
        data = _read_json_safe(path)
        mutate(data)
        write_json(path, data)
    finally:
        if locked:
            _release_lock(lock_path)
