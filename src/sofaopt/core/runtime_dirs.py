"""Runtime directory housekeeping: trials/ reset and generation numbering."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_console_logging() -> None:
    """Attach a plain console handler to the sofaopt logger tree.

    Called by the entry points (``run_optimization``, ``launch_dashboard``) —
    never by library modules — and only when the user hasn't configured
    logging themselves. The bare ``%(message)s`` format preserves the
    familiar ``[tag] message`` console output.
    """
    root = logging.getLogger("sofaopt")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_RUN_LOG_NAME = "optimize.log"


def attach_run_log_file(logs_dir: Path, *, truncate: bool) -> Path:
    """Tee the ``sofaopt`` logger to ``logs_dir/optimize.log`` so any launch
    path — the ``sofaopt`` CLI, a project ``run.py``, or the dashboard's Run
    button — produces the one file the dashboard log window tails.

    The format carries the level (``HH:MM:SS LEVEL message``) so the log
    viewer can filter Info/Warn/Error; the console handler keeps its bare
    ``%(message)s`` look. Idempotent: a second call for the same path is a
    no-op, so re-entering ``run_optimization`` in one process won't double-log.

    One active run log per process: a call for a *different* path closes the
    previous run-log handler first (production runs one campaign per process, so
    the only caller that hits this is a test driving several runs — leaving old
    handlers open would pin their now-deleted temp dirs open on Windows).

    ``truncate`` starts a fresh file (a brand-new run); resume appends. The log
    lives outside ``runtime/`` (see :meth:`SofaOptProject.logs_dir`), so it is
    not moved by archiving and must be reset here rather than by that move.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / _RUN_LOG_NAME
    root = logging.getLogger("sofaopt")
    # Match logging.FileHandler's own normalization (it stores os.path.abspath);
    # normcase folds Windows case so a second call is reliably a no-op.
    target = os.path.normcase(os.path.abspath(path))
    for h in list(root.handlers):
        base = getattr(h, "baseFilename", None)
        if not (isinstance(h, logging.FileHandler) and base is not None):
            continue
        # Only touch run-log handlers we created (basename optimize.log), never
        # a FileHandler a user attached for their own purposes.
        if os.path.basename(base) != _RUN_LOG_NAME:
            continue
        if os.path.normcase(base) == target:
            return path
        root.removeHandler(h)
        h.close()
    handler = logging.FileHandler(path, mode="w" if truncate else "a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return path


def reset_trials_dir(trials_dir: Path, previews_dir: Path) -> None:
    """Wipe the trials directory and recreate it fresh, including previews."""
    if trials_dir.exists():
        shutil.rmtree(trials_dir)
        logger.info(f"[reset] Cleared {trials_dir}")
    trials_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True, exist_ok=True)


def last_gen_index(trials_dir: Path) -> int:
    """Highest generation number present on disk (0 when none).

    The resume source of truth: counting COMPLETE Optuna trials under-counts
    when a previous run was killed mid-generation or trials were pruned, which
    made a resumed run reuse — and overwrite — existing gen_XXXX directories.
    """
    last = 0
    for d in trials_dir.glob("gen_*"):
        try:
            last = max(last, int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return last
