"""Runtime directory housekeeping: trials/ reset and generation numbering."""

from __future__ import annotations

import logging
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
