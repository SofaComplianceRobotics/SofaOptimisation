"""Runtime directory housekeeping: trials/ reset and generation numbering."""

from __future__ import annotations

import shutil
from pathlib import Path


def reset_trials_dir(trials_dir: Path, previews_dir: Path) -> None:
    """Wipe the trials directory and recreate it fresh, including previews."""
    if trials_dir.exists():
        shutil.rmtree(trials_dir)
        print(f"[reset] Cleared {trials_dir}")
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
