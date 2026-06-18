"""Directory and file housekeeping for the optimization run."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

# Quiet by default so cleanup logs don't interleave with the live progress bar.
PRINT_CLEANUP_LOGS = False


def reset_trials_dir(trials_dir: Path, previews_dir: Path) -> None:
    """Wipe the trials directory and recreate it fresh, including previews."""
    if trials_dir.exists():
        shutil.rmtree(trials_dir)
        print(f"[reset] Cleared {trials_dir}")
    trials_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True, exist_ok=True)


def delete_after_delay(path: Path, delay: float) -> None:
    """Delete a file after ``delay`` seconds in a background daemon thread."""

    def _delete():
        time.sleep(delay)
        try:
            path.unlink()
            if PRINT_CLEANUP_LOGS:
                print(f"[cleanup] Deleted {path.name}")
        except FileNotFoundError:
            pass

    threading.Thread(target=_delete, daemon=True).start()


def cleanup_assets(assets_by_trial: dict[int, Path]) -> None:
    """Delete per-trial prepared asset files (e.g. collision meshes)."""
    for asset in assets_by_trial.values():
        try:
            if asset and Path(asset).exists():
                Path(asset).unlink()
                if PRINT_CLEANUP_LOGS:
                    print(f"[cleanup] Deleted {Path(asset).name}")
        except Exception:
            pass
