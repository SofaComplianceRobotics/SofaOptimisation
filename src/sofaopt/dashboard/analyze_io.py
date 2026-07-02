"""Dashboard adapters over the canonical record loader in sofaopt.core.results.

All score reading/ranking logic lives in :mod:`sofaopt.core.results` — the
dashboard must display exactly the scores the study recorded, never recompute
its own (dev_guidelines §11).
"""

from __future__ import annotations

from sofaopt.core import results
from sofaopt.dashboard import context


def load_all_trials() -> list[dict]:
    """Every trial record for the active project, ordered chronologically."""
    return results.load_trial_records(context.trials_dir())


def load_gen_summaries() -> list[dict]:
    """Generation summaries for the active project."""
    return results.load_gen_summaries(context.trials_dir())
