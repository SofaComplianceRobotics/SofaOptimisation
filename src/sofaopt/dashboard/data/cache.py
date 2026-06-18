"""Short-lived caching of trial records and generation summaries."""

import json
import time
from pathlib import Path

from sofaopt.dashboard import context

_DATA_CACHE: dict = {"records": [], "summaries": [], "last_load": 0.0}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_trial_state(trial_record: dict) -> dict | None:
    """Load the trial_state.json for one trial record, or None if missing."""
    trial_path = (
        context.trials_dir()
        / trial_record.get("gen_name", "")
        / trial_record.get("trial_name", "")
        / "trial_state.json"
    )
    if not trial_path.exists():
        return None
    return _read_json(trial_path)


def _load_data():
    """Return cached (records, summaries), reloading when the cache expires."""
    from sofaopt.dashboard.analyze_io import load_all_trials, load_gen_summaries

    try:
        now = time.time()
        if _DATA_CACHE.get("records") and (
            now - float(_DATA_CACHE.get("last_load", 0))
        ) < max(0.5, float(context.LIVE_REFRESH_SECONDS)):
            return _DATA_CACHE["records"], _DATA_CACHE["summaries"]

        records = load_all_trials()
        summaries = load_gen_summaries()
        _DATA_CACHE["records"] = records
        _DATA_CACHE["summaries"] = summaries
        _DATA_CACHE["last_load"] = now
        return records, summaries
    except Exception as exc:
        print(f"[warn] Error loading data: {exc}")
        return (
            _DATA_CACHE.get("records", []) or [],
            _DATA_CACHE.get("summaries", []) or [],
        )


def _current_generation_records(records: list[dict]) -> list[dict]:
    """Records belonging to the most recent generation, ordered by trial index."""
    if not records:
        return []
    current_gen = max((r.get("gen_index", -1) for r in records), default=-1)
    if current_gen < 0:
        return []
    result = [r for r in records if r.get("gen_index", -1) == current_gen]
    return sorted(result, key=lambda r: r.get("trial_index", 0))
