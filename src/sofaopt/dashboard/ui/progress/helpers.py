"""Progress-tab state helpers: labels, colors, per-run progress, canonical state."""

from sofaopt.dashboard import context
from sofaopt.dashboard.data.cache import _load_trial_state

_MAX_SCORE_CACHE: dict[str, float] = {}


def _get_test_max_score(test_name: str) -> float:
    """Configured max score for a test (defaults to 1.0)."""
    if not test_name:
        return 1.0
    if test_name in _MAX_SCORE_CACHE:
        return _MAX_SCORE_CACHE[test_name]
    try:
        spec = context.catalog().get(test_name)
        result = spec.max_score if spec else 1.0
    except Exception:
        result = 1.0
    _MAX_SCORE_CACHE[test_name] = result
    return result


# Friendly labels for raw run-state strings written into trial_state.json.
_RUN_STATE_LABELS = {
    "not-started": "queued",
    "queued": "queued",
    "pending": "gated — waiting for ungated run",
    "waiting-slot": "waiting for SOFA slot",
    "preparing": "preparing trial",
    "generating-geometry": "preparing trial",
    "rendering-preview": "rendering preview",
    "launching": "launching SOFA",
    "running": "running",
    "done": "done",
    "failed": "failed",
    "error": "error",
    "pruned": "pruned",
    "skipped": "skipped",
    "cancelled": "cancelled",
}

_ACTIVE_STATES = {"running", "launching", "preparing", "generating-geometry", "rendering-preview"}
_WAITING_STATES = {"waiting-slot"}


def _run_state_label(state: str) -> str:
    key = str(state or "").lower()
    return _RUN_STATE_LABELS.get(key, key.replace("-", " ") or "unknown")


def _state_color(state: str) -> str:
    state = state.lower()
    if state == "done":
        return "#2f9e44"
    if state in _ACTIVE_STATES:
        return "#0270ff"
    if state in _WAITING_STATES:
        return "#f08c00"
    if state in {"failed", "error", "pruned", "skipped", "cancelled"}:
        return "#e03131"
    return "#868e96"


def _weight_segment_style(segment: dict) -> dict:
    """Inline styles for one ladder segment (relaunchable-probe visualization)."""
    state = str(segment.get("state", "unknown")).lower()
    color = str(segment.get("color") or "#dee2e6")
    border = str(segment.get("border_color") or color)
    text = "#ffffff" if state in {"tested_success", "tested_failure", "pending"} else "#343a40"
    return {"background": color, "border": f"1px solid {border}", "color": text, "fontWeight": 700}


def _run_progress_pct(run: dict) -> float:
    """Frame-based progress percent for one run."""
    state = str(run.get("state", "")).lower()
    if state in {"done", "failed", "error", "pruned", "skipped", "cancelled"}:
        return 100.0
    current_frame = run.get("current_frame")
    total_frames = run.get("total_frames")
    if (
        isinstance(current_frame, (int, float))
        and isinstance(total_frames, (int, float))
        and total_frames > 0
    ):
        return max(0.0, min(100.0, 100.0 * float(current_frame) / float(total_frames)))
    return 0.0


def _get_live_score(run: dict) -> tuple[float | None, bool]:
    """Live score estimate for a run and whether it is final."""
    score = run.get("score")
    if isinstance(score, (int, float)):
        return float(score), True
    hold_time = run.get("hold_time")
    if isinstance(hold_time, (int, float)):
        return float(hold_time), False
    return None, False


def _get_trial_actual_state(trial_record: dict) -> str:
    """Canonical state for a trial, accounting for per-run states."""
    raw_state = _load_trial_state(trial_record)
    if raw_state is None and not trial_record.get("is_complete"):
        return "waiting"
    trial_state = raw_state or {}
    runs = trial_state.get("runs") if isinstance(trial_state.get("runs"), list) else []
    state = str(
        trial_state.get("state") or ("done" if trial_record.get("is_complete") else "running")
    ).lower()
    terminal = {"done", "failed", "error", "pruned", "skipped", "cancelled"}
    if (
        state not in terminal
        and runs
        and all(str(r.get("state", "")).lower() in terminal for r in runs if isinstance(r, dict))
    ):
        state = "done"
    return state


def _find_earliest_not_done(records: list[dict]) -> str | None:
    """DOM id of the earliest non-terminal trial card, for auto-scroll."""
    terminal = {"done", "failed", "error", "pruned", "skipped", "cancelled"}
    for record in records:
        if _get_trial_actual_state(record) not in terminal:
            return f"trial-card-{record.get('gen_index', 0):04d}-{record.get('trial_index', 0):04d}"
    return None
