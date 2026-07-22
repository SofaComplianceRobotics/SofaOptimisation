"""Restart-event log — a display-only record of IPOP restarts.

Each time :func:`sofaopt.core.restart.maybe_restart` swaps in a new CMA-ES it
appends a :class:`RestartEvent` here, so the dashboard can mark *where* a
restart happened on the convergence curve and show its values — without ever
touching ``study.db`` on its live-polled path (the dashboard reconstructs
state from files under ``trials_dir``, and this is one more such file).

**Not load-bearing.** ``RESTART_ATTR`` in the Optuna study's user attributes
stays the only thing control flow (resume, restart-budget guard) depends on.
``restarts.json`` is a derived, best-effort cache: a crash between
``study.set_user_attr`` and :func:`record_restart_event` loses one display
marker but never affects the run's correctness. Do not make anything read it
for a decision.

Lifecycle matches ``progress.json``: it lives in ``trials_dir``, is wiped by
``reset_trials_dir`` on a fresh start, kept on resume, and moved intact when a
run is archived.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sofaopt.core.io import _read_json_safe, update_json_locked

logger = logging.getLogger(__name__)

RESTARTS_FILENAME = "restarts.json"


@dataclass(frozen=True)
class RestartEvent:
    """One IPOP restart, positioned for annotation on the convergence curve.

    ``trial_chron`` is the completed-trial count at the restart boundary — the
    x coordinate the dashboard's performance graph uses (chronological trial
    index), so a marker lands between the stalled population and the grown one.
    """

    restart_index: int
    gen: int
    trial_chron: int
    old_popsize: int
    new_popsize: int
    sigma0: float
    incumbent_score: float | None
    timestamp: float
    incumbent_params: dict = field(default_factory=dict)


def restart_events_path(trials_dir: Path) -> Path:
    return Path(trials_dir) / RESTARTS_FILENAME


def load_restart_events(trials_dir: Path) -> list[dict]:
    """All recorded restart events, or ``[]`` when the file is missing/corrupt."""
    data = _read_json_safe(restart_events_path(trials_dir))
    events = data.get("restarts")
    return events if isinstance(events, list) else []


def record_restart_event(trials_dir: Path, event: RestartEvent) -> None:
    """Append ``event`` to ``restarts.json`` (locked read-modify-write).

    Idempotent by ``restart_index``: re-recording the same restart (e.g. a
    retried write) does not duplicate it, so the log stays one entry per
    restart.

    Best-effort: a write failure is logged and swallowed. The restart itself
    has already happened (study state mutated by the caller) — losing the
    display marker must never abort the run (see the module docstring).
    """
    def _append(data: dict) -> None:
        events = data.get("restarts")
        if not isinstance(events, list):
            events = []
        if any(
            isinstance(e, dict) and e.get("restart_index") == event.restart_index
            for e in events
        ):
            return
        events.append(asdict(event))
        data["restarts"] = events

    try:
        Path(trials_dir).mkdir(parents=True, exist_ok=True)
        update_json_locked(restart_events_path(trials_dir), _append)
    except Exception as exc:
        logger.warning("[restart] Could not record restart event (display-only): %s", exc)
