"""Multi-fidelity step pruning: successive halving on the simulation-step prefix.

At each rung ``(step, keep_fraction)`` of the prunable test's schedule, once
every non-terminal run in the generation has reached ``step`` (or finished),
the candidates are ranked by their anytime ``partial_score`` and only the top
``ceil(population * keep_fraction)`` keep running. In ``"kill"`` mode the rest
have their SOFA processes terminated (the existing ``prune_trial`` path — they
finalize as PRUNED); in ``"shadow"`` mode the decision is only logged and
marked on the run slot, so a campaign can validate the machinery risk-free.

Rung synchronization makes the rule CMA-ES-safe (design doc §5): the ranking
is within one generation, only bottom-ranked candidates are stopped, and the
schedule's last keep fraction should leave the top mu fully evaluated. Before
each rung's kills, every live candidate's partial score is reported to its
Optuna trial (``trial.report(value, step)``) so a pruned trial carries its
last partial as an intermediate value — which
``CmaEsSampler(consider_pruned_trials=True)`` feeds back into the update.

A wedged run that never reaches a rung step does not deadlock the rung: the
wall-clock timeout eventually prunes it (terminal), which unblocks readiness.

Design + measured calibration: ``docs/design/multi-fidelity.md``; the trace
replay that produced the rung schedules: ``examples/prefix_pruning_study/``.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field

from sofaopt.core.generation.plan import prune_trial
from sofaopt.core.generation.types import LaunchedTrial
from sofaopt.core.runconfig import RunConfig
from sofaopt.core.trial_state import read_trial_run, update_trial_run

logger = logging.getLogger(__name__)

# Run-slot states that end a run's participation in rung decisions.
_TERMINAL = {"done", "failed", "error", "pruned", "skipped", "cancelled", "interrupted"}


@dataclass
class _Candidate:
    """One trial's standing at a rung: its ranking value and liveness."""

    entry: LaunchedTrial
    value: float
    alive: bool


@dataclass
class GenerationPruner:
    """One generation's rung scheduler (created by :func:`build_pruner`)."""

    cfg: RunConfig
    gen_index: int
    entries: list[LaunchedTrial]
    run_slot: int
    rungs: tuple[tuple[int, float], ...]
    fired: set[int] = field(default_factory=set)
    disabled: bool = False

    def check(self) -> None:
        """Fire every rung whose readiness condition holds (called each scan pass)."""
        for index, (step, keep_fraction) in enumerate(self.rungs):
            if self.disabled:
                return
            if index in self.fired:
                continue
            candidates = self._snapshot(step)
            if candidates is None:
                return  # rungs are ascending: later ones cannot be ready either
            self.fired.add(index)
            self._fire(step, keep_fraction, candidates)

    # -- readiness -------------------------------------------------------------

    def _snapshot(self, step: int) -> list[_Candidate] | None:
        """Every trial's ranking value at ``step``; None while any run lags."""
        candidates = []
        for entry in self.entries:
            run = read_trial_run(entry.trial_state_path, self.run_slot) or {}
            state = str(run.get("state", "")).lower()
            if state in _TERMINAL:
                raw = run.get("score")
                value = float(raw) if isinstance(raw, (int, float)) else float("-inf")
                candidates.append(_Candidate(entry, value, alive=False))
                continue
            if int(run.get("current_frame") or 0) < step:
                return None
            partial = run.get("partial_score")
            if not isinstance(partial, (int, float)):
                self._disable_missing_signal(entry)
                return None
            candidates.append(_Candidate(entry, float(partial), alive=True))
        return candidates

    def _disable_missing_signal(self, entry: LaunchedTrial) -> None:
        """A prunable test whose scene reports no partial_score: never guess."""
        self.disabled = True
        logger.warning(
            f"[prune] Gen {self.gen_index:04d}: trial {entry.trial_index:02d} reached a "
            f"rung step without a partial_score — the scene does not honor "
            f"TestSpec.prunable; pruning disabled for this generation."
        )

    # -- rung firing -----------------------------------------------------------

    def _fire(self, step: int, keep_fraction: float, candidates: list[_Candidate]) -> None:
        self._report_partials(step, candidates)
        keep = max(1, math.ceil(len(candidates) * keep_fraction))
        ranked = sorted(candidates, key=lambda c: c.value)
        doomed = [c for c in ranked[: len(ranked) - keep] if c.alive]
        if not doomed:
            return
        cut = ranked[len(ranked) - keep].value
        for c in doomed:
            reason = (
                f"rung {step}: partial {c.value:.2f} below cut {cut:.2f} "
                f"(kept {keep}/{len(candidates)})"
            )
            self._apply(c.entry, step, reason)

    def _report_partials(self, step: int, candidates: list[_Candidate]) -> None:
        """Bridge partial scores to Optuna BEFORE any kill, so a pruned trial's
        last intermediate value is its partial at this rung."""
        for c in candidates:
            if not c.alive:
                continue
            # A closed/duplicate-step report must never break the generation.
            with contextlib.suppress(Exception):
                c.entry.trial.report(c.value, step)

    def _apply(self, entry: LaunchedTrial, step: int, reason: str) -> None:
        if self.cfg.project.prune_mode == "kill":
            prune_trial(
                self.cfg, self.gen_index, entry.trial_index,
                entry.trial_state_path, entry.runs, reason,
            )
            return
        update_trial_run(
            entry.trial_state_path, self.run_slot,
            {"shadow_prune_step": step, "shadow_prune_note": reason},
        )
        logger.info(
            f"[prune-shadow] Gen {self.gen_index:04d} Trial {entry.trial_index:02d}: "
            f"would kill — {reason}"
        )


def build_pruner(
    cfg: RunConfig, gen_index: int, entries: list[LaunchedTrial]
) -> GenerationPruner | None:
    """A pruner for this generation, or None when pruning does not apply.

    Project-level shape (single-objective, one prunable test with rungs) is
    enforced by ``SofaOptProject._validate_pruning``; this only re-checks the
    runtime selection, since the dashboard may deselect tests per run.
    """
    project = cfg.project
    if project.prune_mode == "off" or project.multi_objective:
        return None
    selected = list(cfg.selected_tests)
    if len(selected) != 1 or not selected[0].prunable or not selected[0].prune_rungs:
        return None
    run_slot = next(
        i + 1 for i, (name, _, _) in enumerate(cfg.run_plan) if name == selected[0].name
    )
    return GenerationPruner(
        cfg=cfg,
        gen_index=gen_index,
        entries=entries,
        run_slot=run_slot,
        rungs=selected[0].prune_rungs,
    )
