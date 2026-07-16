"""Anytime-score trace analysis for multi-fidelity step pruning.

Reads the anytime-score traces (``score_trace_run*.json``, written by a project's
scene when ``OPT_SCORE_TRACE`` is set — see ``docs/design/multi-fidelity.md``) out
of a trials tree and answers the two questions that decide whether pruning is
worth turning on:

1. **Anytime rank validity** — at simulation step k, how well do the partial
   scores rank a generation's candidates against their final scores (Spearman
   rho, per test)?
2. **Simulated pruning** — had the bottom of each generation been killed at rung
   step k (always keeping ``keep`` trials — CMA-ES's mu), how many simulation
   steps would have been saved, and how often would a top-``keep``-by-final
   trial (or the generation winner, or the eventual study best) have been
   killed by mistake?

Promoted out of ``examples/prefix_pruning_study/replay.py`` so both the CLI tool
there and the dashboard's Optimization Health panel share one implementation —
project-agnostic, reads only ``score_trace_run*.json`` files already on disk.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunTrace:
    """One run's anytime-score trace plus its final outcome."""

    gen: int
    trial: int
    test_name: str
    points: tuple  # ((step, anytime score), ...) at the scene's trace cadence
    final_score: float
    end_step: int


def load_traces(trials_dir: Path) -> list[RunTrace]:
    """All score traces under ``trials_dir`` (gen_*/trial_*/score_trace_run*.json)."""
    traces = []
    for path in sorted(Path(trials_dir).glob("gen_*/trial_*/score_trace_run*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        traces.append(
            RunTrace(
                gen=int(path.parent.parent.name.split("_")[1]),
                trial=int(path.parent.name.split("_")[1]),
                test_name=str(data["test_name"]),
                points=tuple((int(s), float(v)) for s, v in data["points"]),
                final_score=float(data["final_score"]),
                end_step=int(data["end_step"]),
            )
        )
    return traces


def score_at(rt: RunTrace, step: int) -> float | None:
    """The run's anytime score at ``step``: final once ended, else the last
    trace point at or before ``step`` (None while there is no signal yet)."""
    if rt.end_step <= step:
        return rt.final_score
    last = None
    for s, v in rt.points:
        if s > step:
            break
        last = v
    return last


def _runs_by_gen_trial(traces: list[RunTrace]) -> dict:
    """{gen: {trial: [RunTrace, ...]}} view of the flat trace list."""
    gens: dict[int, dict[int, list[RunTrace]]] = {}
    for rt in traces:
        gens.setdefault(rt.gen, {}).setdefault(rt.trial, []).append(rt)
    return gens


def trial_value_at(runs: list[RunTrace], step: int) -> float | None:
    """Unweighted mean of the trial's per-run anytime scores (None = no signal)."""
    values = [score_at(rt, step) for rt in runs]
    if any(v is None for v in values):
        return None
    return statistics.mean(values)


def trial_final(runs: list[RunTrace]) -> float:
    return statistics.mean(rt.final_score for rt in runs)


# -- rank statistics ----------------------------------------------------------

def _ranks(values: list[float]) -> list[float]:
    """Average ranks (ties shared), 1-based."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation; None when undefined (n < 3 or zero variance)."""
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    mean_a, mean_b = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def _gen_test_pairs(gen_runs: dict, step: int) -> dict[str, list[tuple[float, float]]]:
    """(partial@step, final) pairs of one generation's runs, grouped per test."""
    per_test: dict[str, list[tuple[float, float]]] = {}
    for runs in gen_runs.values():
        for rt in runs:
            v = score_at(rt, step)
            if v is not None:
                per_test.setdefault(rt.test_name, []).append((v, rt.final_score))
    return per_test


def _accumulate_gen_rhos(by_test: dict[str, list[float]], gen_runs: dict, step: int) -> None:
    for test, pairs in _gen_test_pairs(gen_runs, step).items():
        rho = spearman([p for p, _ in pairs], [f for _, f in pairs])
        if rho is not None:
            by_test.setdefault(test, []).append(rho)


def rank_validity(traces: list[RunTrace], step: int) -> dict[str, float]:
    """Mean per-generation Spearman(partial@step, final) for each test."""
    by_test: dict[str, list[float]] = {}
    for gen_runs in _runs_by_gen_trial(traces).values():
        _accumulate_gen_rhos(by_test, gen_runs, step)
    return {test: statistics.mean(rhos) for test, rhos in by_test.items()}


# -- pruning simulation --------------------------------------------------------

@dataclass
class ScheduleOutcome:
    """Aggregate result of replaying one rung schedule over the campaign."""

    saved_steps: int = 0
    total_steps: int = 0
    killed: int = 0
    candidates: int = 0
    regret_top_keep: int = 0   # killed trials that finish in their gen's top-keep
    winner_kills: int = 0      # killed trials that were their generation's best
    study_best_kills: int = 0  # killed trials that were the campaign's best

    @property
    def savings_pct(self) -> float:
        return 100.0 * self.saved_steps / self.total_steps if self.total_steps else 0.0


def _kills_for_rung(alive: dict, step: int, keep: int) -> list | None:
    """Trials to kill at one rung: the bottom of the anytime ranking, minus
    already-finished trials (killing them saves nothing). None = no signal yet."""
    values = {t: trial_value_at(runs, step) for t, runs in alive.items()}
    if any(v is None for v in values.values()):
        return None
    order = sorted(alive, key=lambda t: values[t])
    doomed = order[: max(0, len(alive) - keep)]
    return [t for t in doomed if any(rt.end_step > step for rt in alive[t])]


def _apply_kill(out: ScheduleOutcome, gen_runs: dict, trial: int, step: int,
                finals: dict, keep: int, study_best: tuple) -> None:
    out.killed += 1
    out.saved_steps += sum(max(0, rt.end_step - step) for rt in gen_runs[trial])
    top = sorted(finals, key=finals.get, reverse=True)[:keep]
    if trial in top:
        out.regret_top_keep += 1
    if trial == top[0]:
        out.winner_kills += 1
    if study_best == (gen_runs[trial][0].gen, trial):
        out.study_best_kills += 1


def _replay_generation(out: ScheduleOutcome, gen_runs: dict,
                       schedule: list[tuple[int, int]], study_best: tuple) -> None:
    finals = {t: trial_final(runs) for t, runs in gen_runs.items()}
    out.candidates += len(gen_runs)
    alive = dict(gen_runs)
    for step, keep in schedule:
        kills = _kills_for_rung(alive, step, keep)
        if kills is None:
            continue  # rung precedes the first trace point — skip it
        for t in kills:
            _apply_kill(out, gen_runs, t, step, finals, keep, study_best)
            del alive[t]


def simulate_schedule(
    traces: list[RunTrace], schedule: list[tuple[int, int]]
) -> ScheduleOutcome:
    """Replay ``[(rung step, keep), ...]`` (ascending steps) over every generation."""
    gens = _runs_by_gen_trial(traces)
    study_best = max(
        ((g, t) for g, trials in gens.items() for t in trials),
        key=lambda gt: trial_final(gens[gt[0]][gt[1]]),
    )
    out = ScheduleOutcome(total_steps=sum(rt.end_step for rt in traces))
    for gen_runs in gens.values():
        _replay_generation(out, gen_runs, schedule, study_best)
    return out


# -- report --------------------------------------------------------------------

def _step_grid(traces: list[RunTrace], points: int = 24) -> list[int]:
    """Candidate rung steps: trace-cadence-aligned, up to the p90 end step."""
    ends = sorted(rt.end_step for rt in traces)
    p90 = ends[int(0.9 * (len(ends) - 1))]
    first = min(s for rt in traces for s, _ in rt.points[:1]) if traces else 5
    stride = max(5, (p90 - first) // points)
    return list(range(first, p90 + 1, stride))


def analyze(trials_dir: Path, keep_fraction: float = 0.5) -> dict | None:
    """Sweep single rungs over the step grid; report validity/savings/regret.
    Returns ``None`` (rather than raising) when no traces exist yet — callers
    like the dashboard need to degrade gracefully, not crash on a live study
    that hasn't opted into ``OPT_SCORE_TRACE``."""
    traces = load_traces(trials_dir)
    if not traces:
        return None
    n_trials = len({(rt.gen, rt.trial) for rt in traces})
    n_gens = len({rt.gen for rt in traces})
    lam = max(len(v) for v in _runs_by_gen_trial(traces).values())
    keep = max(1, math.ceil(lam * keep_fraction))
    rows = []
    for step in _step_grid(traces):
        outcome = simulate_schedule(traces, [(step, keep)])
        rows.append({
            "step": step,
            "rank_validity": {t: round(r, 3) for t, r in rank_validity(traces, step).items()},
            "savings_pct": round(outcome.savings_pct, 1),
            "killed": outcome.killed,
            "candidates": outcome.candidates,
            "regret_top_keep": outcome.regret_top_keep,
            "winner_kills": outcome.winner_kills,
            "study_best_kills": outcome.study_best_kills,
        })
    return {
        "trials_dir": str(trials_dir),
        "n_gens": n_gens,
        "n_trials": n_trials,
        "n_runs": len(traces),
        "lambda": lam,
        "keep": keep,
        "end_step_median": int(statistics.median(rt.end_step for rt in traces)),
        "rows": rows,
        "best_safe": _best_safe(rows),
    }


def _best_safe(rows: list[dict]) -> dict | None:
    """Highest-savings single rung that never killed a generation winner."""
    safe = [r for r in rows if r["winner_kills"] == 0 and r["study_best_kills"] == 0]
    return max(safe, key=lambda r: r["savings_pct"]) if safe else None


def format_report(report: dict) -> str:
    lines = [
        f"traces: {report['n_runs']} runs / {report['n_trials']} trials / "
        f"{report['n_gens']} gens (lambda={report['lambda']}, keep=mu={report['keep']}, "
        f"median end step {report['end_step_median']})",
        f"{'step':>6} {'savings%':>9} {'killed':>7} {'regret':>7} "
        f"{'winner-kills':>13} {'best-kills':>11}  rank-validity (Spearman)",
    ]
    for r in report["rows"]:
        rho = " ".join(f"{t}={v:+.2f}" for t, v in sorted(r["rank_validity"].items()))
        lines.append(
            f"{r['step']:>6} {r['savings_pct']:>9.1f} "
            f"{r['killed']:>7} {r['regret_top_keep']:>7} "
            f"{r['winner_kills']:>13} {r['study_best_kills']:>11}  {rho}"
        )
    best = report["best_safe"]
    lines.append(
        "best safe single rung: "
        + (f"step {best['step']} -> {best['savings_pct']}% saved, "
           f"{best['regret_top_keep']} top-mu regrets" if best else "none found")
    )
    return "\n".join(lines)
