"""Optimization Health panel — pruning readiness + general run-health summary.

Reflects the study's on-disk state on a short TTL (see ``_PRUNING_CACHE_TTL_S``)
rather than every call -- so if the objective changes later (more targets, more
tests), the recommendation still adapts within moments, not a stale-forever
verdict, but pruning_trace.analyze() (a full glob + read + parse of every
score_trace_run*.json under trials_dir) doesn't re-run on every fast poll
tick once a study has hundreds/thousands of trials.
"""

from __future__ import annotations

import time

from dash import html

from sofaopt import pruning_trace
from sofaopt.core.restart_events import load_restart_events
from sofaopt.dashboard import context as _ctx

_PRUNING_CACHE_TTL_S = 20.0
_pruning_cache: dict = {"trials_dir": None, "report": None, "last_load": 0.0}


def _cached_pruning_report(trials_dir):
    now = time.time()
    fresh = (now - _pruning_cache["last_load"]) < _PRUNING_CACHE_TTL_S
    if _pruning_cache["trials_dir"] == trials_dir and fresh:
        return _pruning_cache["report"]
    report = pruning_trace.analyze(trials_dir)
    _pruning_cache.update(trials_dir=trials_dir, report=report, last_load=now)
    return report


# Recommendation thresholds. Calibrated against two real campaigns, not a
# formula: sofaopt's own reference platforms (trunk/liver, docs/design/
# multi-fidelity.md §7) land 28-40% savings at a low regret rate -- a clear
# promote call -- while a real negative case (a foamboathex campaign) landed
# 4.1% savings at ~60% regret -- clearly not ready. These bars sit well
# inside that gap; the printed numbers always override the label for a
# borderline case.
_MIN_SAVINGS_PCT = 10.0
_MAX_REGRET_RATE = 0.25


def _stat(label: str, value: str, css: str = "text-info") -> html.Div:
    return html.Div(
        [html.H6(label, className="text-muted mb-1"), html.H5(value, className=css)],
        className="col-6 col-md-3",
    )


def _pruning_recommendation(report: dict, prune_mode: str) -> tuple[str, str, str]:
    """(label, detail, css class) for the best-safe rung in ``report``."""
    best = report["best_safe"]
    if best is None:
        return ("NOT READY",
                "No rung avoids killing a generation winner anywhere in the step "
                "sweep — keep prune_mode='off'.", "text-danger")
    regret_rate = best["regret_top_keep"] / best["killed"] if best["killed"] else 0.0
    detail = (
        f"Best safe rung: step {best['step']}, {best['savings_pct']:.1f}% steps saved, "
        f"{best['regret_top_keep']}/{best['killed']} killed trials would have finished "
        f"top-mu ({regret_rate:.0%} regret rate)."
    )
    if best["savings_pct"] < _MIN_SAVINGS_PCT or regret_rate > _MAX_REGRET_RATE:
        return ("MARGINAL", detail + f" Below the readiness bar (>={_MIN_SAVINGS_PCT:.0f}% "
                f"savings, <={_MAX_REGRET_RATE:.0%} regret) — gather more trials or stay "
                "in shadow.", "text-warning")
    if prune_mode == "off":
        return ("READY FOR SHADOW", detail + " Consider prune_mode='shadow' with "
                f"prune_rungs=[({best['step']}, keep_fraction)].", "text-success")
    if prune_mode == "shadow":
        return ("READY TO PROMOTE", detail + " Shadow mode looks safe and worthwhile — "
                "consider prune_mode='kill'.", "text-success")
    return ("HEALTHY", detail + " prune_mode='kill' is active and the numbers still look "
            "safe.", "text-success")


def _build_pruning_health(project) -> html.Div:
    report = _cached_pruning_report(_ctx.trials_dir())
    if report is None:
        body = html.Div(
            "No score_trace_run*.json found for this study — set OPT_SCORE_TRACE=1 "
            "on the prunable test's scene to get a pruning readiness read here.",
            className="text-muted small",
        )
    else:
        label, detail, css = _pruning_recommendation(report, project.prune_mode)
        body = html.Div([
            html.Div([
                _stat("Recommendation", label, css=css),
                _stat("Current mode", project.prune_mode),
                _stat("Traced runs", str(report["n_runs"])),
                _stat("Generations", str(report["n_gens"])),
            ], className="row g-3 mb-2"),
            html.Div(detail, className="small"),
        ])
    return html.Div(
        [html.Div("Pruning readiness", className="text-muted small mb-2 fw-semibold"), body],
        className="p-3 bg-light rounded mb-3",
    )


def _best_per_generation(records: list[dict]) -> dict[str, float]:
    by_gen: dict[str, float] = {}
    for r in records:
        gen, score = r.get("gen_name", ""), r.get("final_score")
        if r.get("failed") or not gen or score is None:
            continue
        by_gen[gen] = max(by_gen.get(gen, float("-inf")), score)
    return by_gen

def _score_trend(records: list[dict]) -> tuple[str, str]:
    """(label, detail) — best-score trend over the last few generations."""
    by_gen = _best_per_generation(records)
    gens = sorted(by_gen)
    if len(gens) < 4:
        return "—", "Not enough completed generations yet."
    bests = [by_gen[g] for g in gens]
    recent_best, prior_best = max(bests[-3:]), max(bests[:-3][-3:] or bests[:1])
    if recent_best > prior_best * 1.01:
        return ("IMPROVING", f"Best score rose {prior_best:.1f} → {recent_best:.1f} "
                "over the last 3 generations.")
    if recent_best < prior_best * 0.99:
        return ("REGRESSED", f"Best score DROPPED {prior_best:.1f} → {recent_best:.1f} "
                "— unexpected for an elitist search, worth investigating.")
    return "PLATEAUED", f"Best score flat at ~{recent_best:.1f} over the last 3 generations."


_TREND_CSS = {"IMPROVING": "text-success", "REGRESSED": "text-danger",
              "PLATEAUED": "text-warning"}


def _build_general_health(records: list[dict], progress: dict | None) -> html.Div:
    label, detail = _score_trend(records)
    rs = (progress or {}).get("restart") or {}
    events = load_restart_events(_ctx.trials_dir())
    cells = [_stat("Score trend", label, css=_TREND_CSS.get(label, "text-muted"))]
    if rs.get("stall_limit") and rs.get("convergence_trigger"):
        # Informational only in this mode -- see panels._build_restart_status.
        cells.append(_stat("Since improvement (info only)",
                           f"{rs.get('stall_count', 0)} gens"))
    elif rs.get("stall_limit"):
        cells.append(_stat("Since improvement",
                           f"{rs.get('stall_count', 0)} / {rs['stall_limit']} gens"))
    if events:
        cells.append(_stat("Restarts fired", str(len(events))))
    return html.Div(
        [
            html.Div("General health", className="text-muted small mb-2 fw-semibold"),
            html.Div(cells, className="row g-3 mb-2"),
            html.Div(detail, className="small"),
        ],
        className="p-3 bg-light rounded mb-3",
    )


def build_health_panel(records: list[dict], project, progress: dict | None) -> html.Div:
    """Optimization Health: pruning readiness + general run-health summary."""
    return html.Div([
        _build_general_health(records, progress),
        _build_pruning_health(project),
    ])
