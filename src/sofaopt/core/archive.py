"""Archiving of optimization runs and cross-run comparison data.

An archive is a *moved* ``runtime/`` directory: ``work_dir/archives/<slug>/``
holds the run's ``trials/``, ``study.db``, ``progress.json``, ``summary.mp4``
exactly as they were, plus an ``archive.json`` manifest (project snapshot via
``project_to_jsonable``, sampler settings, best score/params, notes). Moving —
not copying — keeps archiving instant and disk-neutral, and doubles as the
workspace reset: the orchestrator auto-archives any existing run before
starting a fresh one, so starting a new run can never destroy a previous one.

Comparison data derives ONLY from each archive's recorded scores
(:mod:`sofaopt.core.results`) — never recomputed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from sofaopt.core.results import load_trial_records, rank_completed
from sofaopt.project import SofaOptProject, project_to_jsonable

logger = logging.getLogger(__name__)

MANIFEST_NAME = "archive.json"


@dataclass(frozen=True)
class ArchiveInfo:
    """One archived run, as described by its manifest."""

    path: Path
    name: str
    created_at: float
    notes: str
    sampler: str
    multi_objective: bool
    test_names: list[str]
    n_trials: int
    n_gens: int
    best_score: float | None
    best_params: dict
    project_snapshot: dict
    summary_stats: dict

    @property
    def trials_dir(self) -> Path:
        return self.path / "trials"


def archives_dir(project: SofaOptProject) -> Path:
    # Deliberately OUTSIDE runtime/ — runtime is what gets moved/reset.
    return project.work_dir / "archives"


def _slugify(name: str) -> str:
    """Filesystem-safe archive slug (a name must never escape archives/, §9)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return slug[:80]


def runtime_has_run_data(project: SofaOptProject) -> bool:
    """True when runtime/ holds anything worth archiving."""
    return project.db_path.exists() or any(project.trials_dir.glob("gen_*"))


def archive_run(
    project: SofaOptProject, name: str = "", notes: str = ""
) -> Path:
    """Move the current ``runtime/`` into a named archive and write its manifest.

    Returns the archive directory. Raises ``FileNotFoundError`` when there is
    no run data to archive.
    """
    if not runtime_has_run_data(project):
        raise FileNotFoundError(f"No run data to archive in {project.runtime_dir}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    slug = f"{stamp}_{_slugify(name)}" if _slugify(name) else stamp
    dest = archives_dir(project) / slug
    if dest.exists():  # same-second collision: disambiguate, never overwrite
        dest = archives_dir(project) / f"{slug}_{int(time.time() * 1000) % 1000:03d}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Summarize BEFORE the move so a failed summary can't lose data mid-way.
    manifest = _build_manifest(project, name=name or slug, notes=notes)

    _move_with_retry(project.runtime_dir, dest)
    (dest / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(f"[archive] Run archived -> {dest}")
    return dest


def _move_with_retry(src: Path, dest: Path, attempts: int = 5, delay_s: float = 0.4) -> None:
    """``shutil.move`` with a short retry loop for a transient Windows lock.

    Archiving from the dashboard moves ``runtime/`` while the same process may
    still be releasing a study.db handle (an in-flight interaction analysis);
    the analysis path disposes its engine eagerly (see
    ``analysis._dispose_study_storage``), but a brief retry covers the race
    and any OS-level lag in freeing the handle, turning a hard WinError 32 into
    a short wait. Re-raises the last error if the lock never clears."""
    for i in range(attempts):
        try:
            shutil.move(str(src), str(dest))
            return
        except (PermissionError, OSError) as exc:
            if i == attempts - 1:
                raise
            logger.info(f"[archive] move blocked (attempt {i + 1}/{attempts}): {exc}; retrying")
            time.sleep(delay_s)


def _build_manifest(project: SofaOptProject, *, name: str, notes: str) -> dict:
    records = load_trial_records(project.trials_dir)
    ranked = rank_completed(records)
    best = ranked[0] if ranked else None
    return {
        "name": name,
        "created_at": time.time(),
        "notes": notes,
        "sampler": project.sampler,
        "multi_objective": project.multi_objective,
        "test_names": [t.name for t in project.tests],
        "n_trials": len(records),
        "n_gens": max((r["gen_index"] for r in records), default=0),
        "best_score": best["final_score"] if best else None,
        "best_params": (best or {}).get("params", {}),
        "project_snapshot": project_to_jsonable(project),
        # Compact convergence/diversity summary, computed once here so the
        # Archives comparison can rank runs on more than best score without
        # re-walking every archive's full trial tree on each view.
        "summary_stats": run_summary_stats(project.trials_dir, records=records),
    }


def load_archive_info(path: Path) -> ArchiveInfo:
    """Read one archive's manifest (tolerates a missing/old manifest)."""
    path = Path(path)
    try:
        data = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return ArchiveInfo(
        path=path,
        name=str(data.get("name", path.name)),
        created_at=float(data.get("created_at", 0.0)),
        notes=str(data.get("notes", "")),
        sampler=str(data.get("sampler", "?")),
        multi_objective=bool(data.get("multi_objective", False)),
        test_names=list(data.get("test_names", [])),
        n_trials=int(data.get("n_trials", 0)),
        n_gens=int(data.get("n_gens", 0)),
        best_score=data.get("best_score"),
        best_params=dict(data.get("best_params", {})),
        project_snapshot=dict(data.get("project_snapshot", {})),
        summary_stats=dict(data.get("summary_stats", {})),
    )


def list_archives(project: SofaOptProject) -> list[ArchiveInfo]:
    """All archives for a project, newest first."""
    root = archives_dir(project)
    if not root.is_dir():
        return []
    infos = [load_archive_info(d) for d in root.iterdir() if d.is_dir()]
    return sorted(infos, key=lambda a: (a.created_at, a.path.name), reverse=True)


def _resolve_archive(project: SofaOptProject, archive: str | Path) -> Path:
    """Resolve an archive by dir name, refusing anything outside archives/ (§9)."""
    root = archives_dir(project).resolve()
    path = (root / Path(archive).name).resolve()
    if path.parent != root or not path.is_dir():
        raise FileNotFoundError(f"No archive named {Path(archive).name!r} in {root}")
    return path


def delete_archive(project: SofaOptProject, archive: str | Path) -> None:
    """Permanently delete one archive."""
    path = _resolve_archive(project, archive)
    shutil.rmtree(path)
    logger.info(f"[archive] Deleted {path.name}")


def restore_archive(project: SofaOptProject, archive: str | Path) -> Path:
    """Move an archive back to ``runtime/`` (it becomes the live run again).

    Any current run data is auto-archived first, so a restore can never
    destroy anything. The restored run can then be resumed (its ``study.db``
    is intact) and re-archived later. Returns the runtime dir.
    """
    path = _resolve_archive(project, archive)
    if runtime_has_run_data(project):
        archive_run(project, name="auto_before_restore")
    elif project.runtime_dir.exists():
        shutil.rmtree(project.runtime_dir)  # empty scaffold from a fresh start

    (path / MANIFEST_NAME).unlink(missing_ok=True)  # live runs carry no manifest
    shutil.move(str(path), str(project.runtime_dir))
    logger.info(f"[archive] Restored {path.name} -> {project.runtime_dir}")
    return project.runtime_dir


# ---------------------------------------------------------------------------
# Run summary stats (generic — recorded results only, no domain knowledge)
# ---------------------------------------------------------------------------

def _pop_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def _numeric_param_names(completed: list[dict]) -> list[str]:
    """Numeric params that actually VARIED across the run — the searched
    dimensions. Constants (frozen params, fixed knobs) are excluded so they
    never clutter a per-param view or skew a diversity aggregate."""
    seen: dict[str, set] = {}
    for r in completed:
        for key, val in (r.get("params") or {}).items():
            if isinstance(val, (int, float)):
                seen.setdefault(key, set()).add(val)
    return [k for k, vals in seen.items() if len(vals) > 1]


def _param_values(records: list[dict], name: str) -> list[float]:
    return [r["params"][name] for r in records if isinstance(r.get("params", {}).get(name), (int, float))]


def _diversity_collapse_pct(completed: list[dict], k_gens: int = 5) -> float | None:
    """How much the searched population narrowed, in [0, 100] (0 = no
    narrowing, 100 = fully converged). Mean over each numeric param that
    actually VARIED early of ``1 - final_std/initial_std`` (first k gens vs
    last k gens). Bounded and aggregate, so a single near-frozen param can't
    dominate and truly-constant params (never varied) drop out. Generic —
    reads only ``params``. ``None`` when there are too few generations."""
    by_gen: dict[int, list[dict]] = {}
    for r in completed:
        by_gen.setdefault(r.get("gen_index", 0), []).append(r)
    gens = sorted(by_gen)
    if len(gens) < 4:
        return None
    half = min(k_gens, len(gens) // 2)
    first = [r for g in gens[:half] for r in by_gen[g]]
    last = [r for g in gens[-half:] for r in by_gen[g]]
    collapses: list[float] = []
    for name in _numeric_param_names(completed):
        s0 = _pop_std(_param_values(first, name))
        if s0 <= 1e-9:  # never varied early -> not a searched dimension here
            continue
        s1 = _pop_std(_param_values(last, name))
        collapses.append(max(0.0, 1.0 - s1 / s0))
    if not collapses:
        return None
    return round(100.0 * sum(collapses) / len(collapses), 1)


def _final_param_stats(completed: list[dict]) -> dict:
    """Per searched (varying) numeric param: the FINAL generation's population
    mean and std — where each run's search actually settled per parameter, not
    just its single best point. Powers the same-case in-depth comparison."""
    by_gen: dict[int, list[dict]] = {}
    for r in completed:
        by_gen.setdefault(r.get("gen_index", 0), []).append(r)
    if not by_gen:
        return {}
    finals = by_gen[max(by_gen)]
    out: dict[str, dict] = {}
    for name in _numeric_param_names(completed):
        vals = _param_values(finals, name)
        if vals:
            out[name] = {"mean": round(sum(vals) / len(vals), 3), "std": round(_pop_std(vals), 3)}
    return out


def _trials_to_fraction(completed: list[dict], best: float, frac: float) -> int | None:
    """First evaluation (1-based chron) reaching ``frac`` of the eventual best
    — a convergence-speed proxy. ``None`` when best is non-positive (the
    fraction is meaningless) or never reached."""
    if best <= 0:
        return None
    target = best * frac
    for r in completed:  # chronological (load_trial_records order preserved)
        if r.get("final_score", float("-inf")) >= target:
            return r.get("chron", 0) + 1
    return None


# Bump when run_summary_stats's schema changes so stored manifests recompute
# instead of showing stale/absent fields (see _ensure_summary_stats).
_STATS_VERSION = 2


def run_summary_stats(trials_dir: Path, *, records: list[dict] | None = None) -> dict:
    """Compact, generic convergence/diversity summary for one run.

    Everything from recorded results only (``final_score``, ``params``,
    ``gen_index``, ``failed``) plus restart-event count — no domain concepts,
    so it works for any project. Stored in the archive manifest at archive
    time and shown side-by-side in the Archives comparison. Pass ``records``
    to reuse an already-loaded list (the archive-time caller has one).
    """
    from sofaopt.core.restart_events import load_restart_events

    if records is None:
        records = load_trial_records(trials_dir)
    completed = [
        r for r in records
        if not r.get("failed") and isinstance(r.get("final_score"), (int, float))
    ]
    n_failed = sum(1 for r in records if r.get("failed"))
    if not completed:
        return {"version": _STATS_VERSION, "n_completed": 0, "n_failed": n_failed}

    scores = [r["final_score"] for r in completed]
    best = max(scores)
    final_gen = max(r.get("gen_index", 0) for r in completed)
    final_scores = [r["final_score"] for r in completed if r.get("gen_index") == final_gen]
    return {
        "version": _STATS_VERSION,
        "n_completed": len(completed),
        "n_failed": n_failed,
        "mean_score": round(sum(scores) / len(scores), 3),
        "final_mean": round(sum(final_scores) / len(final_scores), 3),
        "final_std": round(_pop_std(final_scores), 3),
        "trials_to_90pct_best": _trials_to_fraction(completed, best, 0.9),
        "diversity_collapse_pct": _diversity_collapse_pct(completed),
        "final_param_stats": _final_param_stats(completed),
        "n_restarts": len(load_restart_events(trials_dir)),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def best_so_far_curve(trials_dir: Path) -> tuple[list[int], list[float]]:
    """(trial index, running best recorded score) for one run's trials dir.

    Failed trials advance the x axis but never the best line, so curves from
    runs with different failure rates stay comparable per evaluation spent.
    """
    xs: list[int] = []
    ys: list[float] = []
    best: float | None = None
    for r in load_trial_records(trials_dir):
        score = r.get("final_score")
        if (
            not r.get("failed")
            and isinstance(score, (int, float))
            and (best is None or score > best)
        ):
            best = float(score)
        if best is not None:
            xs.append(r["chron"] + 1)
            ys.append(best)
    return xs, ys


def case_signature(snapshot: dict, test_names: list[str]) -> tuple:
    """A run's 'problem identity': its searched parameter space + test set.

    Two runs share a case when this matches — only then is a per-parameter
    comparison meaningful (same knobs, same objective). Frozen params
    (``low == high``) are excluded so freezing an unused knob doesn't split
    otherwise-identical cases.
    """
    params = tuple(sorted(
        (p.get("name"), p.get("type"), p.get("low"), p.get("high"))
        for p in snapshot.get("params", [])
        if p.get("low") != p.get("high")
    ))
    return (params, tuple(sorted(test_names)))


def _ensure_summary_stats(path: Path, info: ArchiveInfo) -> dict:
    """Return an archive's stored summary stats, computing + persisting them
    for archives that predate the feature or carry an older schema version
    (write-through backfill: the first comparison pays the walk, later ones
    are instant).
    """
    if info.summary_stats.get("version") == _STATS_VERSION:
        return info.summary_stats
    stats = run_summary_stats(info.trials_dir)
    with contextlib.suppress(Exception):  # read-path best-effort: never fail a comparison
        manifest_path = path / MANIFEST_NAME
        data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        data["summary_stats"] = stats
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return stats


def comparison_data(
    project: SofaOptProject,
    archives: list[str | Path],
    *,
    include_current: bool = False,
) -> list[dict]:
    """Comparison series for N archives (and optionally the live run).

    One entry per run: ``{label, curve: (xs, ys), info: ArchiveInfo | None,
    best_score, best_params, n_trials, sampler, notes, summary_stats}`` —
    everything read from recorded results, nothing recomputed.
    """
    from sofaopt.core.restart_events import load_restart_events

    entries: list[dict] = []
    for archive in archives:
        path = _resolve_archive(project, archive)
        info = load_archive_info(path)
        entries.append(
            {
                "label": info.name,
                "curve": best_so_far_curve(info.trials_dir),
                "best_score": info.best_score,
                "best_params": info.best_params,
                "n_trials": info.n_trials,
                "sampler": info.sampler,
                "notes": info.notes,
                "info": info,
                "restarts": load_restart_events(info.trials_dir),
                "summary_stats": _ensure_summary_stats(path, info),
                "case_signature": case_signature(info.project_snapshot, info.test_names),
            }
        )
    if include_current and runtime_has_run_data(project):
        records = load_trial_records(project.trials_dir)
        ranked = rank_completed(records)
        best = ranked[0] if ranked else None
        entries.append(
            {
                "label": "current run",
                "curve": best_so_far_curve(project.trials_dir),
                "best_score": best["final_score"] if best else None,
                "best_params": (best or {}).get("params", {}),
                "n_trials": len(records),
                "sampler": project.sampler,
                "notes": "",
                "info": None,
                "restarts": load_restart_events(project.trials_dir),
                "summary_stats": run_summary_stats(project.trials_dir, records=records),
                "case_signature": case_signature(
                    project_to_jsonable(project), [t.name for t in project.tests]
                ),
            }
        )
    return entries
