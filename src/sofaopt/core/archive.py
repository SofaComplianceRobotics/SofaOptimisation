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

    shutil.move(str(project.runtime_dir), str(dest))
    (dest / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(f"[archive] Run archived -> {dest}")
    return dest


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


def comparison_data(
    project: SofaOptProject,
    archives: list[str | Path],
    *,
    include_current: bool = False,
) -> list[dict]:
    """Comparison series for N archives (and optionally the live run).

    One entry per run: ``{label, curve: (xs, ys), info: ArchiveInfo | None,
    best_score, best_params, n_trials, sampler, notes}`` — everything read
    from recorded results, nothing recomputed.
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
            }
        )
    return entries
