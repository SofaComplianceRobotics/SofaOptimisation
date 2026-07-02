"""Per-trial parameter sampling and the optional preparation step.

This is the generalized replacement for the old shape-specific geometry
pipeline. The flow per trial is:

  1. :func:`params_from_trial` samples values from the project's specs (and
     applies the optional ``constrain_params`` hook).
  2. :func:`prepare_trial` writes ``params.json`` into the trial dir and, if the
     project supplies a ``prepare_trial`` hook, runs it to build any per-trial
     asset (e.g. a mesh) and collect extra scene env / cleanup / preview.

A project that only tunes scene quantities (stiffness, mass, gains, ...) needs
no hook at all — the scene just reads ``params.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sofaopt.project import SofaOptProject, TrialPrep


def _round_float(value: float) -> float:
    return round(float(value), 3)


def params_from_trial(trial, project: SofaOptProject) -> dict[str, Any]:
    """Sample a full parameter dict from an Optuna trial using the project specs.

    Frozen params (float/int with ``low == high``) use their default; the rest
    are suggested. The optional ``constrain_params`` hook may then adjust the
    *used* values without affecting what the optimizer recorded.
    """
    result: dict[str, Any] = {}
    for spec in project.params:
        name = spec.name
        if spec.is_frozen:
            result[name] = spec.default
        elif spec.type == "float":
            if project.float_step is not None:
                value = trial.suggest_float(
                    name, spec.low, spec.high, step=project.float_step
                )
            else:
                value = trial.suggest_float(name, spec.low, spec.high)
            result[name] = _round_float(value)
        elif spec.type == "int":
            result[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
        elif spec.type == "bool":
            result[name] = trial.suggest_categorical(name, [False, True])
        else:
            result[name] = spec.default

    if project.constrain_params is not None:
        result = dict(project.constrain_params(result))
    return result


def prepare_trial(
    project: SofaOptProject, params: dict[str, Any], trial_dir: Path
) -> TrialPrep:
    """Write params.json and run the project's prepare hook (if any).

    The hook runs under ``project.prepare_timeout`` (a hung user hook must not
    wedge the whole generation). Returns a :class:`TrialPrep`. Raises whatever
    the hook raises, or :class:`TimeoutError` — the caller treats either as a
    hard failure for the trial.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )

    if project.prepare_trial is None:
        return TrialPrep()

    prep = _run_hook_with_timeout(project, params, trial_dir)
    if prep is None:  # tolerate hooks that only set env and return nothing
        return TrialPrep()
    # Normalize env values to strings.
    prep.env = {k: str(v) for k, v in prep.env.items()}
    return prep


def _run_hook_with_timeout(
    project: SofaOptProject, params: dict[str, Any], trial_dir: Path
) -> TrialPrep | None:
    """Run the prepare hook, enforcing ``project.prepare_timeout`` when > 0.

    Python threads cannot be killed: on timeout the worker is abandoned as a
    daemon (it can no longer affect the trial, which is hard-failed) and a
    TimeoutError is raised so the generation keeps moving.
    """
    timeout = float(project.prepare_timeout or 0)
    if timeout <= 0:
        return project.prepare_trial(params, trial_dir)

    import threading

    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["prep"] = project.prepare_trial(params, trial_dir)
        except BaseException as exc:  # re-raised on the caller thread below
            result["exc"] = exc

    t = threading.Thread(target=_worker, daemon=True, name="sofaopt-prepare")
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(
            f"prepare_trial hook exceeded prepare_timeout={timeout:.0f}s "
            f"for {trial_dir.name}"
        )
    if "exc" in result:
        raise result["exc"]
    return result.get("prep")


def render_preview(
    image: Path,
    trial_dir: Path,
    gen_index: int,
    trial_index: int,
    previews_dir: Path,
    failed_preview: Path | None = None,
) -> None:
    """Publish a per-trial preview image into the trial dir and flat previews dir.

    If ``image`` is an ``.stl`` it is rendered offscreen with PyVista (requires
    the ``preview`` extra); otherwise it is copied as-is. Best-effort: failures
    fall back to ``failed_preview`` if provided.
    """
    import shutil

    previews_dir.mkdir(parents=True, exist_ok=True)
    local_path = trial_dir / "preview.png"
    flat_name = f"gen_{gen_index:04d}_trial_{trial_index:02d}.png"

    try:
        if image.suffix.lower() == ".stl":
            _render_stl(image, local_path)
        else:
            shutil.copy2(image, local_path)
        shutil.copy2(local_path, previews_dir / flat_name)
        print(f"[preview] Saved {flat_name}")
    except Exception as e:
        print(f"[warn] Preview failed for {image.name}: {e}")
        if failed_preview is not None and failed_preview.exists():
            try:
                shutil.copy2(failed_preview, local_path)
                shutil.copy2(local_path, previews_dir / flat_name)
            except Exception as fallback_err:
                print(f"[warn] Failed-preview fallback failed: {fallback_err}")


def _render_stl(stl_path: Path, out_png: Path) -> None:
    """Render an STL to a PNG offscreen. Imported lazily so PyVista stays optional."""
    import pyvista as pv  # type: ignore

    plotter = None
    try:
        mesh = pv.read(str(stl_path))
        if mesh.n_cells == 0 or mesh.n_points == 0:
            raise ValueError("mesh is empty")
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        plotter.add_mesh(mesh, color="#4a90d9", pbr=True, metallic=0.1, roughness=0.4)
        plotter.add_light(pv.Light(position=(200, 200, 400), intensity=0.8))
        plotter.background_color = "white"
        plotter.screenshot(str(out_png))
    finally:
        if plotter is not None:
            plotter.close()
