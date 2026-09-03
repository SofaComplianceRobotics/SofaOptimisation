"""Real-SOFA e2e on the liver_elastography example (the in-depth study tier).

Deterministic anchor trials through the SAME primitives the optimizer uses
(``init_trial_state`` / ``launch_sofa`` / ``wait_or_kill`` / ``read_trial_run``),
python runner:

- the REFERENCE field must reproduce the measured target for all three load
  cases (score ~100), proving per-element youngModulus still takes effect, the
  committed region partition still matches, and the targets' provenance holds;
- a blind uniform field must land in its measured mid-band (~46 supine),
  proving the landscape has the documented hill to climb.

Skip behavior mirrors the cube_drop e2e: no sofascene → module skip; no SOFA
build → the ``sofa`` marker auto-skips.

Interactive replay (visible GUI, default params, supine, freezes when settled)::

    runSofa -l SofaPython3 -g imgui examples/liver_elastography/scene.py

with ``src/`` on PYTHONPATH; set ``OPT_LIVER_CASE=lateral|tilt`` for the others.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("sofascene")  # provides the `sofa` marker + auto-skip

from sofaopt.core.sofa_runner import kill_process_tree, launch_sofa, wait_or_kill
from sofaopt.core.trial_state import init_trial_state, read_trial_run

pytestmark = pytest.mark.sofa

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
EXAMPLE_DIR = HERE.parent / "examples" / "liver_elastography"

# Provenance — measured 2026-07-14, Win 11 dev machine, python runner, command:
#   python -m pytest tests/test_e2e_elastography.py -q
# Reference field (E=3000 except soft lesion region 5 at E=1000, nu=0.30,
# rho=1.0) settled supine 310 / lateral 300 / tilt 215, rms ~0, score ~100 all
# three; a blind uniform field (E=5000 everywhere, nu=0.10) scored 46.15 supine.
# Bands +-~20% (deterministic fixed-step physics; slack covers build/platform
# drift). NEVER widen a band to silence a failure — re-measure after an
# INTENTIONAL physics change and update this comment.
_N_REGIONS = 6
_LESION = 5
REFERENCE_PARAMS = {
    **{f"young_region_{k}": 3000.0 for k in range(_N_REGIONS)},
    f"young_region_{_LESION}": 1000.0,
    "poisson_ratio": 0.30,
    "mass_density": 1.0,
}
REFERENCE_BAND = (95.0, 100.0)
UNIFORM_PARAMS = {
    **{f"young_region_{k}": 5000.0 for k in range(_N_REGIONS)},
    "poisson_ratio": 0.10,
    "mass_density": 1.0,
}
UNIFORM_SUPINE_BAND = (38.0, 55.0)

TIMEOUT_S = 120.0  # wall-clock wedge net only; the scene stops itself (~1 s)


def _load_project():
    spec = importlib.util.spec_from_file_location(
        "elasto_project_e2e", EXAMPLE_DIR / "project.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROJECT


def _scene_env(project) -> dict:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in project.sofa_env.items()})
    extra = [str(SRC)]
    sofa_site = Path(os.environ.get("SOFA_ROOT", "")) / "lib" / "python3" / "site-packages"
    if sofa_site.is_dir():
        extra.append(str(sofa_site))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([existing] if existing else []))
    return env


def _run_single_case(project, params: dict, case: str) -> tuple[bool, dict | None, Path]:
    trial_dir = project.trials_dir / "gen_0001" / "trial_01"
    trial_dir.mkdir(parents=True)
    trial_state_path = trial_dir / "trial_state.json"
    test = next(t for t in project.tests if t.name == case)

    init_trial_state(
        trial_state_path, gen_index=1, trial_index=1,
        run_plan=[(test.name, 1, 1)], params=params,
        test_weights={test.name: 1.0}, test_max_scores={test.name: test.max_score},
    )
    (trial_dir / "params.json").write_text(json.dumps(params), encoding="utf-8")

    proc = launch_sofa(
        project, scene_file=test.scene_file, test_name=test.name,
        test_run_index=1, test_run_total=1, trial_state_path=trial_state_path,
        params_path=trial_dir / "params.json", run_slot=1,
        gen_index=1, trial_index=1, run_index=1, env=_scene_env(project),
    )
    try:
        exited = wait_or_kill(proc, TIMEOUT_S)
    finally:
        if proc.poll() is None:  # never leave a SOFA child behind
            kill_process_tree(proc)
    return exited, read_trial_run(trial_state_path, 1), trial_dir


def _assert_settled(exited: bool, run: dict | None, band: tuple, log_path: Path) -> None:
    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        )
    assert exited, f"trial hit the {TIMEOUT_S}s wall-clock kill; log tail:\n{log_tail}"
    assert run is not None, f"no run slot written; log tail:\n{log_tail}"
    assert run.get("state") == "done", f"run did not finish: {run}\nlog tail:\n{log_tail}"
    assert str(run.get("reason", "")).startswith("settled at step"), run
    score = run.get("score")
    assert isinstance(score, (int, float))
    assert band[0] <= score <= band[1], (
        f"score {score} outside provenance band {band} — investigate the physics "
        f"change before touching the band. Run: {run}"
    )


@pytest.mark.parametrize("case", ["supine", "lateral", "tilt"])
def test_reference_field_hits_the_measured_target(tmp_path, case):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_case(project, REFERENCE_PARAMS, case)
    print(f"[e2e] elastography reference/{case}: {run and run.get('reason')}")
    _assert_settled(exited, run, REFERENCE_BAND, trial_dir / "sofa_run1.log")


def test_uniform_field_lands_in_the_documented_mid_band(tmp_path):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_case(project, UNIFORM_PARAMS, "supine")
    print(f"[e2e] elastography uniform/supine: {run and run.get('reason')}")
    _assert_settled(exited, run, UNIFORM_SUPINE_BAND, trial_dir / "sofa_run1.log")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
