"""Real-SOFA e2e on the soft_finger example (the soft-robot SOFA tier).

Two deterministic anchor trials through the SAME primitives the optimizer uses
(``init_trial_state`` / ``launch_sofa`` / ``wait_or_kill`` /
``read_trial_run``), python runner:

- the REFERENCE configuration must land the fingertip on the measured target
  (score ~100), proving the target's provenance still holds AND that the
  SoftRobots Lagrangian stack (FreeMotionAnimationLoop + CableConstraint +
  stlib3 prefabs) still runs headless;
- the search-start DEFAULTS must land in their measured mid-band (~37),
  proving the landscape has the documented hill to climb.

Skip behavior mirrors the cube_drop e2e: no sofascene → module skip; no SOFA
build → the ``sofa`` marker auto-skips. Additionally skips when the SoftRobots
python packages are not importable from SOFA's site-packages.

Interactive replay (visible GUI, default params, freezes at steady state)::

    runSofa -l SofaPython3 -g imgui examples/soft_finger/scene.py

with ``src/`` and SOFA's ``python3/site-packages`` on PYTHONPATH.
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
EXAMPLE_DIR = HERE.parent / "examples" / "soft_finger"

_SOFA_SITE = Path(os.environ.get("SOFA_ROOT", "")) / "lib" / "python3" / "site-packages"
if not (_SOFA_SITE / "softrobots").is_dir():
    pytest.skip("SoftRobots python packages not found under SOFA_ROOT", allow_module_level=True)

# Provenance — measured 2026-07-14, Win 11 dev machine, python runner, command:
#   python -m pytest tests/test_e2e_finger.py -q
# Reference (young=18000, displacement=15, pull_y=0 — the shipped material and
# pull point) reached steady state at step 255, tip [-94.696, 33.445, -4.488],
# score ~100; search defaults (young=30000, displacement=5, pull_y=10) scored
# 37.34 at step 168. Bands ±~20% (deterministic fixed-step physics; the slack
# covers build/platform drift). NEVER widen a band to silence a failure —
# re-measure after an INTENTIONAL physics change and update this comment.
REFERENCE_PARAMS = {"young_modulus": 18000.0, "cable_displacement": 15.0, "pull_point_y": 0.0}
REFERENCE_BAND = (95.0, 100.0)
DEFAULTS_PARAMS = {"young_modulus": 30000.0, "cable_displacement": 5.0, "pull_point_y": 10.0}
DEFAULTS_BAND = (30.0, 46.0)

TIMEOUT_S = 180.0  # wall-clock wedge net only; the scene stops itself (~2 s)


def _load_project():
    spec = importlib.util.spec_from_file_location(
        "finger_project_e2e", EXAMPLE_DIR / "project.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROJECT


def _scene_env(project) -> dict:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in project.sofa_env.items()})
    extra = [str(SRC), str(_SOFA_SITE)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([existing] if existing else []))
    return env


def _run_single_trial(project, params: dict) -> tuple[bool, dict | None, Path]:
    trial_dir = project.trials_dir / "gen_0001" / "trial_01"
    trial_dir.mkdir(parents=True)
    trial_state_path = trial_dir / "trial_state.json"
    test = project.tests[0]

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


def _assert_steady(exited: bool, run: dict | None, band: tuple, log_path: Path) -> None:
    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        )
    assert exited, f"trial hit the {TIMEOUT_S}s wall-clock kill; log tail:\n{log_tail}"
    assert run is not None, f"no run slot written; log tail:\n{log_tail}"
    assert run.get("state") == "done", f"run did not finish: {run}\nlog tail:\n{log_tail}"
    assert str(run.get("reason", "")).startswith("steady at step"), run
    score = run.get("score")
    assert isinstance(score, (int, float))
    assert band[0] <= score <= band[1], (
        f"score {score} outside provenance band {band} — investigate the physics "
        f"change before touching the band. Run: {run}"
    )


def test_reference_configuration_reaches_the_measured_tip(tmp_path):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_trial(project, REFERENCE_PARAMS)
    print(f"[e2e] finger reference: {run and run.get('reason')}")
    _assert_steady(exited, run, REFERENCE_BAND, trial_dir / "sofa_run1.log")


def test_search_defaults_land_in_the_documented_mid_band(tmp_path):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_trial(project, DEFAULTS_PARAMS)
    print(f"[e2e] finger defaults: {run and run.get('reason')}")
    _assert_steady(exited, run, DEFAULTS_BAND, trial_dir / "sofa_run1.log")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
