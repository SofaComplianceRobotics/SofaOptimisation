"""Real-SOFA e2e on the caduceus_settle example (the heavy SOFA tier).

Two deterministic anchor trials through the SAME primitives the optimizer uses
(``init_trial_state`` / ``launch_sofa`` / ``wait_or_kill`` /
``read_trial_run``), python runner:

- the REFERENCE values must reproduce the measured wrapped pose (score ~100),
  proving the target's provenance still holds AND that the frictional-contact
  stack (CollisionPipeline + LCPConstraintSolver) still runs bit-deterministic
  headless;
- the search-start DEFAULTS must land in their measured mid-band (~36),
  proving the landscape has the documented hill to climb.

Skip behavior mirrors the cube_drop e2e: no sofascene → module skip; no SOFA
build → the ``sofa`` marker auto-skips.

Interactive replay (visible GUI, default params, stops at the horizon)::

    runSofa -l SofaPython3 -g imgui examples/caduceus_settle/scene.py

with ``src/`` on PYTHONPATH.
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
EXAMPLE_DIR = HERE.parent / "examples" / "caduceus_settle"

# Provenance — measured 2026-07-14, Win 11 dev machine, python runner, command:
#   python -m pytest tests/test_e2e_caduceus.py -q
# Reference (young=30000, mu=0.2, mass=1.0 — the caduceus.scn values) observed
# at the fixed horizon step 600, rms ~0, score ~100 (bit-deterministic: two
# measurement runs differed by rms 0.0); search defaults (young=60000, mu=0.4,
# mass=1.0) scored 36.04 (rms 2.04). Bands ±~20% (deterministic fixed-step
# physics; the slack covers build/platform drift — contact scenes are the most
# drift-sensitive tier, so investigate any excursion carefully). NEVER widen a
# band to silence a failure — re-measure after an INTENTIONAL physics change
# and update this comment.
REFERENCE_PARAMS = {"young_modulus": 30000.0, "friction_mu": 0.2, "total_mass": 1.0}
REFERENCE_BAND = (95.0, 100.0)
DEFAULTS_PARAMS = {"young_modulus": 60000.0, "friction_mu": 0.4, "total_mass": 1.0}
DEFAULTS_BAND = (29.0, 44.0)

TIMEOUT_S = 300.0  # wall-clock wedge net only; the scene stops itself (~2.5 s)


def _load_project():
    spec = importlib.util.spec_from_file_location(
        "cadu_project_e2e", EXAMPLE_DIR / "project.py"
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


def _assert_observed(exited: bool, run: dict | None, band: tuple, log_path: Path) -> None:
    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        )
    assert exited, f"trial hit the {TIMEOUT_S}s wall-clock kill; log tail:\n{log_tail}"
    assert run is not None, f"no run slot written; log tail:\n{log_tail}"
    assert run.get("state") == "done", f"run did not finish: {run}\nlog tail:\n{log_tail}"
    reason = str(run.get("reason", ""))
    assert reason.startswith(("observed at step", "settled at step")), run
    score = run.get("score")
    assert isinstance(score, (int, float))
    assert band[0] <= score <= band[1], (
        f"score {score} outside provenance band {band} — investigate the physics "
        f"change before touching the band. Run: {run}"
    )


def test_reference_values_reproduce_the_measured_wrapped_pose(tmp_path):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_trial(project, REFERENCE_PARAMS)
    print(f"[e2e] caduceus reference: {run and run.get('reason')}")
    _assert_observed(exited, run, REFERENCE_BAND, trial_dir / "sofa_run1.log")


def test_search_defaults_land_in_the_documented_mid_band(tmp_path):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_trial(project, DEFAULTS_PARAMS)
    print(f"[e2e] caduceus defaults: {run and run.get('reason')}")
    _assert_observed(exited, run, DEFAULTS_BAND, trial_dir / "sofa_run1.log")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
