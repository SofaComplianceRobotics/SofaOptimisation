"""Real-SOFA e2e on the trunk_control example (the control study tier).

Two deterministic anchor trials through the SAME primitives the optimizer uses
(``init_trial_state`` / ``launch_sofa`` / ``wait_or_kill`` / ``read_trial_run``),
python runner:

- the REFERENCE actuation must reproduce the measured backbone (score ~100),
  proving the target's provenance holds AND that the SoftRobots Lagrangian
  cable stack still runs bit-deterministic headless;
- the blind start (all cables slack) must land in its measured mid-band (~22),
  proving the landscape has the documented hill to climb.

Skip behavior mirrors the finger e2e: no sofascene → module skip; no SOFA
build → the ``sofa`` marker auto-skips; no SoftRobots packages or trunk mesh →
module skip.

Interactive replay (visible GUI, default = all cables slack, freezes when
settled)::

    runSofa -l SofaPython3 -g imgui examples/trunk_control/scene.py

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
EXAMPLE_DIR = HERE.parent / "examples" / "trunk_control"

_SOFA_SITE = Path(os.environ.get("SOFA_ROOT", "")) / "lib" / "python3" / "site-packages"
if not (_SOFA_SITE / "softrobots").is_dir():
    pytest.skip("SoftRobots python packages not found under SOFA_ROOT", allow_module_level=True)

# Provenance — measured 2026-07-14, Win 11 dev machine, python runner, command:
#   python -m pytest tests/test_e2e_trunk.py -q
# Reference actuation (cable_L0=30, cable_S1=20, others slack) settled at step
# 79, backbone rms ~0, score ~100 (bit-deterministic: two runs rms diff 0.0);
# the blind start (all cables slack) scored 22.33 (rms 59.97) at step 43.
# Bands +-~20% (deterministic fixed-step physics; slack covers build/platform
# drift). NEVER widen a band to silence a failure — re-measure after an
# INTENTIONAL physics change and update this comment.
REFERENCE_PARAMS = {
    "cable_L0": 30.0, "cable_L1": 0.0, "cable_L2": 0.0, "cable_L3": 0.0,
    "cable_S0": 0.0, "cable_S1": 20.0, "cable_S2": 0.0, "cable_S3": 0.0,
}
REFERENCE_BAND = (95.0, 100.0)
SLACK_PARAMS = {name: 0.0 for name in REFERENCE_PARAMS}
SLACK_BAND = (17.0, 28.0)

TIMEOUT_S = 240.0  # wall-clock wedge net only; the scene stops itself (~3 s)


def _load_project():
    spec = importlib.util.spec_from_file_location(
        "trunk_project_e2e", EXAMPLE_DIR / "project.py"
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


def test_reference_actuation_reproduces_the_measured_backbone(tmp_path):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_trial(project, REFERENCE_PARAMS)
    print(f"[e2e] trunk reference: {run and run.get('reason')}")
    _assert_settled(exited, run, REFERENCE_BAND, trial_dir / "sofa_run1.log")


def test_slack_start_lands_in_the_documented_mid_band(tmp_path):
    project = dataclasses.replace(_load_project(), work_dir=tmp_path)
    exited, run, trial_dir = _run_single_trial(project, SLACK_PARAMS)
    print(f"[e2e] trunk slack: {run and run.get('reason')}")
    _assert_settled(exited, run, SLACK_BAND, trial_dir / "sofa_run1.log")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
