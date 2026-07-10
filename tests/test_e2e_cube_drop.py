"""Real-SOFA end-to-end tests on the cube_drop example.

The unit suite deliberately fakes the SOFA process (``test_scene_contract.py``);
nothing there launches a real simulation. These tests close that gap: ONE
deterministic trial each, driven through the SAME primitives the optimizer uses
(``prepare_trial`` / ``init_trial_state`` / ``launch_sofa`` / ``wait_or_kill`` /
``read_trial_run``) and measured at the real interface — the run-slot dict in
``trial_state.json`` that the generation finalizer consumes.

Deterministic single-trial only: fixed params, fixed-step implicit Euler, no
contact response. Optimizer convergence is never banded here (samplers are
stochastic).

Skip behavior (public clones stay green): ``sofascene`` not installed → module
skip; no SOFA build → the ``sofa`` marker auto-skips (sofascene's pytest
plugin); no runSofa executable → fixture skip.

Replay the trial interactively (visible GUI, same physics, parameter defaults;
the scene freezes when the cube lands)::

    runSofa -l SofaPython3 -g imgui examples/cube_drop/scene.py

with ``src/`` on PYTHONPATH. The python-runner test records ``trial.mp4`` next
to its numeric outputs (pygame + ffmpeg); artifacts remain under the pytest tmp
root (kept for the last 3 runs) — the printed path in the test output locates
them.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("sofascene")  # provides the `sofa` marker + auto-skip

from sofaopt.core.archive import archive_run, load_archive_info
from sofaopt.core.sofa_runner import kill_process_tree, launch_sofa, wait_or_kill
from sofaopt.core.trial_state import init_trial_state, read_trial_run
from sofaopt.core.trialprep import prepare_trial

pytestmark = pytest.mark.sofa

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
EXAMPLE_DIR = HERE.parent / "examples" / "cube_drop"

# One fixed, mid-range trial: size 20 (bottom starts at y=50), mass 10
# (net accel g - 8/10). Obviously lands well before the 800-step horizon.
PARAMS = {"cube_size": 20.0, "cube_mass": 10.0}

# Provenance — measured 2026-07-10, Win 11 dev machine, SOFA build at
# %SOFA_ROOT% (runSofa Release), command:
#   python -m pytest tests/test_e2e_cube_drop.py -q
# Both runners landed at step 333 of 800 → score 58.375 exactly (same scene,
# same dt, fixed-step implicit Euler; ~17 s wall clock for the two tests).
# Band is ±20% around the measurement. NEVER widen this band to silence a
# failure — after an INTENTIONAL physics or scene change, re-measure and
# update this comment.
SCORE_BAND = (46.0, 70.0)

# Wall-clock wedge net only (~20x the measured few-second trial); the scene
# stops itself via ScoreWriter long before this.
TIMEOUT_S = 60.0


def _load_cube_project():
    spec = importlib.util.spec_from_file_location(
        "cube_drop_project_e2e", EXAMPLE_DIR / "project.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROJECT


def _find_runsofa() -> Path | None:
    exe = os.environ.get("RUNSOFA_EXE", "")
    if exe and Path(exe).is_file():
        return Path(exe)
    sofa_root = os.environ.get("SOFA_ROOT", "")
    if sofa_root:
        for sub in ("bin/Release/runSofa.exe", "bin/runSofa.exe", "bin/runSofa"):
            p = Path(sofa_root) / sub
            if p.is_file():
                return p
    found = shutil.which("runSofa")
    return Path(found) if found else None


def _scene_env(project) -> dict:
    """Child env: project's sofa_env over the live env, sofaopt importable."""
    env = os.environ.copy()
    env.update({k: str(v) for k, v in project.sofa_env.items()})
    extra = [str(SRC)]
    # The python runner is a plain interpreter: give it SOFA's bindings too.
    sofa_site = Path(os.environ.get("SOFA_ROOT", "")) / "lib" / "python3" / "site-packages"
    if sofa_site.is_dir():
        extra.append(str(sofa_site))
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra + ([existing] if existing else []))
    return env


def _run_single_trial(project) -> tuple[bool, dict | None, Path]:
    """One production-shaped trial launch; returns (exited, run_slot, trial_dir)."""
    trial_dir = project.trials_dir / "gen_0001" / "trial_01"
    trial_dir.mkdir(parents=True)
    trial_state_path = trial_dir / "trial_state.json"
    test = project.tests[0]

    init_trial_state(
        trial_state_path,
        gen_index=1,
        trial_index=1,
        run_plan=[(test.name, 1, 1)],
        params=PARAMS,
        test_weights={test.name: 1.0},
        test_max_scores={test.name: test.max_score},
    )
    prep = prepare_trial(project, PARAMS, trial_dir)

    proc = launch_sofa(
        project,
        scene_file=test.scene_file,
        test_name=test.name,
        test_run_index=1,
        test_run_total=1,
        trial_state_path=trial_state_path,
        params_path=trial_dir / "params.json",
        run_slot=1,
        gen_index=1,
        trial_index=1,
        run_index=1,
        env={**_scene_env(project), **prep.env},
    )
    try:
        exited = wait_or_kill(proc, TIMEOUT_S)
    finally:
        if proc.poll() is None:  # never leave a SOFA child behind
            kill_process_tree(proc)
    return exited, read_trial_run(trial_state_path, 1), trial_dir


def _assert_landed(exited: bool, run: dict | None, log_path: Path) -> None:
    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        )
    assert exited, f"trial hit the {TIMEOUT_S}s wall-clock kill; log tail:\n{log_tail}"
    assert run is not None, f"no run slot written to trial_state.json; log tail:\n{log_tail}"
    assert run.get("state") == "done", f"run did not finish: {run}\nlog tail:\n{log_tail}"
    assert str(run.get("reason", "")).startswith("landed at step"), run
    score = run.get("score")
    assert isinstance(score, (int, float))
    assert SCORE_BAND[0] <= score <= SCORE_BAND[1], (
        f"score {score} outside provenance band {SCORE_BAND} — investigate the "
        f"physics change before touching the band. Run: {run}"
    )


def test_runsofa_trial_scores_stops_and_archives(tmp_path):
    """The optimizer's runSofa path, end to end on real physics.

    Asserts the full per-trial contract a live campaign depends on: the scene
    scores itself into trial_state.json (ScoreWriter), the child STOPS ITSELF
    (write_score_and_stop's kill — not our wall-clock timeout), and
    archive_run() moves the finished runtime intact.
    """
    runsofa = _find_runsofa()
    if runsofa is None:
        pytest.skip("no runSofa executable (set RUNSOFA_EXE or SOFA_ROOT)")

    project = dataclasses.replace(
        _load_cube_project(), work_dir=tmp_path, runsofa_exe=runsofa
    )
    exited, run, trial_dir = _run_single_trial(project)
    print(f"[e2e] runSofa trial artifacts: {trial_dir}")
    _assert_landed(exited, run, trial_dir / "sofa_run1.log")

    # Archiving must move the finished run out of runtime/ without loss.
    archived = archive_run(project, name="e2e", notes="real-runSofa e2e trial")
    assert not project.runtime_dir.exists()
    moved_state = archived / "trials" / "gen_0001" / "trial_01" / "trial_state.json"
    assert moved_state.is_file()
    assert json.loads(moved_state.read_text(encoding="utf-8"))["runs"][0]["state"] == "done"
    info = load_archive_info(archived)
    assert info.name == "e2e"


def test_python_runner_trial_scores_and_records(tmp_path):
    """The in-process python runner path — same trial, plus the recording rule.

    Same deterministic physics through ``runner="python"`` with
    ``record_frames=True``: asserts the identical score band AND that the run
    recorded itself (trial.mp4 written next to trial_state.json via the
    pygame/ffmpeg FrameRecorder). Recording must never fail the run, so the
    score assertions hold even where mp4 support is absent; the mp4 assertion
    is gated on pygame + ffmpeg being available.
    """
    project = dataclasses.replace(
        _load_cube_project(),
        work_dir=tmp_path,
        runner="python",
        record_frames=True,
        record_frame_skip=16,
        record_frame_size=(320, 240),
    )
    exited, run, trial_dir = _run_single_trial(project)
    print(f"[e2e] python-runner trial artifacts: {trial_dir}")
    _assert_landed(exited, run, trial_dir / "sofa_run1.log")

    recordable = (
        importlib.util.find_spec("pygame") is not None
        and shutil.which("ffmpeg") is not None
    )
    if recordable:
        video = trial_dir / "trial.mp4"
        assert video.is_file() and video.stat().st_size > 0, (
            "recording stack present but no trial.mp4 was produced"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
