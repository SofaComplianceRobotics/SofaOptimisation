"""SOFA scene: the analytic landscape harness as a viewable trial.

The score is a classic optimization test function of the params (see
``benchmark_functions.py``) — this scene wraps that pure objective in a real,
tiny SOFA simulation so the whole optimizer path executes end to end (subprocess
launch, ``trial_state.json`` contract, live progress, frame recording) and so a
trial is watchable: a marker mass settles at a height proportional to the score,
i.e. the higher it rests, the better the candidate.

Launched by hand (no optimizer) it settles at the default params' score.

Config comes from the trial params / env:
  - ``x0..x{d-1}``  the searched coordinates (floats)
  - env ``OPT_LANDSCAPE_FN``     which benchmark function (default "rastrigin")
  - env ``OPT_LANDSCAPE_NOISE``  per-run Gaussian score noise sigma (default 0)
"""

import random
import sys
from pathlib import Path

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

# examples/ is not an installed package — load the sibling functions module by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_functions as bf

SETTLE_STEPS = 60          # short real simulation before scoring
FLOOR_Y = 0.0
MAX_HEIGHT = 100.0         # marker rest height for a perfect (100) score


def _params_vector(trial) -> list[float]:
    """The searched coordinates x0..x{d-1}, in order, from the trial params."""
    xs = []
    i = 0
    while f"x{i}" in trial.params:
        xs.append(float(trial.params[f"x{i}"]))
        i += 1
    return xs


class Scorer(Sofa.Core.Controller):
    """Settles a marker to a height ∝ score, then reports it and stops."""

    def __init__(self, trial, root, dofs, function, noise_sigma, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.dofs = dofs
        self.function = function
        self.noise_sigma = noise_sigma
        self.x = _params_vector(trial)
        # Per-run RNG seeded by the run slot so repeats are i.i.d. noise samples.
        seed = int(trial.env.get("OPT_RUN_SLOT", "1"))
        self.rng = random.Random(seed)
        self.step = 0
        self.done = False

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        self.step += 1
        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": SETTLE_STEPS},
            min_interval=0.2,
        )
        if self.step < SETTLE_STEPS:
            return
        self.done = True
        score = bf.scored(self.function, self.x, self.noise_sigma, self.rng)
        # Park the marker at a height that visualizes the score.
        target_y = FLOOR_Y + MAX_HEIGHT * (score / 100.0)
        pos = self.dofs.position.value.copy()
        pos[0][1] = target_y
        self.dofs.position.value = pos
        self._report(score)

    def _report(self, score):
        reason = f"{self.function}({[round(v, 3) for v in self.x]}) -> {score:.2f}"
        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=reason)
        else:
            print(f"[landscape] {reason}")
            self.root.animate = False


def createScene(root):
    trial = open_trial(root)
    function = trial.env.get("OPT_LANDSCAPE_FN", "rastrigin")
    noise_sigma = float(trial.env.get("OPT_LANDSCAPE_NOISE", "0") or 0.0)

    root.dt = 0.02
    root.gravity = [0.0, 0.0, 0.0]  # the marker is placed analytically, not dropped
    root.addObject("DefaultAnimationLoop")
    root.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
    root.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    root.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Iterative")
    root.addObject("RequiredPlugin", name="Sofa.Component.Mass")
    root.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")
    root.addObject("VisualStyle", displayFlags="showBehaviorModels")

    marker = root.addChild("marker")
    marker.addObject("EulerImplicitSolver")
    marker.addObject("CGLinearSolver", iterations=25, tolerance=1e-5, threshold=1e-5)
    dofs = marker.addObject(
        "MechanicalObject", name="dofs", template="Vec3",
        position=[0.0, 0.0, 0.0], showObject=True, showObjectScale=8.0,
    )
    marker.addObject("UniformMass", totalMass=1.0)
    marker.addObject(
        Scorer(trial=trial, root=root, dofs=dofs, function=function,
               noise_sigma=noise_sigma, name="Scorer")
    )
    return root
