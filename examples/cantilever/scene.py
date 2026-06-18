"""SOFA scene for the cantilever example.

A soft beam is fixed at one end and sags under gravity. After a fixed horizon
we measure the free tip's vertical deflection and score how close it is to a
target. The whole optimizer contract is exercised in ~20 lines of logic:

    trial = open_trial(root)          # read sampled params (+ run metadata)
    E  = trial.params["young_modulus"]
    ... build scene using E ...
    trial.write_score(score, reason)  # report once, at the horizon

Run it directly under runSofa to debug (``open_trial`` returns empty params and
``is_optimizing == False``, so it just simulates without scoring/stopping).

Note: SOFA component/plugin names vary slightly between versions. This targets
SOFA v23.06+. If a RequiredPlugin name errors on your build, adjust the list.
"""

import math

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

TARGET_TIP_DEFLECTION = 12.0  # mm of downward tip drop we want to hit
HORIZON_STEPS = 250           # simulation steps before scoring

_PLUGINS = [
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mass",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Visual",
]


class TipController(Sofa.Core.Controller):
    """Counts steps; at the horizon, scores the tip deflection and stops."""

    def __init__(self, trial, dofs, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.dofs = dofs
        self.step = 0

    def onAnimateEndEvent(self, _event):
        self.step += 1
        # Surface live progress so the dashboard's Progress tab animates.
        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS},
            min_interval=0.2,
        )
        if self.step < HORIZON_STEPS:
            return

        pos = self.dofs.position.value
        x_max = max(p[0] for p in pos)
        tip_nodes = [p for p in pos if p[0] > x_max - 1e-6]
        tip_y = sum(p[1] for p in tip_nodes) / len(tip_nodes)
        deflection = -tip_y  # beam starts centered on y=0; sags negative
        score = 100.0 * math.exp(-abs(deflection - TARGET_TIP_DEFLECTION) / 4.0)
        reason = f"tip deflection {deflection:.2f} mm (target {TARGET_TIP_DEFLECTION})"

        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=reason)
        else:
            print(f"[cantilever] {reason} -> score {score:.2f}")


def createScene(root):
    trial = open_trial(root)
    young = float(trial.params.get("young_modulus", 5.0e4))
    poisson = float(trial.params.get("poisson_ratio", 0.30))

    root.dt = 0.02
    root.gravity = [0.0, -9810.0, 0.0]  # mm/s^2
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields")

    beam = root.addChild("beam")
    beam.addObject("EulerImplicitSolver", rayleighStiffness=0.1, rayleighMass=0.1)
    beam.addObject("CGLinearSolver", iterations=25, tolerance=1e-5, threshold=1e-5)
    beam.addObject(
        "RegularGridTopology",
        name="grid",
        nx=12, ny=3, nz=3,
        xmin=0, xmax=120, ymin=-6, ymax=6, zmin=-6, zmax=6,
    )
    dofs = beam.addObject("MechanicalObject", name="dofs", template="Vec3d")
    beam.addObject("UniformMass", totalMass=0.2)
    beam.addObject(
        "HexahedronFEMForceField",
        youngModulus=young,
        poissonRatio=poisson,
        method="large",
    )
    # Clamp the x=0 end.
    beam.addObject("BoxROI", name="clamp", box=[-1, -7, -7, 1, 7, 7], drawBoxes=False)
    beam.addObject("FixedConstraint", indices="@clamp.indices")

    beam.addObject(TipController(trial=trial, dofs=dofs, name="TipController"))
    return root
