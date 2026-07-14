"""SOFA scene: cable-driven soft finger — actuated reach (SoftRobots plugin).

Adapted from the SoftRobots plugin's shipped finger part
(``softrobots/parts/finger/finger.py``: same volume mesh, fixing box, cable
geometry and material defaults), reworked for optimization: instead of
keyboard-driven cable jogging, a controller **ramps the cable displacement**
to a commanded value at a fixed rate (production-style walked command — no
teleports), waits for the fingertip to reach steady state, and scores how
closely the tip lands on a **target position measured once from reference
values** — an actuated-reach calibration mixing control (cable displacement),
material (Young's modulus) and design (cable pull-point) parameters.

Dropped from the source (documented on purpose): the three collision meshes
and the contact header — in the moderate-bend regime commanded here the finger
touches nothing (self-contact starts at much larger bends), while the
Lagrangian stack (FreeMotionAnimationLoop + constraint solver) stays because
``CableConstraint`` is a Lagrangian actuator.

Config from the trial params:
  - ``young_modulus``       finger silicone stiffness
  - ``cable_displacement``  commanded cable shortening (the control input)
  - ``pull_point_y``        y of the cable pull point (design parameter)
"""

import math
from pathlib import Path

import Sofa
import Sofa.Core
from softrobots.actuators import PullingCable
from splib3.loaders import loadPointListFromFile
from stlib3.physics.constraints import FixedBox
from stlib3.physics.deformable import ElasticMaterialObject

from sofaopt.scene import open_trial

# Geometry shipped with the SoftRobots plugin (finger extends toward -x).
FINGER_MESH_DIR = Path(__import__("softrobots.parts.finger", fromlist=["parts"]).__file__).parent / "mesh"
FIXING_BOX = [-1.0, -1.0, -1.0, 1.0, 15.0, 15.0]  # as in the source part

CABLE_RATE = 0.25          # cable shortening per step (walked command, no jumps)
HORIZON_STEPS = 800        # hard stop at dt=0.01; measured runs settle before
SETTLE_SPEED = 0.05        # |tip velocity| below this ...
SETTLE_STEPS = 15          # ... for this many consecutive steps = steady state

# Target fingertip steady-state position + score scale. MEASURED 2026-07-14
# (Win 11 dev machine, python runner, reference young_modulus=18000,
# cable_displacement=15, pull_point_y=0 — the shipped material and pull point —
# steady at step 255); command in the README. Re-measure after any INTENTIONAL
# physics change; never tweak to silence a failure.
TARGET_TIP = [-94.696, 33.445, -4.488]
SCORE_SCALE = 20.0         # score = 100 * exp(-|tip - target| / SCORE_SCALE);
#                            measured tip span over the param box is ~85 mm
#                            (soft sag [-102,-11] to stiff max-pull [-66,73]);
#                            the search-start defaults are ~19.7 mm out -> ~37,
#                            the far corners ~45-49 mm -> ~9-11, 1 mm -> 95.

_PLUGINS = [
    "SoftRobots",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Visual",
]


def score_from_tip(tip_xyz) -> float:
    """Map the steady-state tip position to the 0-100 objective."""
    err = math.dist([float(v) for v in tip_xyz], TARGET_TIP)
    return 100.0 * math.exp(-err / SCORE_SCALE)


class CableReach(Sofa.Core.Controller):
    """Ramps the cable to the commanded displacement, then scores the tip."""

    def __init__(self, trial, root, dofs, cable, displacement, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.dofs = dofs
        self.cable = cable
        self.displacement = float(displacement)
        self.tip_index = None  # resolved on the first step (prefab dofs fill at init)
        self.step = 0
        self.calm_steps = 0
        self.done = False

    def onAnimateBeginEvent(self, _event):
        if self.done:
            return
        current = float(self.cable.value.value[0])
        if current < self.displacement:
            self.cable.value = [min(current + CABLE_RATE, self.displacement)]

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        if self.tip_index is None:
            # Tip = the rest-position node farthest along -x (the finger axis).
            rest = self.dofs.rest_position.value
            self.tip_index = min(range(len(rest)), key=lambda i: float(rest[i][0]))
        self.step += 1
        ramping = float(self.cable.value.value[0]) < self.displacement
        vel = self.dofs.velocity.value[self.tip_index]
        speed = math.sqrt(sum(float(v) ** 2 for v in vel))
        self.calm_steps = 0 if ramping or speed >= SETTLE_SPEED else self.calm_steps + 1

        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS},
            min_interval=0.2,
        )

        if self.calm_steps >= SETTLE_STEPS:
            self.done = True
            self._report(f"steady at step {self.step}")
        elif self.step >= HORIZON_STEPS:
            self.done = True
            self._report(f"horizon at step {self.step} (slow settle)")

    def _report(self, reason):
        tip = self.dofs.position.value[self.tip_index]
        score = score_from_tip(tip)
        tip_str = "[" + ", ".join(f"{float(v):.3f}" for v in tip) + "]"
        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=f"{reason}; tip {tip_str}")
        else:
            print(f"[finger] {reason}; tip {tip_str} -> score {score:.2f}")
            self.root.animate = False


def createScene(root):
    trial = open_trial(root)
    young = float(trial.params.get("young_modulus", 18000.0))
    displacement = float(trial.params.get("cable_displacement", 15.0))
    pull_y = float(trial.params.get("pull_point_y", 0.0))

    root.dt = 0.01
    root.gravity = [0.0, -981.0, 0.0]  # as in the source part (mm units)
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields")
    # The Lagrangian header the source builds via stlib3's ContactHeader,
    # minus the collision pipeline (nothing to collide with here).
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("BlockGaussSeidelConstraintSolver", tolerance=1e-6, maxIterations=1000)

    finger = ElasticMaterialObject(
        name="Finger",
        volumeMeshFileName=str(FINGER_MESH_DIR / "finger.vtk"),
        poissonRatio=0.3,
        youngModulus=young,
        totalMass=0.5,
    )
    root.addChild(finger)
    FixedBox(finger, atPositions=FIXING_BOX, doVisualization=True)
    cable_node = PullingCable(
        finger,
        "PullingCable",
        pullPointLocation=[0.0, pull_y, 0.0],
        cableGeometry=loadPointListFromFile(str(FINGER_MESH_DIR / "cable.json")),
        valueType="displacement",
    )
    finger.addObject(
        CableReach(
            trial=trial, root=root, dofs=finger.dofs,
            cable=cable_node.CableConstraint, displacement=displacement,
            name="CableReach",
        )
    )
    return root
