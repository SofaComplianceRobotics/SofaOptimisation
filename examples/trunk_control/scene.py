"""SOFA scene: cable-driven soft trunk — 8-cable control allocation.

Adapted from the SoftRobots plugin's shipped Trunk tutorial
(``examples/tutorials/Trunk/trunk.py``: same VTK mesh, FEM material, fixed
extremity and the 8-cable layout — 4 long + 4 short). Reworked for
optimization: a controller **ramps all 8 cable displacements** to their
commanded values at a fixed rate (production-style walked command), waits for
the trunk to settle, and scores how closely its **backbone curve** matches a
target backbone measured once from a reference actuation — the classic
**control-allocation** problem (8 actuators shaping one continuum).

This is the control-flavored in-depth study platform (companion to
``liver_elastography``, which is the identification-flavored one): 8 coupled
control inputs, a heavily redundant map (8 cables → a low-dimensional shape),
and antagonism (opposing cables cancel) — the setting where an optimizer's
covariance and restart machinery meet a genuinely non-convex control landscape.

Dropped from the source (documented on purpose): visual + collision models and
the AnimationManager — the observation is the mechanical backbone; the
Lagrangian stack (FreeMotionAnimationLoop + constraint solver) stays because
``CableConstraint`` is a Lagrangian actuator.

Config from the trial params:
  - ``cable_L0..L3``  displacement of the 4 long cables
  - ``cable_S0..S3``  displacement of the 4 short cables
"""

import math
import os
from pathlib import Path

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

HERE = Path(__file__).resolve().parent

# 8 cables in the source order: 4 long (cableL0..3) then 4 short (cableS0..3).
CABLE_NAMES = [f"cableL{i}" for i in range(4)] + [f"cableS{i}" for i in range(4)]
PARAM_NAMES = [f"cable_L{i}" for i in range(4)] + [f"cable_S{i}" for i in range(4)]

CABLE_RATE = 0.5           # displacement per step (walked command, no jumps)
HORIZON_STEPS = 700        # hard stop at dt=0.01; measured settles land before
SETTLE_SPEED = 0.5         # max |backbone velocity| below this ...
SETTLE_STEPS = 15          # ... for this many consecutive steps = settled

# Backbone tracking: the trunk axis runs along +z to ~195. We sample the node
# nearest the axis in each of these z-bands, resolved once from rest positions.
BACKBONE_Z = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

# Target backbone + score scale. MEASURED 2026-07-14 (Win 11 dev machine,
# python runner) from the reference actuation (cable_L0=30, cable_S1=20, all
# other cables 0; settled at step 79) — command in the README. Re-measure
# after any INTENTIONAL physics change; never tweak to silence a failure.
TARGET_BACKBONE = [
    [-0.72, -2.293, 31.323], [-4.122, -15.663, 48.639],
    [-22.12, -36.614, 63.842], [-44.647, -36.453, 74.109],
    [-56.851, -23.118, 95.065], [-42.273, -2.891, 100.713],
]
SCORE_SCALE = 40.0         # score = 100 * exp(-rms / SCORE_SCALE); rms is the
#                            root-mean-square backbone-point distance to target.
#                            Measured: blind start (all cables slack) rms 60.0
#                            -> 22; the opposite actuation rms 97.6 -> 9;
#                            reference rms 0 -> 100 (trunk backbone ~195 long).

_PLUGINS = [
    "SoftRobots",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Visual",
]


def find_trunk_mesh() -> Path:
    """Locate the SoftRobots Trunk tutorial mesh across install/build layouts."""
    override = os.environ.get("OPT_TRUNK_MESH_DIR")
    candidates = [Path(override)] if override else []
    root = Path(os.environ.get("SOFA_ROOT", ""))
    rel = "plugins/SoftRobots/examples/tutorials/Trunk/mesh"
    candidates += [root / rel, root.parent / rel, root / "share" / rel]
    for cand in candidates:
        if (cand / "trunk.vtk").is_file():
            return cand / "trunk.vtk"
    raise FileNotFoundError(
        "trunk.vtk not found; set OPT_TRUNK_MESH_DIR to the SoftRobots Trunk "
        f"tutorial mesh directory (tried: {[str(c) for c in candidates]})"
    )


def _cable_geometry():
    """The 8 cables' pull points + guide positions (from trunk.py, verbatim)."""
    from splib3.numerics import Quat, Vec3

    length1, length2, length_trunk = 10.0, 2.0, 195.0
    pull = [[0., length1, 0.], [-length1, 0., 0.], [0., -length1, 0.], [length1, 0., 0.]]
    direction = Vec3(0., length2 - length1, length_trunk)
    direction.normalize()

    geoms = []  # (pull_point, positions) for each of the 8 cables
    for count, npts in ((4, 20), (4, 10)):  # long cables then short cables
        for i in range(count):
            q = Quat(0., 0., math.sin(1.57 * i / 2.), math.cos(1.57 * i / 2.))
            position = [[0., 0., 0.]] * npts
            for k in range(0, npts - 1, 2):
                v = Vec3(direction[0], direction[1] * 17.5 * (k / 2) + length1,
                         direction[2] * 17.5 * (k / 2) + 21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1] * 17.5 * (k / 2) + length1,
                         direction[2] * 17.5 * (k / 2) + 27)
                position[k + 1] = v.rotateFromQuat(q)
            geoms.append((pull[i], [p.toList() for p in position]))
    return geoms


def rms_to_target(backbone, target) -> float:
    total = 0.0
    for pt, tgt in zip(backbone, target, strict=True):
        total += sum((float(a) - float(b)) ** 2 for a, b in zip(pt, tgt, strict=True))
    return math.sqrt(total / len(target))


def score_from_rms(rms: float) -> float:
    return 100.0 * math.exp(-rms / SCORE_SCALE)


class TrunkAllocator(Sofa.Core.Controller):
    """Ramps 8 cables to commanded displacements, scores the backbone."""

    def __init__(self, trial, root, dofs, cables, commands, dump_path, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.dofs = dofs
        self.cables = cables
        self.commands = commands
        self.dump_path = dump_path
        self.backbone_idx = None  # resolved on the first step (rest positions)
        self.step = 0
        self.calm_steps = 0
        self.prev = None
        self.done = False

    def _resolve_backbone(self):
        rest = self.dofs.rest_position.value
        idx = []
        for z in BACKBONE_Z:
            idx.append(min(
                range(len(rest)),
                key=lambda i: (float(rest[i][2]) - z) ** 2
                + float(rest[i][0]) ** 2 + float(rest[i][1]) ** 2,
            ))
        return idx

    def _backbone(self):
        pos = self.dofs.position.value
        return [[float(v) for v in pos[i]] for i in self.backbone_idx]

    def onAnimateBeginEvent(self, _event):
        if self.done:
            return
        for cable, target in zip(self.cables, self.commands, strict=True):
            cur = float(cable.value.value[0])
            if cur < target:
                cable.value = [min(cur + CABLE_RATE, target)]

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        if self.backbone_idx is None:
            self.backbone_idx = self._resolve_backbone()
        self.step += 1
        ramping = any(
            float(c.value.value[0]) < t for c, t in zip(self.cables, self.commands, strict=True)
        )
        bb = self._backbone()
        if self.prev is None:
            speed = math.inf
        else:
            speed = max(math.dist(a, b) for a, b in zip(bb, self.prev, strict=True))
        self.prev = bb
        self.calm_steps = 0 if ramping or speed >= SETTLE_SPEED else self.calm_steps + 1

        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS},
            min_interval=0.2,
        )

        if self.calm_steps >= SETTLE_STEPS:
            self.done = True
            self._report(bb, f"settled at step {self.step}")
        elif self.step >= HORIZON_STEPS:
            self.done = True
            self._report(bb, f"horizon at step {self.step} (slow settle)")

    def _report(self, backbone, reason):
        if self.dump_path:  # measurement mode: record the backbone
            import json
            Path(self.dump_path).write_text(
                json.dumps({"backbone": backbone}), encoding="utf-8"
            )
        if TARGET_BACKBONE is None:
            score, detail = 0.0, "no target yet (measurement mode)"
        else:
            rms = rms_to_target(backbone, TARGET_BACKBONE)
            score, detail = score_from_rms(rms), f"rms {rms:.4f}"
        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=f"{reason}; {detail}")
        else:
            print(f"[trunk] {reason}; {detail} -> score {score:.2f}")
            self.root.animate = False


def createScene(root):
    trial = open_trial(root)
    commands = [float(trial.params.get(p, 0.0)) for p in PARAM_NAMES]
    young = float(trial.params.get("young_modulus", 450.0))
    mesh = find_trunk_mesh()

    root.dt = 0.01
    root.gravity = [0.0, -9810.0, 0.0]  # as in trunk.py (mm units)
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("VisualStyle", displayFlags="showBehaviorModels")
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("BlockGaussSeidelConstraintSolver", maxIterations=100, tolerance=1e-5)

    sim = root.addChild("Simulation")
    sim.addObject("EulerImplicitSolver", name="odesolver", rayleighMass=0.1, rayleighStiffness=0.1)
    sim.addObject("SparseLDLSolver", name="precond")
    sim.addObject("GenericConstraintCorrection")

    trunk = sim.addChild("Trunk")
    trunk.addObject("MeshVTKLoader", name="loader", filename=str(mesh))
    trunk.addObject("MeshTopology", src="@loader", name="container")
    dofs = trunk.addObject("MechanicalObject", name="dofs", template="Vec3")
    trunk.addObject("UniformMass", totalMass=0.042)
    trunk.addObject(
        "TetrahedronFEMForceField", template="Vec3", name="FEM", method="large",
        poissonRatio=0.45, youngModulus=young,
    )
    trunk.addObject("BoxROI", name="boxROI", box=[[-20, -20, 0], [20, 20, 20]], drawBoxes=False)
    trunk.addObject(
        "PartialFixedProjectiveConstraint", fixedDirections=[1, 1, 1],
        indices="@boxROI.indices",
    )

    cables = []
    for name, (pull, positions) in zip(CABLE_NAMES, _cable_geometry(), strict=True):
        node = trunk.addChild(name)
        node.addObject("MechanicalObject", name="dofs", position=[pull, *positions])
        cable = node.addObject(
            "CableConstraint", template="Vec3", name="cable", hasPullPoint=False,
            indices=list(range(len(positions) + 1)), maxPositiveDisp=70,
            maxDispVariation=1, minForce=0,
        )
        node.addObject("BarycentricMapping", name="mapping", mapForces=False, mapMasses=False)
        cables.append(cable)

    trunk.addObject(
        TrunkAllocator(
            trial=trial, root=root, dofs=dofs, cables=cables, commands=commands,
            dump_path=trial.env.get("OPT_TRUNK_DUMP", ""), name="TrunkAllocator",
        )
    )
    return root
