"""SOFA scene: caduceus — contact-parameter identification from the wrapped pose.

Adapted from SOFA's shipped ``examples/Demos/caduceus.scn`` (the iconic demo:
a deformable FEM snake drops onto the SOFA pod and wraps itself around it
under frictional contact, LCP constraint solver). Reworked for optimization:
the snake's **final wrapped pose** (all sparse-grid FEM dofs at rest) is the
observation; the optimizer must recover (young_modulus, friction_mu,
total_mass) that reproduce a **target pose measured once from the shipped
values** — contact-parameter identification from an observed rest pose.

This is the *heavy* validation tier: real collision pipeline + LCP friction
contact per step, trials of tens of seconds — the rung that exercises
wall-clock timeout wedges, parallel launches under load and
restart/convergence logic at realistic budgets.

Deviations from the source ``.scn`` (documented on purpose): visuals, lights
and camera are dropped (headless observation is the mechanical dofs);
``parallelCollisionDetectionAndFreeMotion`` is disabled so trials are
deterministic (provenance bands require reproducible physics).

Config from the trial params / env:
  - params ``young_modulus``, ``friction_mu``, ``total_mass``
  - env ``OPT_CADU_DUMP``  measurement mode: path to write the settled
                           pose JSON (used once to create target.json)
"""

import json
import math
import os
from pathlib import Path

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

HERE = Path(__file__).resolve().parent

HORIZON_STEPS = 600        # THE observation time: the pose is scored at this
#                            fixed step (24 s sim at dt=0.04). Contact chatter
#                            keeps a residual max dof speed of ~0.5-1.4 long
#                            after the pose is stable (measured: mean_y flat
#                            from ~step 440), so unlike the material tiers the
#                            protocol here is a fixed-horizon observation, not
#                            a settle gate.
SETTLE_SPEED = 0.05        # ... but if max |dof velocity| stays below this ...
SETTLE_STEPS = 15          # ... this many steps, motion is truly dead -> exit
#                            early (same observation, cheaper trial).

# Target wrapped pose. MEASURED 2026-07-14 from the shipped values
# (youngModulus=30000, mu=0.2, totalMass=1.0), bit-deterministic (two runs,
# rms diff 0.0); provenance in target.json / README. Re-measure after any
# INTENTIONAL physics change; never tweak to silence a failure.
TARGET_FILE = HERE / "target.json"
SCORE_SCALE = 2.0          # score = 100 * exp(-rms / SCORE_SCALE); rms is the
#                            root-mean-square dof distance to the target pose.
#                            Measured spans: search defaults (E=60000, mu=0.4)
#                            rms 2.04 -> 36; mild offset rms 0.95 -> 62; box
#                            corners rms 3.2-6.6 -> 4-20; below mu ~0.05 the
#                            snake slides OFF the pod (rms 57000 -> 0): a real
#                            basin cliff, not a smooth ridge.

_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.LinearSystem",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.Visual",
]


def find_mesh_dir() -> Path:
    """Locate SOFA's shared mesh directory across install/build-tree layouts."""
    override = os.environ.get("OPT_SOFA_MESH_DIR")
    candidates = [Path(override)] if override else []
    root = Path(os.environ.get("SOFA_ROOT", ""))
    candidates += [
        root / "share" / "sofa" / "mesh",       # installed layout
        root / "share" / "mesh",
        root.parent / "src" / "share" / "mesh",  # build tree next to a src checkout
        root / "src" / "share" / "mesh",
    ]
    for cand in candidates:
        if (cand / "snake_body.obj").is_file():
            return cand
    raise FileNotFoundError(
        "snake_body.obj not found; set OPT_SOFA_MESH_DIR to SOFA's share/mesh "
        f"directory (tried: {[str(c) for c in candidates]})"
    )


def load_target() -> list | None:
    """Settled target pose, or None before measurement."""
    if not TARGET_FILE.is_file():
        return None
    return json.loads(TARGET_FILE.read_text(encoding="utf-8")).get("positions")


def rms_to_target(positions, target) -> float:
    """RMS dof distance to the target pose."""
    total = 0.0
    for pos, tgt in zip(positions, target, strict=True):
        total += sum((float(a) - float(b)) ** 2 for a, b in zip(pos, tgt, strict=True))
    return math.sqrt(total / len(target))


def score_from_rms(rms: float) -> float:
    return 100.0 * math.exp(-rms / SCORE_SCALE)


class Settler(Sofa.Core.Controller):
    """Waits for the snake to come to rest, then scores the pose and stops."""

    def __init__(self, trial, root, dofs, target, dump_path, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.dofs = dofs
        self.target = target
        self.dump_path = dump_path
        self.step = 0
        self.calm_steps = 0
        self.done = False
        self.speed_trace = []  # (step, max_speed, mean_y) samples, dump mode only

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        self.step += 1
        vel = self.dofs.velocity.value
        max_speed = max(math.sqrt(sum(float(v) ** 2 for v in row)) for row in vel)
        self.calm_steps = self.calm_steps + 1 if max_speed < SETTLE_SPEED else 0
        if self.dump_path and self.step % 20 == 0:
            pos = self.dofs.position.value
            mean_y = sum(float(p[1]) for p in pos) / len(pos)
            self.speed_trace.append((self.step, round(max_speed, 5), round(mean_y, 3)))

        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS},
            min_interval=0.5,
        )

        if self.calm_steps >= SETTLE_STEPS:
            self.done = True
            self._report(f"settled at step {self.step}")
        elif self.step >= HORIZON_STEPS:
            self.done = True
            self._report(f"observed at step {self.step} (fixed horizon)")

    def _report(self, reason):
        positions = [[float(v) for v in row] for row in self.dofs.position.value]
        if self.dump_path:  # measurement mode: record the settled pose
            Path(self.dump_path).write_text(
                json.dumps({"positions": positions, "speed_trace": self.speed_trace}),
                encoding="utf-8",
            )
        if self.target is None:
            score, detail = 0.0, "no target yet (measurement mode)"
        else:
            rms = rms_to_target(positions, self.target)
            score, detail = score_from_rms(rms), f"rms {rms:.4f}"
        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=f"{reason}; {detail}")
        else:
            print(f"[caduceus] {reason}; {detail} -> score {score:.2f}")
            self.root.animate = False


def _add_static_collision(base, mesh_dir, name, filename, with_triangles):
    node = base.addChild(name)
    node.addObject("MeshOBJLoader", name="loader", filename=str(mesh_dir / filename))
    node.addObject("MeshTopology", src="@loader")
    node.addObject("MechanicalObject", src="@loader", name="CollisModel")
    if with_triangles:
        node.addObject("TriangleCollisionModel", simulated=False, moving=False)
    node.addObject("LineCollisionModel", simulated=False, moving=False)
    node.addObject("PointCollisionModel", simulated=False, moving=False)


def _add_snake(root, mesh_dir, young, mass):
    snake = root.addChild("Snake")
    snake.addObject(
        "SparseGridRamificationTopology", name="grid", n=[4, 12, 3],
        fileTopology=str(mesh_dir / "snake_body.obj"),
        nbVirtualFinerLevels=3, finestConnectivity=False,
    )
    snake.addObject("EulerImplicitSolver", rayleighMass=1.0, rayleighStiffness=0.03)
    snake.addObject(
        "MatrixLinearSystem", template="CompressedRowSparseMatrixMat3x3",
        name="linearSystem",
    )
    snake.addObject(
        "CGLinearSolver", name="linear_solver", iterations=20, tolerance=1e-12,
        threshold=1e-18, template="CompressedRowSparseMatrixMat3x3",
        linearSystem="@linearSystem",
    )
    dofs = snake.addObject(
        "MechanicalObject", name="dofs", position="@grid.position", dy=2.0
    )
    snake.addObject("UniformMass", totalMass=mass)
    snake.addObject(
        "HexahedronFEMForceField", name="FEM", youngModulus=young,
        poissonRatio=0.3, method="large", updateStiffnessMatrix=False,
    )
    snake.addObject(
        "UncoupledConstraintCorrection", defaultCompliance=184,
        useOdeSolverIntegrationFactors=False,
    )

    collis = snake.addChild("Collis")
    collis.addObject(
        "MeshOBJLoader", name="loader", filename=str(mesh_dir / "meca_snake_900tri.obj")
    )
    collis.addObject("MeshTopology", src="@loader")
    collis.addObject("MechanicalObject", src="@loader", name="CollisModel")
    collis.addObject("TriangleCollisionModel", selfCollision=False)
    collis.addObject("LineCollisionModel", selfCollision=False)
    collis.addObject("PointCollisionModel", selfCollision=False)
    collis.addObject("BarycentricMapping", input="@..", output="@.")
    return dofs


def createScene(root):
    trial = open_trial(root)
    young = float(trial.params.get("young_modulus", 30000.0))
    mu = float(trial.params.get("friction_mu", 0.2))
    mass = float(trial.params.get("total_mass", 1.0))
    mesh_dir = find_mesh_dir()

    root.dt = 0.04
    root.gravity = [0.0, -1000.0, 0.0]  # as in caduceus.scn
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showCollisionModels")
    root.addObject(
        "LCPConstraintSolver", tolerance=1e-3, maxIt=1000, initial_guess=False,
        build_lcp=False, mu=mu,
    )
    # Deviation from the source: parallel collision/free-motion disabled so
    # trials are deterministic (provenance bands need reproducible physics).
    root.addObject("FreeMotionAnimationLoop", parallelCollisionDetectionAndFreeMotion=False)
    root.addObject("CollisionPipeline", depth=15, verbose=False, draw=False)
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject("MinProximityIntersection", name="Proximity", alarmDistance=1.5, contactDistance=1.0)
    root.addObject("CollisionResponse", name="Response", response="FrictionContactConstraint")

    dofs = _add_snake(root, mesh_dir, young, mass)

    base = root.addChild("Base")
    _add_static_collision(base, mesh_dir, "Stick", "collision_batons.obj", with_triangles=False)
    _add_static_collision(base, mesh_dir, "Blobs", "collision_boules_V3.obj", with_triangles=True)
    _add_static_collision(base, mesh_dir, "Foot", "collision_pied.obj", with_triangles=True)

    root.addObject(
        Settler(
            trial=trial, root=root, dofs=dofs, target=load_target(),
            dump_path=trial.env.get("OPT_CADU_DUMP", ""), name="Settler",
        )
    )
    return root
