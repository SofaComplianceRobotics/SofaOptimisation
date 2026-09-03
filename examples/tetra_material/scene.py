"""SOFA scene: single-tetrahedron FEM — inverse material identification.

Adapted from SOFA's shipped ``examples/Demos/oneTetrahedron.scn`` (same
geometry, mass, FEM force field and solver components), reworked for
optimization: the **base triangle is fixed** and the apex sags under gravity by
an amount set by the material — so the settled apex position identifies
(youngModulus, poissonRatio, mass). The score is how closely the settled apex
matches a target position measured once from reference material values.

This is the *light* real-SOFA validation tier (one tetra, 4 nodes — near-instant
trials): it exercises the genuine runSofa/python-runner path, settle detection,
and — via the noise knob — racing, without the cost of a full mesh.

Identifiability note: in linear elasticity the static sag depends on the load /
stiffness ratio, so (youngModulus, mass) form an approximate ridge — many pairs
reach score ~100. That is intentional (it is the classic identifiability
structure of material identification; CMA-ES's covariance learns the ridge).

Config from the trial params / env:
  - params ``young_modulus``, ``poisson_ratio``, ``total_mass``
  - env ``OPT_TETRA_NOISE``  per-run Gaussian score noise sigma (default 0)
"""

import math
import random

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

# Geometry from oneTetrahedron.scn: apex on top, equilateral base at y=0.
POSITIONS = [
    [0.0, 10.0, 0.0],      # 0: apex (free — its settled position is the signal)
    [10.0, 0.0, 0.0],      # 1..3: base triangle (fixed)
    [-5.0, 0.0, 8.66],
    [-5.0, 0.0, -8.66],
]
FIXED_INDICES = [1, 2, 3]
APEX = 0

HORIZON_STEPS = 600        # hard stop; most materials self-stop well before
#                            (reference settles ~step 340); only the softest+
#                            heaviest corner scores its (deterministic)
#                            horizon-end position instead.
SETTLE_SPEED = 1e-3        # |apex velocity| below this ...
SETTLE_STEPS = 10          # ... for this many consecutive steps = settled

# Target apex settle position + score scale. MEASURED 2026-07-13 (Win 11 dev
# machine, python runner, reference material E=4, nu=0.30, mass=5.0, settled at
# step 339) — command in the README. Re-measure after any INTENTIONAL physics
# change; never tweak to silence a failure.
TARGET_APEX = [0.0, 9.457, 0.0]
SCORE_SCALE = 0.5          # score = 100 * exp(-|apex - target| / SCORE_SCALE);
#                            apex spans ~[8.7, 9.98] over the param box (measured),
#                            so the worst error (~0.8) scores ~20 and err 0.02
#                            scores ~96; the search-start defaults score ~41.

_PLUGINS = [
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mass",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Visual",
]


def score_from_apex(apex_xyz) -> float:
    """Map the settled apex position to the 0-100 objective (100 = on target)."""
    err = math.dist([float(v) for v in apex_xyz], TARGET_APEX)
    return 100.0 * math.exp(-err / SCORE_SCALE)


class Settler(Sofa.Core.Controller):
    """Waits for the apex to settle, then scores the trial and stops."""

    def __init__(self, trial, root, dofs, noise_sigma, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.dofs = dofs
        self.noise_sigma = noise_sigma
        self.rng = random.Random(int(trial.env.get("OPT_RUN_SLOT", "1")))
        self.step = 0
        self.calm_steps = 0
        self.done = False

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        self.step += 1
        apex = self.dofs.position.value[APEX]
        vel = self.dofs.velocity.value[APEX]
        speed = math.sqrt(sum(float(v) ** 2 for v in vel))
        self.calm_steps = self.calm_steps + 1 if speed < SETTLE_SPEED else 0

        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS},
            min_interval=0.2,
        )

        if self.calm_steps >= SETTLE_STEPS:
            self.done = True
            self._report(apex, f"settled at step {self.step}")
        elif self.step >= HORIZON_STEPS:
            # Still counts: score the horizon-end position (nearly settled —
            # implicit damping; only extreme params get here).
            self.done = True
            self._report(apex, f"horizon at step {self.step} (slow settle)")

    def _report(self, apex, reason):
        score = score_from_apex(apex)
        if self.noise_sigma > 0.0:
            score = max(0.0, min(100.0, score + self.rng.gauss(0.0, self.noise_sigma)))
        apex_str = "[" + ", ".join(f"{float(v):.3f}" for v in apex) + "]"
        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=f"{reason}; apex {apex_str}")
        else:
            print(f"[tetra] {reason}; apex {apex_str} -> score {score:.2f}")
            self.root.animate = False


def createScene(root):
    trial = open_trial(root)
    young = float(trial.params.get("young_modulus", 10.0))
    poisson = float(trial.params.get("poisson_ratio", 0.30))
    mass = float(trial.params.get("total_mass", 2.0))
    noise_sigma = float(trial.env.get("OPT_TETRA_NOISE", "0") or 0.0)

    root.dt = 0.01
    root.gravity = [0.0, -10.0, 0.0]  # as in oneTetrahedron.scn
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields")

    tetra = root.addChild("tetra")
    # Heavier Rayleigh damping than the source .scn: damping vanishes at zero
    # velocity so the STATIC equilibrium (the signal) is unchanged — it only
    # makes soft/heavy materials stop oscillating within the horizon (measured:
    # E=2,m=6 settles at ~step 200 instead of ringing past 400).
    tetra.addObject("EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=1.0)
    tetra.addObject(
        "CGLinearSolver", iterations=25, tolerance=1e-9, threshold=1e-9
    )
    tetra.addObject("MeshTopology", name="topo", tetrahedra=[[0, 1, 2, 3]])
    dofs = tetra.addObject(
        "MechanicalObject", name="dofs", template="Vec3", position=POSITIONS,
        showObject=True, showObjectScale=8.0,
    )
    tetra.addObject("UniformMass", totalMass=mass)
    tetra.addObject(
        "TetrahedronFEMForceField", name="FEM", method="large",
        youngModulus=young, poissonRatio=poisson, updateStiffnessMatrix=True,
    )
    tetra.addObject("FixedProjectiveConstraint", indices=FIXED_INDICES)
    tetra.addObject(
        Settler(trial=trial, root=root, dofs=dofs, noise_sigma=noise_sigma,
                name="Settler")
    )
    return root
