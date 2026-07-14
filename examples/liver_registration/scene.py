"""SOFA scene: liver FEM — multi-load-case material identification (registration).

Adapted from SOFA's shipped ``examples/Demos/liver.scn`` (same mesh, FEM force
field, mass, solver and fixed nodes), reworked for optimization: the liver hangs
from its three fixed nodes and sags under gravity; the settled shape of all 181
mesh nodes is the observation. The optimizer must recover the material
(youngModulus, poissonRatio, massDensity) that reproduces **target settled
shapes measured once from reference values** — surgical-registration-style
inverse identification on a real tetrahedral organ mesh.

The same scene serves TWO load cases, selected by the sofaopt test name
(``trial.test_name``): the patient orientation changes the gravity vector.
Each case is its own ``TestSpec``; this example is the framework's reference
for the **multi-test weighted scoring pipeline** (two tests per trial).

Dropped from the source ``.scn`` (documented on purpose): the collision
pipeline, sphere collision models and the OBJ visual mapping — the liver
interacts with nothing here, and the observation is the mechanical dofs.

Config from the trial params / env:
  - params ``young_modulus``, ``poisson_ratio``, ``mass_density``
  - env ``OPT_LIVER_CASE``   load-case override for standalone/GUI runs
  - env ``OPT_LIVER_DUMP``   measurement mode: path to write the settled
                             positions JSON (used once to create targets.json)
"""

import json
import math
import os
from pathlib import Path

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

HERE = Path(__file__).resolve().parent

# Load cases: patient orientation = gravity direction (|g| = 9.81 in both).
LOAD_CASES = {
    "supine": [0.0, -9.81, 0.0],
    "lateral": [-6.937, -6.937, 0.0],  # 45-degree roll toward -x
}
DEFAULT_CASE = "supine"

FIXED_INDICES = [3, 39, 64]  # ligament attachment points, as in liver.scn

HORIZON_STEPS = 500        # hard stop at dt=0.02 (10 s sim); measured settles
#                            land well before (reference: supine step 268,
#                            lateral step 296); only the softest+heaviest
#                            corner (E=500, rho=3) scores its (deterministic)
#                            horizon-end shape instead.
SETTLE_SPEED = 2e-3        # max |node velocity| below this ...
SETTLE_STEPS = 10          # ... for this many consecutive steps = settled

# Target settled shapes, one per load case. MEASURED 2026-07-14 from the
# reference material (young_modulus=3000, poisson_ratio=0.30, mass_density=1.0
# — the values shipped in liver.scn); provenance in targets.json / README.
# Re-measure after any INTENTIONAL physics change; never tweak to silence a
# failure.
TARGETS_FILE = HERE / "targets.json"
SCORE_SCALE = 0.15         # score = 100 * exp(-rms / SCORE_SCALE); rms is the
#                            root-mean-square node distance to the target shape
#                            over the free nodes (mesh spans ~5 units). Measured
#                            spread (supine): search defaults rms 0.146 -> 38,
#                            stiff+light corner rms 0.207 -> 25, soft+heavy
#                            corner rms 6.5 -> ~0, nu-only error rms 0.018 -> 89.

_PLUGINS = [
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.Mass",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Dynamic",
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
        if (cand / "liver.msh").is_file():
            return cand
    raise FileNotFoundError(
        "liver.msh not found; set OPT_SOFA_MESH_DIR to SOFA's share/mesh directory "
        f"(tried: {[str(c) for c in candidates]})"
    )


def load_target(case: str) -> list | None:
    """Settled target positions for the case, or None before measurement."""
    if not TARGETS_FILE.is_file():
        return None
    data = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    return data.get("cases", {}).get(case)


def rms_to_target(positions, target, skip=frozenset(FIXED_INDICES)) -> float:
    """RMS node distance to the target shape over the free (non-fixed) nodes."""
    total, count = 0.0, 0
    for i, (pos, tgt) in enumerate(zip(positions, target, strict=True)):
        if i in skip:
            continue
        total += sum((float(a) - float(b)) ** 2 for a, b in zip(pos, tgt, strict=True))
        count += 1
    return math.sqrt(total / count)


def score_from_rms(rms: float) -> float:
    return 100.0 * math.exp(-rms / SCORE_SCALE)


class Settler(Sofa.Core.Controller):
    """Waits for every node to settle, then scores the shape and stops."""

    def __init__(self, trial, root, dofs, case, target, dump_path, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.dofs = dofs
        self.case = case
        self.target = target
        self.dump_path = dump_path
        self.step = 0
        self.calm_steps = 0
        self.done = False

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        self.step += 1
        vel = self.dofs.velocity.value
        max_speed = max(math.sqrt(sum(float(v) ** 2 for v in row)) for row in vel)
        self.calm_steps = self.calm_steps + 1 if max_speed < SETTLE_SPEED else 0

        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS},
            min_interval=0.2,
        )

        if self.calm_steps >= SETTLE_STEPS:
            self.done = True
            self._report(f"settled at step {self.step}")
        elif self.step >= HORIZON_STEPS:
            # Still counts: score the horizon-end shape (nearly settled —
            # implicit damping; only extreme params get here).
            self.done = True
            self._report(f"horizon at step {self.step} (slow settle)")

    def _report(self, reason):
        positions = [[float(v) for v in row] for row in self.dofs.position.value]
        if self.dump_path:  # measurement mode: record the settled shape
            Path(self.dump_path).write_text(
                json.dumps({"case": self.case, "positions": positions}),
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
            print(f"[liver:{self.case}] {reason}; {detail} -> score {score:.2f}")
            self.root.animate = False


def createScene(root):
    trial = open_trial(root)
    young = float(trial.params.get("young_modulus", 3000.0))
    poisson = float(trial.params.get("poisson_ratio", 0.30))
    density = float(trial.params.get("mass_density", 1.0))
    case = trial.test_name or os.environ.get("OPT_LIVER_CASE", DEFAULT_CASE)
    if case not in LOAD_CASES:
        raise ValueError(f"Unknown load case '{case}'. Have: {sorted(LOAD_CASES)}")
    mesh = find_mesh_dir() / "liver.msh"

    root.dt = 0.02
    root.gravity = LOAD_CASES[case]
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields")

    liver = root.addChild("liver")
    liver.addObject("EulerImplicitSolver", rayleighStiffness=0.1, rayleighMass=0.1)
    liver.addObject("CGLinearSolver", iterations=25, tolerance=1e-9, threshold=1e-9)
    liver.addObject("MeshGmshLoader", name="loader", filename=str(mesh))
    liver.addObject("TetrahedronSetTopologyContainer", name="topo", src="@loader")
    dofs = liver.addObject("MechanicalObject", name="dofs", src="@loader")
    liver.addObject("TetrahedronSetGeometryAlgorithms", template="Vec3")
    liver.addObject("DiagonalMass", massDensity=density)
    liver.addObject(
        "TetrahedralCorotationalFEMForceField",
        template="Vec3", name="FEM", method="large",
        youngModulus=young, poissonRatio=poisson, computeGlobalMatrix=False,
    )
    liver.addObject("FixedProjectiveConstraint", indices=FIXED_INDICES)
    liver.addObject(
        Settler(
            trial=trial, root=root, dofs=dofs, case=case,
            target=load_target(case),
            dump_path=trial.env.get("OPT_LIVER_DUMP", ""),
            name="Settler",
        )
    )
    return root
