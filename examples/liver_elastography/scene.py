"""SOFA scene: liver FEM with a REGIONAL stiffness field — elastography.

Extends ``liver_registration`` from a scalar material to a **heterogeneous
stiffness field**: the 596 tetrahedra are partitioned into 6 committed regions
(``regions.json``) and each region has its own Young's modulus, passed to
``TetrahedronFEMForceField`` as a per-element list. The reference field is
homogeneous E=3000 with one SOFT **lesion** (region 5, E=1000 — measured to
give ~2x the shape signal of a stiff inclusion); the optimizer must localize
it from the settled shapes under THREE patient orientations (gravity
directions) — inverse elastography in miniature.

This is the framework's in-depth study platform: 8 parameters with graded
sensitivity (region 0 hugs the fixed nodes and barely moves; region 5 is deep
in the free bulk), 3 weighted tests whose number genuinely changes what is
identifiable, ~1 s launches so thousand-trial studies stay cheap.

Deviations from ``liver_registration`` (documented on purpose):
``TetrahedronFEMForceField`` (same "large" corotational formulation) instead of
``TetrahedralCorotationalFEMForceField``, because only the former accepts a
per-element ``youngModulus`` list; a third load case is added.

Config from the trial params / env:
  - params ``young_region_0`` .. ``young_region_5``, ``poisson_ratio``,
    ``mass_density``
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
N_REGIONS = 6
REGIONS_FILE = HERE / "regions.json"

# Load cases: patient orientation = gravity direction (|g| = 9.81 in all).
LOAD_CASES = {
    "supine": [0.0, -9.81, 0.0],
    "lateral": [-6.937, -6.937, 0.0],   # 45-degree roll toward -x
    "tilt": [0.0, -6.937, 6.937],       # 45-degree pitch toward +z
}
DEFAULT_CASE = "supine"

FIXED_INDICES = [3, 39, 64]  # ligament attachment points, as in liver.scn

HORIZON_STEPS = 500        # hard stop at dt=0.02 (10 s sim); measured settles
#                            land well before (provenance in targets.json).
SETTLE_SPEED = 2e-3        # max |node velocity| below this ...
SETTLE_STEPS = 10          # ... for this many consecutive steps = settled

# Target settled shapes, one per load case. MEASURED from the reference field
# (E=3000 everywhere except soft lesion region 5 at E=1000, poisson_ratio=0.30,
# mass_density=1.0); provenance in targets.json / README. Re-measure after any
# INTENTIONAL physics change; never tweak to silence a failure.
TARGETS_FILE = HERE / "targets.json"
SCORE_SCALE = 0.15         # score = 100 * exp(-rms / SCORE_SCALE); rms is the
#                            root-mean-square node distance to the target shape
#                            over the free nodes (calibrated: README anchors).

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


def region_of_tet() -> list:
    """The committed per-tet region assignment (see regions.json provenance)."""
    return json.loads(REGIONS_FILE.read_text(encoding="utf-8"))["region_of_tet"]


def stiffness_field(region_young: list) -> list:
    """Per-element Young's modulus list from the 6 per-region values."""
    return [float(region_young[r]) for r in region_of_tet()]


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
            print(f"[elasto:{self.case}] {reason}; {detail} -> score {score:.2f}")
            self.root.animate = False


def createScene(root):
    trial = open_trial(root)
    region_young = [
        float(trial.params.get(f"young_region_{k}", 3000.0)) for k in range(N_REGIONS)
    ]
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
        "TetrahedronFEMForceField",  # accepts per-element youngModulus (README)
        template="Vec3", name="FEM", method="large",
        youngModulus=stiffness_field(region_young), poissonRatio=poisson,
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
