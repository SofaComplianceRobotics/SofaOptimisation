"""SOFA scene: a cube falls toward a floor; score = how soon it lands.

A rigid cube starts at a fixed center height and falls under gravity against a
fixed upward force (so heavier = faster). The generated cube `.obj` is loaded as
the visual; a static floor quad is drawn at y = 0. When the cube's bottom face
crosses ``y = 0`` we score by elapsed time and stop; if it never lands within
the horizon (too light -> it floats), we score 0.

Launched by hand (no optimizer) it still shows a cube + floor and freezes on
landing, using the parameter defaults.

Note: SOFA component/plugin names target v23.06+. Adjust ``_PLUGINS`` if a
RequiredPlugin errors on your build.
"""

import os
import tempfile
from pathlib import Path

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

SPAWN_CENTER_Y = 60.0   # cube center starts here; bigger cubes reach floor sooner
FLOOR_Y = 0.0
FLOOR_HALF = 150.0      # floor visual half-extent
BUOYANCY = 8.0          # fixed upward force; net accel = g - F/m (heavier = faster)
HORIZON_STEPS = 800     # if not landed by here, it floated -> score 0

_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mass",
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Mapping.NonLinear",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.Visual",
    "Sofa.GL.Component.Rendering3D",
]


def _write_cube_obj(path: Path, size: float) -> None:
    """Axis-aligned cube OBJ of edge ``size``, centered at origin."""
    h = size / 2.0
    verts = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1),
    ]
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_floor_obj(path: Path, half: float, y: float) -> None:
    """A flat square in the XZ plane at height ``y``."""
    verts = [(-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half)]
    faces = [(1, 2, 3), (1, 3, 4)]
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Lander(Sofa.Core.Controller):
    """Scores the trial when the cube's bottom reaches the floor."""

    def __init__(self, trial, root, dofs, size, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.dofs = dofs
        self.half = size / 2.0
        self.step = 0
        self.done = False

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        self.step += 1
        center_y = float(self.dofs.position.value[0][1])
        bottom_y = center_y - self.half

        self.trial.write_status(
            {"state": "running", "current_frame": self.step, "total_frames": HORIZON_STEPS},
            min_interval=0.2,
        )

        if bottom_y <= FLOOR_Y:
            self.done = True
            score = 100.0 * (HORIZON_STEPS - self.step) / HORIZON_STEPS
            self._report(score, f"landed at step {self.step}")
        elif self.step >= HORIZON_STEPS:
            self.done = True
            self._report(0.0, "did not land (too light - floated)")

    def _report(self, score, reason):
        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=reason)
        else:
            print(f"[cube_drop] {reason} -> score {score:.2f}")
            self.root.animate = False  # freeze so it visibly rests on the floor


def createScene(root):
    trial = open_trial(root)
    size = float(trial.params.get("cube_size", 10.0))
    mass = float(trial.params.get("cube_mass", 1.0))

    # Cube visual: the optimizer injects OPT_CUBE_MESH; by hand we generate one
    # so the scene is always viewable.
    mesh = trial.env.get("OPT_CUBE_MESH")
    if not mesh:
        mesh = os.path.join(tempfile.gettempdir(), "sofaopt_cube_view.obj")
        _write_cube_obj(Path(mesh), size)

    floor_obj = os.path.join(tempfile.gettempdir(), "sofaopt_floor_view.obj")
    _write_floor_obj(Path(floor_obj), FLOOR_HALF, FLOOR_Y)

    root.dt = 0.01
    root.gravity = [0.0, -9.81, 0.0]
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showVisualModels showBehaviorModels")

    # Static floor.
    floor = root.addChild("floor")
    floor.addObject("MeshOBJLoader", name="floorloader", filename=floor_obj)
    floor.addObject("OglModel", name="floormodel", src="@floorloader", color=[0.55, 0.57, 0.60, 1.0])

    # Falling cube (rigid body).
    cube = root.addChild("cube")
    cube.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    cube.addObject("CGLinearSolver", iterations=25, tolerance=1e-5, threshold=1e-5)
    dofs = cube.addObject(
        "MechanicalObject", name="dofs", template="Rigid3",
        position=[0, SPAWN_CENTER_Y, 0, 0, 0, 0, 1],
    )
    cube.addObject("UniformMass", totalMass=mass)
    cube.addObject("ConstantForceField", totalForce=[0, BUOYANCY, 0, 0, 0, 0])

    visu = cube.addChild("visu")
    visu.addObject("MeshOBJLoader", name="loader", filename=mesh)
    visu.addObject("OglModel", name="model", src="@loader", color=[0.30, 0.60, 0.90, 1.0])
    visu.addObject("RigidMapping")

    cube.addObject(Lander(trial=trial, root=root, dofs=dofs, size=size, name="Lander"))
    return root
