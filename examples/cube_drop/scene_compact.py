"""Second SOFA scene for the multi-objective demo: reward a small cube.

Scores purely on cube_size — smaller = more compact = higher score.  This
creates a genuine Pareto trade-off with scene.py (which wants a *big* cube to
land fast): no single parameter setting maximises both objectives, so NSGA-II
will find the full trade-off frontier.

The scene does one simulation step then writes its score and exits.
"""

import Sofa
import Sofa.Core

from sofaopt.scene import open_trial

MAX_SIZE = 50.0   # cube_size upper bound from project.py
MIN_SIZE = 5.0    # cube_size lower bound from project.py

_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.StateContainer",
]


class CompactScorer(Sofa.Core.Controller):
    """Writes score = 100 * (1 - normalised_size) on the first step."""

    def __init__(self, trial, root, size, **kw):
        Sofa.Core.Controller.__init__(self, **kw)
        self.trial = trial
        self.root = root
        self.size = size
        self.done = False

    def onAnimateEndEvent(self, _event):
        if self.done:
            return
        self.done = True
        score = max(0.0, 100.0 * (MAX_SIZE - self.size) / (MAX_SIZE - MIN_SIZE))
        if self.trial.is_optimizing:
            self.trial.write_score(score, reason=f"size={self.size:.2f}")
        else:
            print(f"[compact] cube_size={self.size:.2f} -> compact_score={score:.2f}/100")
            self.root.animate = False


def createScene(root):
    trial = open_trial(root)
    size = float(trial.params.get("cube_size", 10.0))

    root.dt = 0.01
    root.gravity = [0.0, 0.0, 0.0]
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", name=plugin)
    root.addObject("DefaultAnimationLoop")

    node = root.addChild("scorer")
    node.addObject("MechanicalObject", name="dofs", template="Vec3d")
    node.addObject(CompactScorer(trial=trial, root=root, size=size, name="CompactScorer"))
    return root
