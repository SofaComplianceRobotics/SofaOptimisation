"""Smoke test: the dashboard app assembles (tabs + callbacks register) with the
merged Run tab, the sampler/prune controls, and the Parameters tab (which now
carries the importance/interaction section). No browser / SOFA needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# Alias TestSpec so pytest does not try to collect it as a test class.
from sofaopt.project import ParamSpec, SofaOptProject
from sofaopt.project import TestSpec as _TestSpec


def _minimal_project() -> SofaOptProject:
    tmp = Path(tempfile.mkdtemp())
    return SofaOptProject(
        name="dash_smoke",
        work_dir=tmp,
        params=[
            ParamSpec("a", "float", 0.0, 1.0, default=0.5),
            ParamSpec("b", "int", 0, 5, default=2),
        ],
        tests=[_TestSpec("t", scene_file=tmp / "scene.py", max_score=100.0, default_selected=True)],
        runner="python",
        sampler="gp",
        seed_sampler="sobol",
        cmaes_with_margin=True,
        n_parallel=4,
    )


def test_create_app_builds():
    from sofaopt.dashboard.app import create_app

    app = create_app(_minimal_project())
    assert app is not None
    layout_str = str(app.layout)
    for needle in (
        # merged tab structure (Scenes+Optimise -> Run; Bounds+Interactions -> Parameters)
        "'Run'", "'Monitor'", "'Results'", "'Parameters'",
        "Importance & Interactions",
        # Run-tab controls (optimizer + new prune toggle + per-row preview + shared log filter)
        "opt-sampler", "opt-cmaes-margin", "opt-seed-sampler",
        "opt-run-until-converged", "opt-restart-patience", "opt-prune-mode",
        "opt-cmaes-restarts", "opt-stall-generations", "opt-inc-popsize",
        "opt-restart-flags",
        "scene-preview", "run-log-filter",
        # Parameters tab table
        "param-table",
    ):
        assert needle in layout_str, f"missing dashboard element: {needle}"


def test_interaction_figures_placeholder_without_db():
    from sofaopt.dashboard import context
    from sofaopt.dashboard.plotting.interactions import (
        build_importance_bar,
        build_interaction_heatmap,
    )

    context.set_project(_minimal_project())  # no study.db under fresh work_dir
    # Should degrade to a placeholder figure, not raise.
    assert build_importance_bar() is not None
    assert build_interaction_heatmap() is not None


if __name__ == "__main__":
    test_create_app_builds()
    test_interaction_figures_placeholder_without_db()
    print("Dashboard build smoke test passed.")
