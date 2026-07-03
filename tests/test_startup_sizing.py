"""Auto-sizing of the Sobol' startup phase from searched dimensionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sofaopt.project import ParamSpec, SofaOptProject
from sofaopt.project import TestSpec as _TestSpec


def _proj(n_free: int, sampler: str = "cmaes", explicit=None) -> SofaOptProject:
    tmp = Path(tempfile.mkdtemp())
    params = [ParamSpec(f"p{i}", "int", 0, 5, default=1) for i in range(n_free)]
    params.append(ParamSpec("frozen", "float", 1.0, 1.0, default=1.0))
    return SofaOptProject(
        name="sizing",
        work_dir=tmp,
        params=params,
        tests=[_TestSpec("t", scene_file=tmp / "scene.py", max_score=100.0,
                         default_selected=True)],
        sampler=sampler,
        cmaes_startup_trials=explicit,
    )


def test_auto_sizes_scale_with_searched_dims():
    # power of two nearest (log2) to 5*d for cmaes; frozen params excluded
    assert _proj(6).resolve_startup_trials() == 32
    assert _proj(9).resolve_startup_trials() == 32
    assert _proj(12).resolve_startup_trials() == 64


def test_gp_uses_10d_rule():
    assert _proj(6, sampler="gp").resolve_startup_trials() == 64


def test_explicit_value_passes_through():
    assert _proj(6, explicit=24).resolve_startup_trials() == 24


def test_floor_is_eight():
    assert _proj(1).resolve_startup_trials() == 8
