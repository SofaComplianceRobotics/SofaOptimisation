"""Correctness of the analytic benchmark functions (examples/landscape).

Pure functions with known optima — no SOFA, no optimizer. If these are wrong,
every feature-advantage measurement built on them is meaningless, so they are
pinned to the textbook optima and score mapping.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples" / "landscape" / "benchmark_functions.py"
)
_spec = importlib.util.spec_from_file_location("benchmark_functions", _MOD_PATH)
bf = importlib.util.module_from_spec(_spec)
# Register before exec so the @dataclass field-type resolution can find the
# module by name (dataclasses looks it up in sys.modules).
sys.modules["benchmark_functions"] = bf
_spec.loader.exec_module(bf)


@pytest.mark.parametrize("name", list(bf.BENCHMARKS))
def test_global_optimum_scores_100(name):
    """f at the textbook optimum equals f_opt, and it maps to score 100."""
    bench = bf.get_benchmark(name)
    dim = bench.fixed_dim or 4
    x = (bench.optimum_x * dim)[:dim] if len(bench.optimum_x) == 1 else bench.optimum_x
    assert bench.fn(x) == pytest.approx(bench.f_opt, abs=1e-4)
    assert bench.score(x) == pytest.approx(100.0, abs=1e-3)


def test_score_is_bounded_and_decreases_away_from_optimum():
    bench = bf.get_benchmark("sphere")
    assert bench.score([0.0, 0.0]) == pytest.approx(100.0)
    near = bench.score([0.5, 0.5])
    far = bench.score([4.0, 4.0])
    assert 0.0 <= far < near < 100.0  # monotone down-slope on the bowl


def test_himmelblau_has_four_equal_global_optima():
    bench = bf.get_benchmark("himmelblau")
    for a, b in bf.HIMMELBLAU_OPTIMA:
        assert bench.fn([a, b]) == pytest.approx(0.0, abs=1e-4)
        assert bench.score([a, b]) == pytest.approx(100.0, abs=1e-3)


def test_rastrigin_is_multimodal_local_minima_score_below_global():
    bench = bf.get_benchmark("rastrigin")
    # Integer lattice points are local minima of rastrigin (cos term = 1).
    glob = bench.score([0.0, 0.0])
    local = bench.score([1.0, 0.0])  # a neighbouring local min, f = 1
    assert glob == pytest.approx(100.0)
    assert 0.0 < local < glob


def test_schwefel_optimum_far_from_centre():
    bench = bf.get_benchmark("schwefel")
    # The deceptive part: the centre scores far worse than the far-flung optimum.
    assert bench.score([420.9687, 420.9687]) == pytest.approx(100.0, abs=1e-2)
    assert bench.score([0.0, 0.0]) < 60.0


def test_himmelblau_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="2-D"):
        bf.himmelblau([1.0, 2.0, 3.0])


def test_scored_noise_perturbs_but_stays_bounded():
    clean = bf.scored("sphere", [0.0, 0.0], noise_sigma=0.0)
    assert clean == pytest.approx(100.0)
    rng = random.Random(0)
    noisy = [bf.scored("sphere", [1.0, 1.0], noise_sigma=5.0, rng=rng) for _ in range(50)]
    assert all(0.0 <= s <= 100.0 for s in noisy)     # clamped
    assert len(set(noisy)) > 1                        # actually stochastic


def test_unknown_benchmark_raises():
    with pytest.raises(KeyError, match="Unknown benchmark"):
        bf.get_benchmark("nope")
