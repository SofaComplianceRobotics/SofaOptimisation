"""Classic black-box optimization test functions — the analytic landscape harness.

Each function has a KNOWN global optimum, which is what makes this example useful:
we can *measure* whether the optimizer features actually help (IPOP restarts vs
plain CMA-ES on a multimodal function, racing under injected noise, converged
self-sizing) rather than only checking that the plumbing ran.

sofaopt MAXIMIZES a 0–100 score, so each function ``f`` (to be minimized, global
minimum ``f_opt``) is mapped to::

    score = 100 * exp(-(f(x) - f_opt) / (scale_per_dim * dim))

so the global optimum scores 100 and the landscape stays smooth (no dead zero
region) and dimension-consistent. This module is pure and stdlib-only — the SOFA
``scene.py`` and the SOFA-free feature-advantage bench both score through it, so
the score definition has exactly one implementation.

Functions and their character:
- ``sphere``      — smooth, unimodal. The CONTROL: restarts must not help here.
- ``rosenbrock``  — a curved narrow valley. CMA-ES covariance showcase.
- ``rastrigin``   — many regular local minima on a bowl. IPOP restarts win.
- ``ackley``      — flat outer funnel + fine local minima. Escaping the rim matters.
- ``schwefel``    — deceptive: the global optimum sits far from the centre/second-best.
- ``himmelblau``  — 2-D, FOUR equal global optima. Clean multi-basin restart target.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable


def sphere(x: list[float]) -> float:
    return sum(xi * xi for xi in x)


def rosenbrock(x: list[float]) -> float:
    return sum(
        100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def rastrigin(x: list[float]) -> float:
    a = 10.0
    return a * len(x) + sum(xi * xi - a * math.cos(2.0 * math.pi * xi) for xi in x)


def ackley(x: list[float]) -> float:
    n = len(x)
    s1 = sum(xi * xi for xi in x)
    s2 = sum(math.cos(2.0 * math.pi * xi) for xi in x)
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(s1 / n))
        - math.exp(s2 / n)
        + 20.0
        + math.e
    )


def schwefel(x: list[float]) -> float:
    return 418.9828872724338 * len(x) - sum(
        xi * math.sin(math.sqrt(abs(xi))) for xi in x
    )


def himmelblau(x: list[float]) -> float:
    if len(x) != 2:
        raise ValueError("himmelblau is defined in 2-D only.")
    a, b = x
    return (a * a + b - 11.0) ** 2 + (a + b * b - 7.0) ** 2


@dataclass(frozen=True)
class Benchmark:
    """One test function plus everything needed to search and score it."""

    fn: Callable[[list[float]], float]
    low: float
    high: float
    optimum_x: list[float]          # one global optimizer (himmelblau has four)
    f_opt: float                    # f at the global optimum
    default: float                  # per-param start point (deliberately off-optimum)
    scale_per_dim: float            # score-mapping scale (see module docstring)
    fixed_dim: int | None = None    # None = any dimension; 2 for himmelblau

    def score(self, x: list[float]) -> float:
        """Map f(x) to the 0–100 study objective (100 at the global optimum)."""
        excess = self.fn(x) - self.f_opt
        s = 100.0 * math.exp(-excess / (self.scale_per_dim * len(x)))
        return max(0.0, min(100.0, s))


# name -> Benchmark. Domains and optima are the textbook values.
BENCHMARKS: dict[str, Benchmark] = {
    "sphere": Benchmark(sphere, -5.12, 5.12, [0.0], 0.0, default=3.0, scale_per_dim=8.0),
    "rosenbrock": Benchmark(
        rosenbrock, -2.048, 2.048, [1.0], 0.0, default=-1.5, scale_per_dim=40.0
    ),
    "rastrigin": Benchmark(
        rastrigin, -5.12, 5.12, [0.0], 0.0, default=4.5, scale_per_dim=20.0
    ),
    "ackley": Benchmark(ackley, -32.768, 32.768, [0.0], 0.0, default=20.0, scale_per_dim=6.0),
    "schwefel": Benchmark(
        schwefel, -500.0, 500.0, [420.9687], 0.0, default=-300.0, scale_per_dim=250.0
    ),
    "himmelblau": Benchmark(
        himmelblau, -5.0, 5.0, [3.0, 2.0], 0.0, default=0.0, scale_per_dim=20.0,
        fixed_dim=2,
    ),
}

# The four global optima of himmelblau (all f = 0) — used by the restart bench.
HIMMELBLAU_OPTIMA = [
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]


def get_benchmark(name: str) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise KeyError(
            f"Unknown benchmark '{name}'. Have: {sorted(BENCHMARKS)}"
        ) from None


def scored(name: str, x: list[float], noise_sigma: float = 0.0,
           rng: random.Random | None = None) -> float:
    """The 0–100 score for ``x`` on function ``name``, with optional per-call
    Gaussian noise (the racing knob). ``rng`` seeded per run-slot upstream makes
    the repeats i.i.d. samples of one noisy objective, which is what mean
    aggregation + racing assume.
    """
    bench = get_benchmark(name)
    score = bench.score(x)
    if noise_sigma > 0.0:
        r = rng or random
        score += r.gauss(0.0, noise_sigma)
    return max(0.0, min(100.0, score))
