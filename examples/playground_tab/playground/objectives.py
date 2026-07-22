"""Synthetic N-D landscapes for the optimizer playground.

The search space is the unit hypercube [0, 1]^dim; the objective is a
max-of-Gaussians surface so each "bump" is a clean optimum whose apex height
equals its configured score. Higher score is better (maximization). The heatmap
is only meaningful for dim == 2; everything else works in any dimension.
"""

from __future__ import annotations

import numpy as np

GRID_RESOLUTION = 90  # heatmap sampling density per axis (2D only)


def make_landscape(
    mode: str,
    n_optima: int,
    height_spread: float,
    seed: int,
    dim: int = 2,
) -> list[dict]:
    """Build a landscape as a list of Gaussian peaks in `dim` dimensions.

    Args:
        mode: "default" for a single centred global optimum, "ridge" for a narrow
            diagonal ridge rewarding all coordinates raised together (favours
            CMA-ES), anything else for randomly placed multi-optimum landscapes.
        n_optima: Total number of optima (global + local). Ignored unless random.
        height_spread: 0..1, how far local optima sit below the global.
        seed: RNG seed for reproducibility.
        dim: Number of search dimensions.

    Returns:
        List of peaks, each a dict {"center": [..dim..], "height": h, "width": w}.
        The first peak is always the global optimum.
    """
    rng = np.random.default_rng(seed)

    # A chain of narrow Gaussians up the all-equal diagonal: scoring well needs
    # every coordinate raised together. The narrowness is across the diagonal.
    if mode == "ridge":
        return [{"center": [0.45 + 0.10 * i] * dim, "height": 0.70 + 0.075 * i,
                 "width": 0.07} for i in range(5)]

    if mode == "default" or n_optima <= 1:
        return [{"center": [0.5] * dim, "height": 1.0, "width": 0.12}]

    peaks: list[dict] = []
    gc = rng.uniform(0.18, 0.82, dim)
    peaks.append({"center": gc.tolist(), "height": 1.0,
                  "width": float(rng.uniform(0.08, 0.13))})

    for _ in range(n_optima - 1):
        c = rng.uniform(0.1, 0.9, dim)
        for _attempt in range(200):
            c = rng.uniform(0.1, 0.9, dim)
            if all(np.sum((c - np.asarray(p["center"])) ** 2) > 0.025 for p in peaks):
                break
        height = 1.0 - height_spread * rng.uniform(0.2, 0.9)
        peaks.append({"center": c.tolist(), "height": float(height),
                      "width": float(rng.uniform(0.06, 0.11))})

    return peaks


def score_at(peaks: list[dict], point) -> float:
    """Evaluate the landscape at an N-D point (the optimizer objective)."""
    p = np.asarray(point, dtype=float)
    best = 0.0
    for pk in peaks:
        c = np.asarray(pk["center"], dtype=float)
        v = pk["height"] * np.exp(-np.sum((p - c) ** 2) / (2.0 * pk["width"] ** 2))
        if v > best:
            best = v
    return float(best)


def score_line(peaks: list[dict], resolution: int = 300):
    """Evaluate the landscape along x for the 1D line plot (dim == 1 only)."""
    xs = np.linspace(0.0, 1.0, resolution)
    ys = [score_at(peaks, [x]) for x in xs]
    return xs.tolist(), ys


def axis_profiles(peaks: list[dict], dim: int, resolution: int = GRID_RESOLUTION):
    """Per-axis marginal-max score profiles for the strip view (any dimension).

    For each axis i, sweep its coordinate over [0, 1] and report the best score
    reachable when every *other* coordinate is set optimally. Because the surface
    is a max-of-Gaussians, that best case is, for each peak, sitting on the peak's
    centre in the free dimensions — leaving only the i-th coordinate's penalty:

        profile_i(x) = max over peaks of height * exp(-(x - center_i)^2 / 2w^2)

    This is an optimistic projection (it assumes the other axes are perfect), so
    it must be read alongside the run's true achieved score.

    Returns (xs, rows) where rows[i][k] is the best score reachable on axis i at
    coordinate xs[k].
    """
    xs = np.linspace(0.0, 1.0, resolution)
    rows = []
    for i in range(dim):
        row = np.zeros(resolution)
        for pk in peaks:
            ci = pk["center"][i]
            row = np.maximum(row, pk["height"] * np.exp(
                -((xs - ci) ** 2) / (2.0 * pk["width"] ** 2)))
        rows.append(row.tolist())
    return xs.tolist(), rows


def score_grid(peaks: list[dict], resolution: int = GRID_RESOLUTION):
    """Evaluate the landscape on a 2D grid for heatmap display (dim == 2 only).

    Returns (xs, ys, z) where z[j][i] is the score at (xs[i], ys[j]).
    """
    xs = np.linspace(0.0, 1.0, resolution)
    ys = np.linspace(0.0, 1.0, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = np.zeros_like(grid_x)
    for pk in peaks:
        cx, cy = pk["center"][0], pk["center"][1]
        z = np.maximum(z, pk["height"] * np.exp(
            -((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2.0 * pk["width"] ** 2)))
    return xs.tolist(), ys.tolist(), z.tolist()
