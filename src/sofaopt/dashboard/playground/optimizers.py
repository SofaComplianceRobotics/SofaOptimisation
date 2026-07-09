"""Run an optimization over an N-D landscape with a chosen algorithm.

Random / Grid / Bayesian(TPE) go through Optuna so students see the same engine
the real gripper lab uses. CMA-ES (via the `cmaes` library) and REINFORCE keep
an explicit search Gaussian. Each run records its full trajectory; the first two
coordinates of every evaluation are kept for the 2D heatmap, and the per-
generation Gaussian (2D only) for the ellipse overlay.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

GRID_MAX_TRIALS = 2000  # cap so a high-D grid stays runnable / animatable

# Display labels and which hyperparameters each algorithm exposes.
ALGO_LABELS: dict[str, str] = {
    "random": "Random search",
    "grid": "Grid search",
    "bayesian": "Bayesian (TPE)",
    "reinforce": "REINFORCE (policy grad)",
    "cmaes": "CMA-ES",
}

ALGO_PARAMS: dict[str, tuple[str, ...]] = {
    "random": (),
    "grid": ("resolution",),
    "bayesian": ("n_startup_trials",),
    "reinforce": ("sigma0", "popsize", "learning_rate"),
    "cmaes": ("sigma0", "popsize", "n_startup_trials"),
}

ALGO_ORDER: tuple[str, ...] = ("random", "grid", "bayesian", "reinforce", "cmaes")


def _build_sampler(algo: str, hp: dict, seed: int, dim: int) -> optuna.samplers.BaseSampler:
    """Construct the Optuna sampler for a non-CMA-ES/REINFORCE algorithm."""
    if algo == "random":
        return optuna.samplers.RandomSampler(seed=seed)

    if algo == "grid":
        resolution = max(2, int(hp.get("resolution", 8)))
        axis = np.linspace(0.0, 1.0, resolution).tolist()
        search_space = {f"x{i}": axis for i in range(dim)}
        return optuna.samplers.GridSampler(search_space, seed=seed)

    if algo == "bayesian":
        return optuna.samplers.TPESampler(
            n_startup_trials=max(1, int(hp.get("n_startup_trials", 5))),
            seed=seed,
        )

    raise ValueError(f"Unknown Optuna algorithm: {algo}")


def effective_budget(algo: str, hp: dict, budget: int, dim: int) -> int:
    """Resolve the evaluation count (grid is res^dim, capped to stay runnable)."""
    if algo == "grid":
        resolution = max(2, int(hp.get("resolution", 8)))
        return min(resolution ** dim, GRID_MAX_TRIALS)
    return max(1, int(budget))


def _ellipse_polygon(mean, cov, n: int = 48, n_sigma: float = 2.0):
    """A closed polygon outlining the n_sigma contour of a 2D Gaussian."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    vals = np.maximum(vals, 1e-12)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    axes = vecs @ np.diag(n_sigma * np.sqrt(vals))
    pts = (axes @ circle).T + np.asarray(mean, dtype=float)
    return pts[:, 0].tolist(), pts[:, 1].tolist()


def _xyz(point, dim: int):
    """First three coordinates of a point (for 1D/2D/3D overlays)."""
    return (float(point[0]),
            float(point[1]) if dim > 1 else 0.0,
            float(point[2]) if dim > 2 else 0.0)


def _run_cmaes(hp: dict, score_fn, budget: int, seed: int, dim: int):
    """CMA-ES via the cmaes library, capturing the 2D Gaussian per generation."""
    from cmaes import CMA

    popsize = max(3, int(hp.get("popsize", 6)))
    sigma0 = float(hp.get("sigma0", 0.2))
    bounds = np.tile(np.array([0.0, 1.0]), (dim, 1))
    optimizer = CMA(mean=np.full(dim, 0.5), sigma=sigma0, bounds=bounds,
                    population_size=popsize, seed=seed)

    xs, ys, zs, scores, ellipses, points = [], [], [], [], [], []
    n_evals = 0

    while n_evals < budget:
        if dim == 2:
            mean = np.asarray(optimizer._mean, dtype=float)
            cov = np.asarray(optimizer._C, dtype=float)
            sigma = float(optimizer._sigma)
            ex, ey = _ellipse_polygon(mean, sigma * sigma * cov)
            ellipses.append({"x": ex, "y": ey, "cx": float(mean[0]), "cy": float(mean[1])})

        solutions = []
        for _ in range(optimizer.population_size):
            if n_evals >= budget:
                break
            x = optimizer.ask()
            s = score_fn(x)
            solutions.append((x, -s))  # cmaes minimizes; we maximize score
            px, py, pz = _xyz(x, dim)
            xs.append(px)
            ys.append(py)
            zs.append(pz)
            points.append([float(v) for v in x])
            scores.append(float(s))
            n_evals += 1

        if len(solutions) == optimizer.population_size:
            optimizer.tell(solutions)
        if optimizer.should_stop():
            break

    return xs, ys, zs, scores, ellipses, popsize, points


def _run_reinforce(hp: dict, score_fn, budget: int, seed: int, dim: int):
    """Policy-gradient ES on an isotropic Gaussian N(mean, sigma^2 I).

    Learns the mean and a single scalar sigma from the standardized-reward
    gradient (the cloud translates and shrinks), but sigma is one scalar so the
    cloud is always a sphere/circle and can never stretch directionally — the
    contrast with CMA-ES's full covariance.
    """
    rng = np.random.default_rng(seed)
    popsize = max(3, int(hp.get("popsize", 6)))
    sigma = float(hp.get("sigma0", 0.2))
    lr = float(hp.get("learning_rate", 0.1))
    mean = np.full(dim, 0.5)

    xs, ys, zs, scores, ellipses, points = [], [], [], [], [], []
    n_evals = 0

    while n_evals < budget:
        if dim == 2:
            cov = (sigma * sigma) * np.eye(2)
            ex, ey = _ellipse_polygon(mean, cov)
            ellipses.append({"x": ex, "y": ey, "cx": float(mean[0]), "cy": float(mean[1])})

        pts, rewards = [], []
        for _ in range(popsize):
            if n_evals >= budget:
                break
            x = mean + sigma * rng.standard_normal(dim)
            xc = np.clip(x, 0.0, 1.0)
            s = score_fn(xc)
            pts.append(x)
            rewards.append(s)
            px, py, pz = _xyz(xc, dim)
            xs.append(px)
            ys.append(py)
            zs.append(pz)
            points.append([float(v) for v in xc])
            scores.append(float(s))
            n_evals += 1

        if len(pts) >= 2:
            pts_arr = np.array(pts)
            r = np.array(rewards)
            adv = (r - r.mean()) / (r.std() + 1e-8)
            diff = pts_arr - mean
            grad_mean = (adv[:, None] * diff / (sigma * sigma)).mean(axis=0)
            grad_logsig = (adv * ((diff ** 2).sum(axis=1) / (sigma * sigma) - dim)).mean()
            mean = np.clip(mean + lr * grad_mean, 0.0, 1.0)
            sigma = float(np.clip(sigma * np.exp(0.5 * lr * grad_logsig), 0.03, 0.6))

    return xs, ys, zs, scores, ellipses, popsize, points


def _run_optuna(algo: str, hp: dict, score_fn, budget: int, seed: int, dim: int):
    """Run a sampler-based algorithm through Optuna in `dim` dimensions."""
    sampler = _build_sampler(algo, hp, seed, dim)
    n_trials = effective_budget(algo, hp, budget, dim)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    xs, ys, zs, scores, points = [], [], [], [], []

    def objective(trial: optuna.Trial) -> float:
        point = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)]
        return score_fn(point)

    def record(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        point = [trial.params[f"x{i}"] for i in range(dim)]
        px, py, pz = _xyz(point, dim)
        xs.append(px)
        ys.append(py)
        zs.append(pz)
        points.append([float(v) for v in point])
        scores.append(float(trial.value))

    study.optimize(objective, n_trials=n_trials, callbacks=[record])
    return xs, ys, zs, scores, points


def run_optimization(
    algo: str,
    hp: dict,
    score_fn: Callable[[list], float],
    budget: int,
    seed: int,
    dim: int = 2,
) -> dict:
    """Run one optimization in `dim` dimensions and capture its trajectory.

    score_fn takes a length-`dim` vector. Returns a dict with xs, ys (first two
    coordinates per evaluation, for the 2D heatmap), points (the full vector per
    evaluation, for the per-axis strips), scores, best (running best), plus
    ellipses_xy (per-generation Gaussian, 2D only) and metadata.
    """
    if algo == "cmaes":
        xs, ys, zs, scores, ellipses_xy, popsize, points = _run_cmaes(hp, score_fn, budget, seed, dim)
        gen_size, windowed = popsize, True
    elif algo == "reinforce":
        xs, ys, zs, scores, ellipses_xy, popsize, points = _run_reinforce(hp, score_fn, budget, seed, dim)
        gen_size, windowed = popsize, True
    else:
        xs, ys, zs, scores, points = _run_optuna(algo, hp, score_fn, budget, seed, dim)
        ellipses_xy, gen_size, windowed = [], 1, False

    best: list[float] = []
    running = float("-inf")
    for s in scores:
        running = max(running, s)
        best.append(running)

    return {
        "algo": algo,
        "label": _run_label(algo, hp, len(xs), seed, dim),
        "budget": len(xs),
        "seed": seed,
        "dim": dim,
        "gen_size": gen_size,
        "windowed": windowed,
        "ellipses_xy": ellipses_xy,
        "xs": xs,
        "ys": ys,
        "zs": zs,
        "points": points,
        "scores": scores,
        "best": best,
    }


def _run_label(algo: str, hp: dict, n_trials: int, seed: int, dim: int) -> str:
    """Short legend label encoding the run's defining settings."""
    parts = [ALGO_LABELS.get(algo, algo), f"{dim}D", f"b{n_trials}", f"s{seed}"]
    if algo == "cmaes":
        parts.append(f"σ{hp.get('sigma0', 0.2)}")
        parts.append(f"pop{max(3, int(hp.get('popsize', 6)))}")
    elif algo == "reinforce":
        parts.append(f"σ{hp.get('sigma0', 0.2)}")
        parts.append(f"lr{hp.get('learning_rate', 0.1)}")
        parts.append(f"pop{max(3, int(hp.get('popsize', 6)))}")
    elif algo == "bayesian":
        parts.append(f"warm{max(1, int(hp.get('n_startup_trials', 5)))}")
    return " · ".join(parts)
