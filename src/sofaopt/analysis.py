"""Post-hoc interaction analysis of an Optuna study.

Turns "the parameters are coupled" into a concrete *interaction map*, reusing
the trials the optimizer already ran (cheap — no extra simulations):

* **Main effects** via functional ANOVA (Optuna's built-in
  ``FanovaImportanceEvaluator``) — how much each parameter alone explains of
  the score variance.
* **Pairwise interactions** via a surrogate model (random forest) and
  **Friedman's H-statistic** — the standard dependency-light measure of how
  much two parameters act *jointly* beyond their separate effects. If the
  optional ``SALib`` package is installed, surrogate-based **second-order
  Sobol' indices** can be requested instead (``method="sobol"``).

This is the strategy-aligned replacement for the one-at-a-time (OAT)
sensitivity sweep, which cannot see the diagonal valleys created by coupling.

Typical use::

    from sofaopt.analysis import analyze, save_report
    report = analyze("runtime/study.db")
    save_report(report, "runtime/analysis")

The same :func:`analyze` result also feeds the dashboard's
"Importance / Interactions" tab.
"""

from __future__ import annotations

import contextlib
import logging
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna

logger = logging.getLogger(__name__)

# Quiet Optuna's per-call logging when we read a study purely for analysis.
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class InteractionReport:
    """Result of :func:`analyze`."""

    param_names: list[str]
    """Active (non-frozen) parameters, in matrix order."""
    main_effects: dict[str, float]
    """fANOVA importance per parameter (non-negative, sums to ~1)."""
    interaction_matrix: list[list[float]]
    """Symmetric ``N×N`` pairwise-interaction strengths (diagonal = 0)."""
    interaction_method: str
    """``"h_stat"`` (Friedman H) or ``"sobol"`` (SALib second-order)."""
    n_trials: int
    """Number of completed trials the report is based on."""

    def to_json(self) -> dict:
        return {
            "param_names": self.param_names,
            "main_effects": self.main_effects,
            "interaction_matrix": self.interaction_matrix,
            "interaction_method": self.interaction_method,
            "n_trials": self.n_trials,
        }

    def top_interactions(self, k: int = 5) -> list[tuple[str, str, float]]:
        """The ``k`` strongest parameter pairs, descending."""
        m = np.asarray(self.interaction_matrix, dtype=float)
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(self.param_names)):
            for j in range(i + 1, len(self.param_names)):
                pairs.append((self.param_names[i], self.param_names[j], float(m[i, j])))
        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs[:k]


# ---------------------------------------------------------------------------
# Study loading / data extraction
# ---------------------------------------------------------------------------

def load_study(db_path: str | Path, study_name: str | None = None) -> optuna.Study:
    """Load an existing Optuna study from a SQLite ``study.db``."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No study database at {db_path}")
    storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
    if study_name is None:
        summaries = optuna.study.get_all_study_summaries(storage=storage)
        if not summaries:
            raise ValueError(f"No studies found in {db_path}")
        study_name = summaries[0].study_name
    return optuna.load_study(study_name=study_name, storage=storage)


def _completed_xy(study: optuna.Study) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract a numeric design matrix ``X`` and objective vector ``y`` from the
    study's completed, single-objective trials.

    Columns are the union of suggested (non-frozen) parameter names, sorted for
    stable ordering. Non-numeric params are label-encoded by first appearance.
    """
    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(t.value)
    ]
    if len(trials) < 4:
        raise ValueError(
            f"Need >=4 completed trials for interaction analysis, have {len(trials)}."
        )

    names = sorted({k for t in trials for k in t.params})
    if not names:
        raise ValueError("Completed trials have no recorded parameters.")

    # Per-column encoders for any non-numeric values.
    encoders: dict[str, dict] = {}
    for name in names:
        if any(not isinstance(t.params.get(name), (int, float, bool)) for t in trials):
            seen: dict = {}
            for t in trials:
                v = t.params.get(name)
                if v not in seen:
                    seen[v] = len(seen)
            encoders[name] = seen

    rows = []
    for t in trials:
        row = []
        for name in names:
            v = t.params.get(name)
            if name in encoders:
                row.append(float(encoders[name].get(v, np.nan)))
            else:
                row.append(float(v) if v is not None else np.nan)
        rows.append(row)
    X = np.asarray(rows, dtype=float)
    y = np.asarray([t.value for t in trials], dtype=float)

    # Drop columns that never varied (frozen-in-practice) — no interaction info.
    keep = [j for j in range(X.shape[1]) if np.nanstd(X[:, j]) > 0]
    names = [names[j] for j in keep]
    X = X[:, keep]
    # Impute any stray NaNs with column means so the surrogate can fit.
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return names, X, y


# ---------------------------------------------------------------------------
# Main effects (fANOVA)
# ---------------------------------------------------------------------------

def main_effects(study: optuna.Study) -> dict[str, float]:
    """fANOVA main-effect importance per parameter (non-negative, sums to ~1)."""
    try:
        evaluator = optuna.importance.FanovaImportanceEvaluator()
        imp = optuna.importance.get_param_importances(study, evaluator=evaluator)
        return {k: float(v) for k, v in imp.items()}
    except Exception as exc:  # too few trials / single distinct value, etc.
        logger.info(f"[analysis] main-effects (fANOVA) unavailable: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Pairwise interactions
# ---------------------------------------------------------------------------

def _fit_surrogate(X: np.ndarray, y: np.ndarray, random_state: int):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as err:
        raise ImportError(
            "The interaction analysis needs scikit-learn: "
            "pip install sofaopt[analysis]"
        ) from err

    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=random_state
    )
    model.fit(X, y)
    return model


def _friedman_h(
    model, X: np.ndarray, grid_size: int, max_samples: int, random_state: int
) -> np.ndarray:
    """Friedman's H-statistic interaction matrix from a fitted surrogate.

    ``H_jk`` in ``[0, 1]`` is the share of the joint (j, k) partial-dependence
    variance that is *not* explained by the separate main effects of j and k.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if n > max_samples:
        X = X[rng.choice(n, size=max_samples, replace=False)]

    grids = []
    for j in range(d):
        g = np.unique(np.quantile(X[:, j], np.linspace(0, 1, grid_size)))
        grids.append(g)

    # Centered one-way partial dependence per feature.
    pd1 = []
    for j in range(d):
        vals = []
        for v in grids[j]:
            Xt = X.copy()
            Xt[:, j] = v
            vals.append(float(model.predict(Xt).mean()))
        arr = np.asarray(vals)
        pd1.append(arr - arr.mean())

    H = np.zeros((d, d))
    for j in range(d):
        for k in range(j + 1, d):
            gj, gk = grids[j], grids[k]
            pjk = np.zeros((len(gj), len(gk)))
            for a, vj in enumerate(gj):
                Xt = X.copy()
                Xt[:, j] = vj
                for b, vk in enumerate(gk):
                    Xt[:, k] = vk
                    pjk[a, b] = float(model.predict(Xt).mean())
            pjk -= pjk.mean()
            inter = pjk - pd1[j][:, None] - pd1[k][None, :]
            denom = float(np.sum(pjk ** 2))
            h2 = float(np.sum(inter ** 2) / denom) if denom > 0 else 0.0
            H[j, k] = H[k, j] = float(np.sqrt(max(0.0, h2)))
    return H


def _sobol_s2(X: np.ndarray, y: np.ndarray, names: list[str], random_state: int) -> np.ndarray:
    """Surrogate-based second-order Sobol' indices (requires SALib)."""
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import saltelli

    model = _fit_surrogate(X, y, random_state)
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [[float(X[:, j].min()), float(X[:, j].max())] for j in range(len(names))],
    }
    sample = saltelli.sample(problem, 1024, calc_second_order=True)
    res = sobol_analyze.analyze(problem, model.predict(sample), calc_second_order=True)
    s2 = np.nan_to_num(np.asarray(res["S2"], dtype=float))
    s2 = np.clip(s2, 0.0, None)
    s2 = np.maximum(s2, s2.T)  # symmetrize the upper-triangular S2
    return s2


def interaction_matrix(
    study: optuna.Study,
    method: str = "auto",
    grid_size: int = 6,
    max_samples: int = 256,
    random_state: int = 0,
) -> tuple[list[str], np.ndarray, str]:
    """Pairwise-interaction matrix over the study's active parameters.

    ``method``: ``"h_stat"`` (default surrogate Friedman H, sklearn only),
    ``"sobol"`` (surrogate second-order Sobol', needs SALib), or ``"auto"``
    (Sobol if SALib is importable, else H-statistic).
    """
    names, X, y = _completed_xy(study)

    if method == "auto":
        try:
            import SALib  # noqa: F401
            method = "sobol"
        except Exception:
            method = "h_stat"

    if method == "sobol":
        return names, _sobol_s2(X, y, names, random_state), "sobol"

    model = _fit_surrogate(X, y, random_state)
    H = _friedman_h(model, X, grid_size, max_samples, random_state)
    return names, H, "h_stat"


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def _dispose_study_storage(study: optuna.Study) -> None:
    """Release the SQLite file handle behind a study we opened from a path.

    An RDBStorage keeps a SQLAlchemy connection pool alive; on Windows an open
    pooled connection locks study.db, so a later ``shutil.move`` of runtime/
    (archiving from the dashboard) fails with WinError 32. Disposing the engine
    after we've read everything we need frees the handle. Best-effort — never
    let cleanup raise over a computed report."""
    with contextlib.suppress(Exception):
        storage = study._storage
        backend = getattr(storage, "_backend", storage)  # unwrap _CachedStorage
        engine = getattr(backend, "engine", None)
        if engine is not None:
            engine.dispose()


def analyze(
    study_or_path: optuna.Study | str | Path,
    method: str = "auto",
    study_name: str | None = None,
) -> InteractionReport:
    """Compute main effects + the interaction map for a study (or ``study.db``).

    When given a path we own the storage and dispose it before returning, so
    the dashboard never leaves study.db locked against archiving.
    """
    owns_storage = not isinstance(study_or_path, optuna.Study)
    study = load_study(study_or_path, study_name) if owns_storage else study_or_path
    try:
        names, matrix, used = interaction_matrix(study, method=method)
        n_done = len([
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ])
        return InteractionReport(
            param_names=names,
            main_effects=main_effects(study),
            interaction_matrix=matrix.tolist(),
            interaction_method=used,
            n_trials=n_done,
        )
    finally:
        if owns_storage:
            _dispose_study_storage(study)


def save_report(report: InteractionReport, out_dir: str | Path) -> Path:
    """Write ``interactions.json`` (and a heatmap PNG if matplotlib is present)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "interactions.json"
    json_path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = report.param_names
        m = np.asarray(report.interaction_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(1.4 + 0.6 * len(names), 1.2 + 0.6 * len(names)))
        im = ax.imshow(m, cmap="viridis", vmin=0.0)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"Pairwise interactions ({report.interaction_method})", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / "interactions.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        logger.info(f"[analysis] heatmap PNG skipped: {exc}")

    return json_path