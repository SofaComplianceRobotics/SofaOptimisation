"""IPOP-style CMA-ES restarts (Auger & Hansen 2005), orchestrator-level.

On a multimodal landscape a converged CMA-ES is stuck in one basin; the
standard answer is to *restart* it with a larger population (IPOP), which
searches more globally. Optuna deprecated ``CmaEsSampler(restart_strategy=...)``
in v4.4 (the argument silently falls back to ``None``), so sofaopt implements
the restart where its own stagnation signal already fires: when the stall
tracker sees ``stall_generations`` generations without a best-score
improvement, the run swaps in a fresh CMA-ES with population size multiplied
by ``cmaes_inc_popsize``, the initial ``cmaes_sigma0``, and a uniform-random
start point (the canonical IPOP re-seed), instead of stopping.

Forcing a genuine restart inside one Optuna study leans on how
``CmaEsSampler`` persists state: the serialized CMA object lives in *trial
system attributes* under a fixed key prefix (``cma:`` / ``cmawm:``) and is
restored from the latest completed trial carrying it. Scoping that prefix per
restart (``cma:`` -> ``cma:r1:`` -> ``cma:r2:`` ...) makes every earlier
optimizer invisible to the new sampler, so it initializes fresh with the new
population size — and a resumed run rebuilds the sampler for the restart
index recorded in the study's user attributes, landing on the right state.
The larger population simply means the internal CMA update spans several
sofaopt generations (``n_parallel`` asks each); Optuna handles that natively.
"""

from __future__ import annotations

import logging
import random

import optuna

logger = logging.getLogger(__name__)

# Study-level user attribute persisting how many restarts have happened.
RESTART_ATTR = "sofaopt:cma_restarts"


class _RestartScopedCmaEsSampler(optuna.samplers.CmaEsSampler):
    """``CmaEsSampler`` whose persisted CMA state is scoped to one restart.

    Relies on the private ``_attr_prefix`` seam of Optuna's sampler (stable
    across 3.x/4.x; ``tests/test_restart.py`` trips if upstream changes it).
    """

    def __init__(self, restart_index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        if restart_index > 0:
            self._attr_prefix = f"{self._attr_prefix}r{restart_index}:"


def restart_index(study: optuna.Study) -> int:
    """How many restarts this study has performed (0 = initial run)."""
    return int(study.user_attrs.get(RESTART_ATTR, 0))


def restart_popsize(project, index: int) -> int:
    """IPOP population size at restart ``index``: n_parallel × inc^index."""
    return project.n_parallel * project.cmaes_inc_popsize**index


def _random_x0(project, index: int) -> dict:
    """Uniform-random restart point over the searched box (canonical IPOP).

    Deterministic per restart index so a resumed run rebuilds the identical
    sampler configuration.
    """
    rng = random.Random(project.seed_sampler_seed + index)  # noqa: S311 — restart re-seed, not cryptographic
    x0: dict = {}
    for p in project.params:
        if p.is_frozen or p.type not in ("float", "int"):
            continue
        x0[p.name] = (
            rng.uniform(p.low, p.high)
            if p.type == "float"
            else rng.randint(int(p.low), int(p.high))
        )
    return x0


def build_restart_sampler(
    project, index: int, independent_sampler: optuna.samplers.BaseSampler
) -> optuna.samplers.BaseSampler:
    """The CMA-ES sampler for restart ``index`` (index 0 = the initial one is
    built by ``build_study``, not here). No startup phase: the landscape was
    explored before the first stall; the restart goes straight to CMA-ES."""
    return _RestartScopedCmaEsSampler(
        restart_index=index,
        x0=_random_x0(project, index) or None,
        sigma0=project.cmaes_sigma0,
        popsize=restart_popsize(project, index),
        n_startup_trials=0,
        consider_pruned_trials=True,
        with_margin=project.cmaes_with_margin,
        independent_sampler=independent_sampler,
    )


def maybe_restart(
    study: optuna.Study,
    project,
    independent_sampler: optuna.samplers.BaseSampler,
) -> bool:
    """On a stall: swap in the next IPOP restart. False when restarts are
    off, don't apply (non-CMA-ES sampler, multi-objective), or the restart
    budget is spent — the caller then stops the run as before."""
    if (
        project.cmaes_restarts <= 0
        or project.sampler != "cmaes"
        or project.multi_objective
    ):
        return False
    index = restart_index(study)
    if index >= project.cmaes_restarts:
        logger.info(
            f"[restart] Restart budget spent ({index}/{project.cmaes_restarts})."
        )
        return False

    new_index = index + 1
    study.set_user_attr(RESTART_ATTR, new_index)
    study.sampler = build_restart_sampler(project, new_index, independent_sampler)
    logger.info(
        f"[restart] IPOP restart {new_index}/{project.cmaes_restarts}: "
        f"popsize {restart_popsize(project, index)} -> "
        f"{restart_popsize(project, new_index)}, sigma0 {project.cmaes_sigma0}, "
        f"uniform-random x0."
    )
    return True
