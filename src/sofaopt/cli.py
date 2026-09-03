"""Command-line entry point: ``sofaopt <project.py> [options]``.

Runs a project's optimization headless from a terminal — the same
``run_optimization`` the dashboard's Run button spawns and any project ``run.py``
calls. Optional flags overlay optimizer settings onto the project (the CLI
equivalent of the dashboard's ``OPT_*`` overrides), so a project no longer
*needs* a hand-rolled ``run.py`` just to be runnable.

    sofaopt examples/cube_drop/project.py
    sofaopt examples/cube_drop/project.py --sampler tpe --parallel 8 --gens 50
    sofaopt path/to/project.py --prune-mode shadow

A run resumes automatically when its ``study.db`` already exists; ``--resume`` is
accepted only for clarity. Also available as ``python -m sofaopt ...``.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from sofaopt.project import load_project_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sofaopt",
        description="Run a sofaopt optimization headless.",
    )
    p.add_argument(
        "project", type=Path,
        help="Path to project.py (must define PROJECT = SofaOptProject(...)).",
    )
    p.add_argument(
        "--attr", default="PROJECT",
        help="Project variable name in the file (default: PROJECT).",
    )
    p.add_argument("--sampler", choices=["cmaes", "gp", "tpe", "random"], help="Override the sampler.")
    p.add_argument(
        "--seed-sampler", choices=["sobol", "random"], dest="seed_sampler",
        help="Override the initial-design sampler.",
    )
    p.add_argument("--parallel", type=int, help="Override n_parallel (population / trials per generation).")
    p.add_argument("--gens", type=int, help="Override n_generations.")
    p.add_argument("--restart-patience", type=int, dest="restart_patience", help="Override restart_patience.")
    p.add_argument(
        "--cmaes-restarts", type=int, dest="cmaes_restarts",
        help="Max IPOP restarts (0 = a stall stops the run).",
    )
    p.add_argument(
        "--stall-generations", type=int, dest="stall_generations",
        help="Generations without improvement that trigger a restart/stop (0 = off).",
    )
    p.add_argument(
        "--inc-popsize", type=int, dest="cmaes_inc_popsize",
        help="Population multiplier applied at each IPOP restart.",
    )
    for flag, field, on_help in (
        ("restart-on-convergence", "restart_on_convergence",
         "Trigger restarts on CMA-ES's convergence signal instead of the stall plateau."),
        ("warm-restarts", "warm_restarts",
         "Re-seed each restart from the best-so-far instead of a random point."),
        ("dedup-trials", "dedup_trials",
         "Reuse recorded scores for repeated param vectors (deterministic objectives only)."),
    ):
        grp = p.add_mutually_exclusive_group()
        grp.add_argument(f"--{flag}", dest=field, action="store_true", default=None, help=on_help)
        grp.add_argument(f"--no-{flag}", dest=field, action="store_false", help=f"Disable {flag.replace('-', ' ')}.")
    p.add_argument(
        "--prune-mode", choices=["off", "shadow", "kill"], dest="prune_mode",
        help="Multi-fidelity pruning mode (needs a prunable test).",
    )
    margin = p.add_mutually_exclusive_group()
    margin.add_argument(
        "--cmaes-margin", dest="cmaes_margin", action="store_true", default=None,
        help="Enable CMA-ES with Margin.",
    )
    margin.add_argument(
        "--no-cmaes-margin", dest="cmaes_margin", action="store_false",
        help="Disable CMA-ES with Margin.",
    )
    conv = p.add_mutually_exclusive_group()
    conv.add_argument(
        "--run-until-converged", dest="run_until_converged", action="store_true", default=None,
        help="Stop when restarts stop improving (needs CMA-ES + restarts).",
    )
    conv.add_argument(
        "--no-run-until-converged", dest="run_until_converged", action="store_false",
        help="Use a fixed generation budget instead.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="No-op: a run resumes automatically when study.db exists.",
    )
    return p


def _overrides(args: argparse.Namespace) -> dict:
    """Non-None flags as SofaOptProject field overrides (skip unspecified)."""
    fields = {
        "sampler": args.sampler,
        "seed_sampler": args.seed_sampler,
        "n_parallel": args.parallel,
        "n_generations": args.gens,
        "restart_patience": args.restart_patience,
        "prune_mode": args.prune_mode,
        "cmaes_with_margin": args.cmaes_margin,
        "run_until_converged": args.run_until_converged,
        "cmaes_restarts": args.cmaes_restarts,
        "stall_generations": args.stall_generations,
        "cmaes_inc_popsize": args.cmaes_inc_popsize,
        "restart_on_convergence": args.restart_on_convergence,
        "warm_restarts": args.warm_restarts,
        "dedup_trials": args.dedup_trials,
    }
    return {k: v for k, v in fields.items() if v is not None}


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    # Lazy: keeps `sofaopt --help` from importing Optuna and the whole engine.
    from sofaopt import run_optimization

    project = load_project_file(args.project, attr=args.attr)
    overrides = _overrides(args)
    if overrides:
        # replace() re-runs the project's own validation (e.g. prune-mode shape).
        project = dataclasses.replace(project, **overrides)
    run_optimization(project)


if __name__ == "__main__":
    main()
