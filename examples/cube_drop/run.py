"""Run the cube-drop demo headless.

Usage::

    python run.py                          # default (CMA-ES, runSofa)
    python run.py --variant tpe            # TPE Bayesian sampler
    python run.py --variant python         # Python in-process runner
    python run.py --variant pareto         # Multi-objective NSGA-II (Pareto front, runSofa)
    python run.py --variant pareto-python  # Multi-objective + Python in-process runner
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project import (  # noqa: E402
    PROJECT,
    PROJECT_MULTI_OBJ,
    PROJECT_MULTI_OBJ_PYTHON,
    PROJECT_PYTHON_RUNNER,
    PROJECT_TPE,
)

from sofaopt import run_optimization  # noqa: E402

_VARIANTS = {
    "default":       PROJECT,
    "tpe":           PROJECT_TPE,
    "python":        PROJECT_PYTHON_RUNNER,
    "pareto":        PROJECT_MULTI_OBJ,
    "pareto-python": PROJECT_MULTI_OBJ_PYTHON,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=list(_VARIANTS),
        default="default",
        help="Project variant to run (default: %(default)s)",
    )
    args = parser.parse_args()
    run_optimization(_VARIANTS[args.variant])
