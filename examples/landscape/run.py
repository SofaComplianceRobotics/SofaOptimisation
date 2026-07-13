"""Run the analytic landscape harness headless.

Usage::

    python run.py                     # default: rastrigin, IPOP restarts on
    python run.py --variant converged # self-size via run_until_converged
    python run.py --variant plain     # plain CMA-ES baseline (no restarts)
    python run.py --variant noisy      # noisy objective -> racing (run_count_min)

Edit FUNCTION / DIM / NOISE at the top of project.py to change the landscape.
Compare variants on the dashboard's Archives tab (overlaid convergence curves);
restart markers on the Performance curve show where each IPOP restart fired.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project import get_variant

from sofaopt import run_optimization

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["default", "converged", "plain", "noisy"],
        default="default",
        help="Project variant to run (default: %(default)s)",
    )
    args = parser.parse_args()
    run_optimization(get_variant(args.variant))
