"""Run OAT sensitivity analysis on the cube-drop project.

Sweeps cube_size and cube_mass independently and reports which parameter
drives the score more.  Run this before a full optimisation to understand
the parameter landscape.

Usage::

    python sensitivity_test.py
    python sensitivity_test.py --n-samples 7
    python sensitivity_test.py --n-samples 3 --test fall
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project import PROJECT  # noqa: E402

from sofaopt import run_sensitivity_analysis  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Samples per parameter sweep (min 2, default %(default)s)",
    )
    parser.add_argument(
        "--test", default=None,
        help="Test name to score against (default: first test)",
    )
    args = parser.parse_args()

    run_sensitivity_analysis(
        PROJECT,
        n_samples=args.n_samples,
        test_name=args.test,
    )
