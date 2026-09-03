"""Run the liver elastography study headless.

Usage::

    python run.py    # CMA-ES stiffness-field identification (8 params, 3 tests)

Study knobs are set on the environment before launch, e.g. select fewer load
cases to study identifiability-vs-number-of-tests (see README).
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
        choices=["default"],
        default="default",
        help="Project variant to run (default: %(default)s)",
    )
    args = parser.parse_args()
    run_optimization(get_variant(args.variant))
