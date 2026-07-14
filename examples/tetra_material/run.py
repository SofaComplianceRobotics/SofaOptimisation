"""Run the single-tetra material identification headless.

Usage::

    python run.py                  # deterministic identification (CMA-ES, python runner)
    python run.py --variant noisy  # noisy identification -> racing (run_count_min)
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
        choices=["default", "noisy"],
        default="default",
        help="Project variant to run (default: %(default)s)",
    )
    args = parser.parse_args()
    run_optimization(get_variant(args.variant))
