"""Run the liver multi-load-case registration headless.

Usage::

    python run.py    # CMA-ES identification against both load cases
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
