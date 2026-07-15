"""Open the cube-drop dashboard: ``python dashboard.py`` then browse http://localhost:8050.

Usage::

    python dashboard.py                          # default project
    python dashboard.py --variant tpe            # TPE sampler run
    python dashboard.py --variant gp             # GP Bayesian optimization run
    python dashboard.py --variant sobol          # Sobol' space-filling startup run
    python dashboard.py --variant python         # Python runner run
    python dashboard.py --variant pareto         # Multi-objective run (shows Pareto tab)
    python dashboard.py --variant pareto-python  # Multi-objective + Python runner
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project import (
    PROJECT,
    PROJECT_GP,
    PROJECT_MULTI_OBJ,
    PROJECT_MULTI_OBJ_PYTHON,
    PROJECT_PYTHON_RUNNER,
    PROJECT_SOBOL,
    PROJECT_TPE,
)

from sofaopt import launch_dashboard

_VARIANTS = {
    "default":       PROJECT,
    "tpe":           PROJECT_TPE,
    "gp":            PROJECT_GP,
    "sobol":         PROJECT_SOBOL,
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
        help="Project variant to view (default: %(default)s)",
    )
    args = parser.parse_args()
    launch_dashboard(_VARIANTS[args.variant], port=8050, open_browser=True)
