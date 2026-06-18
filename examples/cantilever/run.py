"""Run the cantilever optimization headless: ``python run.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project import PROJECT  # noqa: E402

from sofaopt import run_optimization  # noqa: E402

if __name__ == "__main__":
    run_optimization(PROJECT)
