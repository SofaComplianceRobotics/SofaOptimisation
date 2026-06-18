"""Open the cantilever dashboard: ``python dashboard.py`` then browse :8050."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project import PROJECT  # noqa: E402

from sofaopt import launch_dashboard  # noqa: E402

if __name__ == "__main__":
    launch_dashboard(PROJECT, port=8050, open_browser=True)
