"""Open a dashboard with the Playground tab added: ``python dashboard.py``.

Then browse http://localhost:8050 — the tab bar carries the project's built-in
tabs plus a "Playground" tab supplied entirely by this directory.

The project shown is ``examples/cube_drop``; any :class:`sofaopt.SofaOptProject`
works, since the playground reads nothing from it.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# The tab package lives beside this script; the demo project one level up.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "cube_drop"))

from project import PROJECT  # noqa: E402  # needs the sys.path entries above

from playground import playground_tab  # noqa: E402  # same

from sofaopt import launch_dashboard  # noqa: E402  # same

if __name__ == "__main__":
    launch_dashboard(
        PROJECT,
        port=8050,
        open_browser=True,
        # The whole integration: one DashboardTab, no change to sofaopt.
        extra_tabs=[playground_tab(before="archives")],
    )
