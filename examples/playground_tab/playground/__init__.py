"""Optimizer playground — a teaching tab supplied from outside ``sofaopt``.

This package is a worked example of the dashboard's extension point: it adds a
whole tab to a project's dashboard without any change to ``sofaopt`` itself.

The engine (:mod:`.objectives`, :mod:`.optimizers`) is pure compute — synthetic
2D landscapes plus real Optuna sampler runs over them, with no Dash dependency,
so it can be exercised on its own. The Dash layer (:mod:`.tab`,
:mod:`.callbacks`) is wired to the app through :class:`sofaopt.DashboardTab`.

Usage::

    from sofaopt import launch_dashboard
    from playground import playground_tab

    launch_dashboard(PROJECT, extra_tabs=[playground_tab()])

The landscapes are synthetic, so nothing here reads a project, a study, or any
trial data — the tab is for building intuition about how samplers explore, not
for inspecting a real run.
"""

from sofaopt import DashboardTab

from .callbacks import register_playground_callbacks
from .tab import build_playground_tab

__all__ = ["playground_tab"]


def playground_tab(before: str | None = None) -> DashboardTab:
    """Return the playground as a :class:`sofaopt.DashboardTab`.

    Args:
        before: Insert before this built-in tab id (e.g. ``"archives"``).
            Default appends after the built-ins.
    """
    return DashboardTab(
        label="Playground",
        value="playground",
        build=build_playground_tab,
        register=register_playground_callbacks,
        before=before,
    )
