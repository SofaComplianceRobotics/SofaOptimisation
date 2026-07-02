"""Tab builders for the dashboard."""

from .archives import build_archives_tab
from .bounds import build_param_bounds_tab
from .config import build_config_tab
from .interactions import build_interactions_tab
from .optimize import PIE_PALETTE, _equal_split, build_optimise_tab
from .pareto import build_pareto_tab
from .performance import build_performance_tab
from .progress import build_progress_tab
from .scenes import build_scenes_tab
from .styles import LOG_STYLE

__all__ = [
    "build_archives_tab",
    "build_config_tab",
    "build_scenes_tab",
    "build_optimise_tab",
    "build_performance_tab",
    "build_param_bounds_tab",
    "build_progress_tab",
    "build_pareto_tab",
    "build_interactions_tab",
    "LOG_STYLE",
    "PIE_PALETTE",
    "_equal_split",
]
