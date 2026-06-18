"""Tab builders for the dashboard."""

from .bounds import build_param_bounds_tab
from .config import build_config_tab
from .optimize import PIE_PALETTE, _equal_split, build_optimise_tab
from .performance import build_performance_tab
from .progress import build_progress_tab
from .scenes import build_scenes_tab
from .styles import LOG_STYLE

__all__ = [
    "build_config_tab",
    "build_scenes_tab",
    "build_optimise_tab",
    "build_performance_tab",
    "build_param_bounds_tab",
    "build_progress_tab",
    "LOG_STYLE",
    "PIE_PALETTE",
    "_equal_split",
]
