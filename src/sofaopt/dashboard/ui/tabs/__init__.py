"""Tab builders for the dashboard."""

from .archives import build_archives_tab
from .bounds import build_param_bounds_tab
from .config import build_config_tab
from .interactions import build_interactions_tab
from .pareto import build_pareto_tab
from .performance import build_performance_tab
from .progress import build_progress_tab
from .run import build_run_tab
from .styles import LOG_STYLE

__all__ = [
    "LOG_STYLE",
    "build_archives_tab",
    "build_config_tab",
    "build_interactions_tab",
    "build_param_bounds_tab",
    "build_pareto_tab",
    "build_performance_tab",
    "build_progress_tab",
    "build_run_tab",
]
