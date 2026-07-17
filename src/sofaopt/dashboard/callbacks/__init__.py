"""Dashboard callback registration."""

from .archives import register_archives_callbacks
from .config import register_config_callbacks
from .monitor import register_monitor_callbacks
from .parameters import register_parameters_callbacks
from .pareto import register_pareto_callbacks
from .results import register_results_callbacks
from .run import register_run_callbacks
from .video import register_video_callbacks

__all__ = [
    "register_archives_callbacks",
    "register_config_callbacks",
    "register_monitor_callbacks",
    "register_parameters_callbacks",
    "register_pareto_callbacks",
    "register_results_callbacks",
    "register_run_callbacks",
    "register_video_callbacks",
]
