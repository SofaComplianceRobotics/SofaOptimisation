"""Dashboard callback registration."""

from .archives import register_archives_callbacks
from .config import register_config_callbacks
from .interactions import register_interactions_callbacks
from .monitoring import register_monitoring_callbacks, register_pareto_callbacks
from .run import register_run_callbacks
from .video import register_video_callbacks

__all__ = [
    "register_archives_callbacks",
    "register_config_callbacks",
    "register_interactions_callbacks",
    "register_monitoring_callbacks",
    "register_pareto_callbacks",
    "register_run_callbacks",
    "register_video_callbacks",
]
