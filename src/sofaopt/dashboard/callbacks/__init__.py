"""Dashboard callback registration."""

from .archives import register_archives_callbacks
from .config import register_config_callbacks
from .interactions import register_interactions_callbacks
from .monitoring import register_monitoring_callbacks, register_pareto_callbacks
from .optimize import register_optimise_callbacks
from .playground import register_playground_callbacks
from .scenes import register_scene_callbacks
from .video import register_video_callbacks

__all__ = [
    "register_archives_callbacks",
    "register_config_callbacks",
    "register_interactions_callbacks",
    "register_monitoring_callbacks",
    "register_pareto_callbacks",
    "register_optimise_callbacks",
    "register_playground_callbacks",
    "register_scene_callbacks",
    "register_video_callbacks",
]
