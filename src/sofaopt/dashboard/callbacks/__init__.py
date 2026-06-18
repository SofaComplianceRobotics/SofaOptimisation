"""Dashboard callback registration."""

from .config import register_config_callbacks
from .monitoring import register_monitoring_callbacks
from .optimize import register_optimise_callbacks
from .scenes import register_scene_callbacks

__all__ = [
    "register_config_callbacks",
    "register_monitoring_callbacks",
    "register_optimise_callbacks",
    "register_scene_callbacks",
]
