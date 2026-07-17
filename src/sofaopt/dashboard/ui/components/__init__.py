"""Reusable dashboard UI components shared across tabs."""

from .log_view import build_log_view, filter_log_lines, register_log_view

__all__ = ["build_log_view", "filter_log_lines", "register_log_view"]
