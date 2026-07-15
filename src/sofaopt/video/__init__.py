"""Offline video generation for sofaopt trials (the ``[video]`` extra).

Renders trial scenes in-process with SofaPython3 + pygame/OpenGL — sharing
the exact camera/render path of the in-run recorder
(:mod:`sofaopt.scene.frame_recorder`) — and encodes/concatenates via ffmpeg
(an external tool expected on PATH).

Public API::

    from sofaopt.video import (
        generate_trial_video,      # one trial → MP4
        generate_selected_videos,  # top/bottom-N batch
        generate_summary_video,    # highlight reel
        cleanup_trial_recordings,  # prune cached trial.mp4s
        apply_generation_overlays, # burn score/params into in-run recordings
    )

CLI: ``python -m sofaopt.video --help``.
"""

from sofaopt.video.select import (
    apply_generation_overlays,
    cleanup_trial_recordings,
    generate_selected_videos,
    generate_summary_video,
)
from sofaopt.video.single import generate_trial_video

__all__ = [
    "apply_generation_overlays",
    "cleanup_trial_recordings",
    "generate_selected_videos",
    "generate_summary_video",
    "generate_trial_video",
]
