"""The ffmpeg boundary: encoding, concatenation and drawtext overlays.

ffmpeg is an external tool invoked by PATH — never a pip dependency. Every
invocation goes through :func:`_run_ffmpeg`, which degrades with a clear
message when the binary is missing instead of crashing mid-run (§8).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as err:
        raise RuntimeError(
            "ffmpeg not found on PATH — install it (https://ffmpeg.org) or add "
            "it to PATH to use sofaopt's video features."
        ) from err
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg {what} failed:\n{result.stderr}")


def encode_video(
    frame_dir: Path,
    output: Path,
    fps: int,
    *,
    crf: int = 28,
    preset: str = "fast",
) -> None:
    """Stitch PNG frames (frame_%06d.png) into an MP4."""
    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(output),
        ],
        "encode",
    )


def concat_videos(
    clip_paths: list[Path], output: Path, *, crf: int = 28, preset: str = "fast"
) -> None:
    """Concatenate MP4 clips into a single MP4 (concat demuxer)."""
    filelist = output.parent / f".{output.stem}_concat.txt"
    # as_posix(): backslashes inside single-quoted concat entries are fragile.
    filelist.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in clip_paths),
        encoding="utf-8",
    )
    try:
        _run_ffmpeg(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(filelist),
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                str(output),
            ],
            "concat",
        )
    finally:
        filelist.unlink(missing_ok=True)


def ffmpeg_escape(s: str) -> str:
    """Escape a string for use inside an ffmpeg drawtext filter value."""
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("=", "\\=")
    s = s.replace("'", "\\'")
    return s


def apply_text_overlay(
    input_path: Path,
    output_path: Path,
    line1: str,
    line2: str,
    *,
    crf: int = 28,
    preset: str = "fast",
) -> None:
    """Re-encode a video with two lines of text burned in via drawtext."""
    vf = (
        f"drawtext=text='{ffmpeg_escape(line1)}'"
        f":fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4:x=10:y=10,"
        f"drawtext=text='{ffmpeg_escape(line2)}'"
        f":fontsize=14:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4:x=10:y=36"
    )
    _run_ffmpeg(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ],
        "drawtext",
    )
