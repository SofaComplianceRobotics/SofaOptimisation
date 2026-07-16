"""The ffmpeg boundary: encoding, concatenation and drawtext overlays.

ffmpeg is an external tool invoked by PATH — never a pip dependency. Every
invocation goes through :func:`_run_ffmpeg`, which degrades with a clear
message when the binary is missing instead of crashing mid-run (§8).
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

# ffmpeg's drawtext filter resolves a bare font FAMILY name via fontconfig, which
# is commonly unconfigured on Windows ffmpeg builds (fails with "Fontconfig error:
# Cannot load default config file") -- an explicit fontfile= sidesteps fontconfig
# entirely. Checked in order; first match wins.
_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
)


@lru_cache(maxsize=1)
def _default_fontfile() -> str | None:
    """First existing font from ``_FONT_CANDIDATES``, ffmpeg-filter-escaped
    (forward slashes + escaped colon), or ``None`` if none exist."""
    for candidate in _FONT_CANDIDATES:
        if candidate.exists():
            return candidate.as_posix().replace(":", "\\:")
    return None


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
    fontfile = _default_fontfile()
    font_arg = f"fontfile='{fontfile}':" if fontfile else ""
    vf = (
        f"drawtext={font_arg}text='{ffmpeg_escape(line1)}'"
        f":fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4:x=10:y=10,"
        f"drawtext={font_arg}text='{ffmpeg_escape(line2)}'"
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
