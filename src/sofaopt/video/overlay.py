"""Frame post-processing: PNG export and in-frame text overlays (PIL path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_png(frame: np.ndarray, path: Path) -> None:
    """Write an (H, W, 3) uint8 RGB array as PNG. Tries PIL, imageio, cv2."""
    try:
        from PIL import Image

        Image.fromarray(frame, "RGB").save(str(path))
        return
    except ImportError:
        pass
    try:
        import imageio  # type: ignore

        imageio.imwrite(str(path), frame)
        return
    except ImportError:
        pass
    try:
        import cv2  # type: ignore

        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return
    except ImportError:
        pass
    raise ImportError(
        "No image library found. Install the [video] extra "
        "(pip install sofaopt[video]) or one of: Pillow, imageio, opencv-python"
    )


def add_overlay_text(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn ``text`` into the top-left corner of an RGB frame using PIL."""
    try:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame, "RGB")
        draw = ImageDraw.Draw(img)
        # Black shadow for readability on any background
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw.text((10 + dx, 10 + dy), text, fill=(0, 0, 0))
        draw.text((10, 10), text, fill=(255, 255, 255))
        return np.array(img)
    except Exception:
        return frame


def build_overlay_text(
    params: dict,
    *,
    gen_name: str = "",
    trial_name: str = "",
    score: float | None = None,
    test_name: str = "",
) -> str:
    """Build a one-or-two-line overlay string from trial metadata."""
    header = " / ".join(x for x in (gen_name, trial_name, test_name) if x)
    if score is not None:
        header += f"   score: {score:.1f}"
    param_str = "  ".join(
        f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in list(params.items())[:6]
    )
    return f"{header}\n{param_str}" if param_str else header


def format_param_line(params: dict) -> str:
    """One-line ``k=v`` summary of the first six params (drawtext line 2)."""
    return "  ".join(
        f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in list(params.items())[:6]
    )
