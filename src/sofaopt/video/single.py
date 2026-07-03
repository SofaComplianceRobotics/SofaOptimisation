"""Render one trial's scene to an MP4, in-process via SofaPython3.

The camera injection and GL render come from
:mod:`sofaopt.scene.frame_recorder` — the *same* code the in-run recorder
uses, so an offline re-render is framed identically to a live recording
(§10 runner parity: one render path, no drift).
"""

from __future__ import annotations

import logging
import importlib.util
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from sofaopt.core import envkeys
from sofaopt.core.sofa_bootstrap import register_sofa_dll_dirs
from sofaopt.scene.frame_recorder import inject_camera_and_lights, render_frame_raw
from sofaopt.video.ffmpeg import encode_video
from sofaopt.video.overlay import add_overlay_text, build_overlay_text, write_png

logger = logging.getLogger(__name__)


@contextmanager
def _env_overlay(extra: dict):
    """Temporarily patch os.environ, restoring originals on exit."""
    old = {}
    for k, v in extra.items():
        old[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _import_sofa_and_pygame():
    """Import the render stack with actionable error messages."""
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            "generate_trial_video requires the [video] extra: "
            "pip install sofaopt[video]"
        ) from exc

    # Must precede the first SOFA import (Windows DLL search rules).
    register_sofa_dll_dirs()
    try:
        import Sofa
        import Sofa.Core
        import Sofa.Simulation
        import Sofa.SofaGL
        import SofaRuntime
    except ImportError as exc:
        raise ImportError(
            "generate_trial_video requires SOFA Python bindings. "
            "Make sure SOFA_ROOT is set and SOFA's python site-packages "
            "directory is on PYTHONPATH."
        ) from exc
    return pygame, Sofa, SofaRuntime


def _prepare_assets(project, params: dict, trial_dir: Path) -> tuple[dict, Path | None]:
    """Re-run the prepare hook into a scratch dir so scene assets exist."""
    if project.prepare_trial is None:
        return {}, None
    video_prep_dir = trial_dir / "_video_prep"
    # Wipe stale files from a previous video generation before rebuilding.
    shutil.rmtree(video_prep_dir, ignore_errors=True)
    video_prep_dir.mkdir(parents=True)
    (video_prep_dir / "params.json").write_text(
        json.dumps(params, indent=2), encoding="utf-8"
    )
    prep = project.prepare_trial(params, video_prep_dir)
    extra_env = (
        {k: str(v) for k, v in prep.env.items()} if prep is not None else {}
    )
    return extra_env, video_prep_dir


def _build_scene(Sofa, SofaRuntime, project, scene_file: Path, env_overlay: dict):
    """Load the scene module and build the (not yet initialized) graph."""
    sofa_root = str(project.sofa_env.get("SOFA_ROOT", "")) or os.environ.get(
        "SOFA_ROOT", ""
    )
    if sofa_root:
        for pd in (Path(sofa_root) / "bin" / "Release", Path(sofa_root) / "bin"):
            SofaRuntime.PluginRepository.addFirstPath(str(pd))

    with _env_overlay(env_overlay):
        spec = importlib.util.spec_from_file_location("_sofa_video_scene", scene_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load scene file: {scene_file}")
        scene_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scene_mod)

        root = Sofa.Core.Node("root")
        scene_mod.createScene(root)

    # Same camera/lights as the in-run recorder; before initRoot.
    inject_camera_and_lights(root)
    return root


def _capture_loop(
    pygame,
    Sofa,
    root,
    *,
    width: int,
    height: int,
    frame_skip: int,
    max_steps: int | None,
    overlay_text: str | None,
    frame_dir: Path,
) -> tuple[int, int]:
    """Animate + capture frames until the scene stops. Returns (steps, captured)."""
    step = 0
    captured = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return step, captured

        Sofa.Simulation.animate(root, root.dt.value)

        if step % frame_skip == 0:
            buff = render_frame_raw(root, Sofa.SofaGL, Sofa.Simulation, width, height)
            if buff:
                frame = np.frombuffer(buff, dtype=np.uint8).reshape(height, width, 3)
                frame = np.flipud(frame)  # OpenGL origin is bottom-left
                if overlay_text:
                    frame = add_overlay_text(frame, overlay_text)
                write_png(frame, frame_dir / f"frame_{captured:06d}.png")
                captured += 1
            pygame.display.flip()

        step += 1

        # Check after capturing so the final frame (e.g. cube on floor) is included
        try:
            if not root.animate.value:
                break
        except Exception:
            pass
    return step, captured


def generate_trial_video(
    project,
    trial_dir: Path | str,
    output_path: Path | str,
    *,
    test_name: str | None = None,
    width: int = 800,
    height: int = 600,
    fps: int = 30,
    max_steps: int | None = None,
    crf: int = 28,
    preset: str = "fast",
    frame_skip: int = 16,
    text_overlay: bool = False,
    _overlay_meta: dict | None = None,
) -> None:
    """Generate a video for a completed trial.

    Runs the trial's scene in-process using SofaPython3 + pygame/OpenGL,
    captures rendered frames, and encodes to MP4 via ffmpeg.

    The scene is run with ``is_optimizing=False`` (no OPT_TRIAL_STATE_PATH),
    so controllers that call ``trial.write_score()`` stop the simulation
    cleanly (``root.animate = False``) without killing the process.

    Args:
        project:      The SofaOptProject that produced this trial.
        trial_dir:    Path to the trial directory (must contain params.json).
        output_path:  Where to write the output MP4.
        test_name:    Which test's scene to render (default: first test).
        width:        Output width in pixels.
        height:       Output height in pixels.
        fps:          Output frame rate.
        max_steps:    Safety cap on simulation steps (default: run until the
                      scene stops by itself via root.animate = False).
        crf:          ffmpeg CRF quality (0=lossless, 51=worst; default 28).
        preset:       ffmpeg preset: ultrafast/fast/medium/slow (default fast).
        frame_skip:   Capture every Nth frame (default 16). At dt=0.01 s and
                      30 fps output, frame_skip=16 gives ~1.9x real-time playback.
        text_overlay: Burn trial ID, score and params into each frame.
        _overlay_meta: Internal — extra fields for the overlay text (score,
                      gen_name, trial_name). Populated by batch/summary helpers.
    """
    pygame, Sofa, SofaRuntime = _import_sofa_and_pygame()

    trial_dir = Path(trial_dir)
    output_path = Path(output_path)

    params_path = trial_dir / "params.json"
    params: dict[str, Any] = {}
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))

    test_spec = project.tests[0] if test_name is None else project.test(test_name)

    extra_env, video_prep_dir = _prepare_assets(project, params, trial_dir)

    # OPT_TRIAL_STATE_PATH intentionally absent → is_optimizing=False.
    root = _build_scene(
        Sofa,
        SofaRuntime,
        project,
        test_spec.scene_file,
        {**extra_env, envkeys.PARAMS_PATH: str(params_path)},
    )

    Sofa.Simulation.initRoot(root)
    root.animate = True  # Sofa.Core.Node starts with animate=False

    pygame.display.init()
    # HIDDEN like frame_recorder: rendering only reads the back buffer (no swap), so the
    # window surface is never shown — without this the render window pops up on the desktop.
    pygame.display.set_mode(
        (width, height),
        pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME | getattr(pygame, "HIDDEN", 0),
    )
    pygame.display.set_caption("sofaopt video render")
    if os.name == "nt":
        try:
            import ctypes
            hwnd = pygame.display.get_wm_info().get("window", 0)
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass

    Sofa.SofaGL.glewInit()
    Sofa.Simulation.initVisual(root)
    Sofa.Simulation.initTextures(root)

    overlay_text: str | None = None
    if text_overlay:
        meta = _overlay_meta or {}
        overlay_text = build_overlay_text(
            params,
            gen_name=meta.get("gen_name", ""),
            trial_name=meta.get("trial_name", str(trial_dir.name)),
            score=meta.get("score"),
            test_name=test_spec.name if test_name else "",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f".{output_path.stem}_frames"
    frame_dir.mkdir(exist_ok=True)

    try:
        step, captured = _capture_loop(
            pygame,
            Sofa,
            root,
            width=width,
            height=height,
            frame_skip=frame_skip,
            max_steps=max_steps,
            overlay_text=overlay_text,
            frame_dir=frame_dir,
        )
    finally:
        pygame.quit()

    if captured == 0:
        raise RuntimeError(
            "No frames were captured — the simulation stopped before the first step."
        )

    logger.info(f"[video] Captured {captured} frames ({step} sim steps) at {width}x{height}")

    encode_video(frame_dir, output_path, fps=fps, crf=crf, preset=preset)
    shutil.rmtree(frame_dir, ignore_errors=True)
    # The prepare hook owns everything under _video_prep; wipe it wholesale.
    if video_prep_dir is not None:
        shutil.rmtree(video_prep_dir, ignore_errors=True)

    logger.info(f"[video] Saved: {output_path}")
