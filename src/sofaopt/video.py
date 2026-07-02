"""Video generation for sofaopt trials using SofaPython3 in-process rendering.

Runs the scene directly in Python (no runSofa subprocess), drives the simulation
via Sofa.Simulation.animate(), renders each frame with Sofa.SofaGL.draw() into a
pygame/OpenGL window, captures pixels with glReadPixels, then stitches to MP4
via ffmpeg.

Requirements (pip install):
    pygame PyOpenGL Pillow          # rendering + frame export
    ffmpeg accessible on PATH       # video encoding

Usage (API):
    from sofaopt.video import generate_trial_video
    generate_trial_video(project, "runtime/trials/gen_0001/trial_01", "out.mp4")

Usage (CLI):
    python -m sofaopt.video \\
        --project examples/cube_drop/project.py \\
        --trial   runtime/trials/gen_0001/trial_01 \\
        --output  trial_01.mp4
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _write_png(frame: np.ndarray, path: Path) -> None:
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
        "No image library found. Install one of: Pillow, imageio, opencv-python"
    )


def _encode_video(
    frame_dir: Path,
    output: Path,
    fps: int,
    *,
    crf: int = 28,
    preset: str = "fast",
) -> None:
    """Stitch PNG frames into an MP4 using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def _concat_videos(clip_paths: list[Path], output: Path, *, crf: int = 28, preset: str = "fast") -> None:
    """Concatenate MP4 clips into a single MP4 using ffmpeg concat demuxer."""
    filelist = output.parent / f".{output.stem}_concat.txt"
    filelist.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clip_paths),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(filelist),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")
    finally:
        filelist.unlink(missing_ok=True)


def _ffmpeg_escape(s: str) -> str:
    """Escape a string for use inside ffmpeg drawtext filter text value."""
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("=", "\\=")
    s = s.replace("'", "\\'")
    return s


def _apply_text_overlay_ffmpeg(
    input_path: Path,
    output_path: Path,
    line1: str,
    line2: str,
    *,
    crf: int = 28,
    preset: str = "fast",
) -> None:
    """Re-encode a video with two lines of text burned in via ffmpeg drawtext."""
    vf = (
        f"drawtext=text='{_ffmpeg_escape(line1)}'"
        f":fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4:x=10:y=10,"
        f"drawtext=text='{_ffmpeg_escape(line2)}'"
        f":fontsize=14:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4:x=10:y=36"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg drawtext failed:\n{result.stderr}")


def _add_overlay_text(frame: np.ndarray, text: str) -> np.ndarray:
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


def _build_overlay_text(
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
    param_str = "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in list(params.items())[:6])
    return f"{header}\n{param_str}" if param_str else header


def _load_project(project_path: Path):
    """Load a PROJECT variable from a project.py file."""
    spec = importlib.util.spec_from_file_location("_sofaopt_project", project_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PROJECT"):
        raise AttributeError(
            f"{project_path} does not define a PROJECT variable. "
            "Make sure the file assigns: PROJECT = SofaOptProject(...)"
        )
    return mod.PROJECT


# ---------------------------------------------------------------------------
# Internal: camera + lighting injection
# ---------------------------------------------------------------------------

def _inject_camera_and_lights(root, *, camera_position=None, look_at=None) -> None:
    """Add a camera and two lights to ``root`` if not already present.

    runSofa provides a default camera and lighting automatically; when driving
    SOFA via Sofa.SofaGL.draw() in Python there is no implicit camera, so the
    framebuffer is black. Call this after createScene() but before initRoot().

    ``camera_position`` and ``look_at`` can be overridden per-scene if the
    defaults don't frame the geometry well.
    """
    try:
        existing_camera = root.getObject("camera")
    except Exception:
        existing_camera = None

    if existing_camera is None:
        pos = list(camera_position) if camera_position else [0.0, 40.0, 200.0]
        lat = list(look_at) if look_at else [0.0, 30.0, 0.0]
        try:
            root.addObject(
                "InteractiveCamera",
                name="camera",
                position=pos,
                lookAt=lat,
                zNear=0.1,
                zFar=2000.0,
                fieldOfView=45.0,
            )
        except Exception:
            pass

    try:
        existing_light = root.getObject("light1")
    except Exception:
        existing_light = None

    if existing_light is None:
        try:
            root.addObject(
                "DirectionalLight",
                name="light1",
                color=[1.0, 1.0, 1.0, 1.0],
                direction=[0.0, -1.0, -0.5],
            )
            root.addObject(
                "PositionalLight",
                name="light2",
                color=[0.5, 0.5, 0.5, 1.0],
                position=[100.0, 200.0, 100.0],
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    captures each rendered frame with glReadPixels, and encodes to MP4.

    The scene is run with ``is_optimizing=False`` (no OPT_TRIAL_STATE_PATH),
    so controllers that call ``trial.write_score()`` will stop the simulation
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
                      30 fps output, frame_skip=16 gives ~1.9× real-time playback.
        text_overlay: Burn trial ID, score and params into each frame.
        _overlay_meta: Internal — extra fields for the overlay text (score,
                      gen_name, trial_name). Populated by batch/summary helpers.
    """
    try:
        import pygame
        from OpenGL.GL import (
            GL_AMBIENT, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
            GL_DIFFUSE, GL_LIGHT0, GL_LIGHTING, GL_MODELVIEW, GL_POSITION,
            GL_PROJECTION, GL_RGB, GL_UNSIGNED_BYTE,
            glClear, glEnable, glLightfv, glLoadIdentity, glMatrixMode,
            glReadPixels, glViewport,
        )
        from OpenGL.GLU import gluLookAt, gluPerspective
    except ImportError as exc:
        raise ImportError(
            "generate_trial_video requires pygame and PyOpenGL:\n"
            "    pip install pygame PyOpenGL"
        ) from exc

    # Python 3.8+ on Windows no longer searches PATH for DLLs loaded by .pyd
    # extension modules. Register SOFA's bin directory explicitly before the
    # first SOFA import so the C extensions can find their DLL dependencies.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        _sofa_r = os.environ.get("SOFA_ROOT", "") or os.environ.get("SOFAPYTHON3_ROOT", "")
        if _sofa_r:
            for _d in [Path(_sofa_r) / "bin" / "Release", Path(_sofa_r) / "bin"]:
                if _d.is_dir():
                    os.add_dll_directory(str(_d))
                    break

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

    trial_dir = Path(trial_dir)
    output_path = Path(output_path)

    # -- 1. Read params -------------------------------------------------------
    params_path = trial_dir / "params.json"
    params: dict[str, Any] = {}
    if params_path.is_file():
        params = json.loads(params_path.read_text(encoding="utf-8"))

    # -- 2. Pick scene file ---------------------------------------------------
    test_spec = project.tests[0] if test_name is None else project.test(test_name)
    scene_file = test_spec.scene_file

    # -- 3. Rebuild per-trial assets via prepare hook (e.g. mesh .obj) --------
    extra_env: dict[str, str] = {}
    video_prep_dir: Path | None = None
    if project.prepare_trial is not None:
        video_prep_dir = trial_dir / "_video_prep"
        # Wipe any stale files from a previous video generation before rebuilding.
        shutil.rmtree(video_prep_dir, ignore_errors=True)
        video_prep_dir.mkdir(parents=True)
        (video_prep_dir / "params.json").write_text(
            json.dumps(params, indent=2), encoding="utf-8"
        )
        prep = project.prepare_trial(params, video_prep_dir)
        if prep is not None:
            extra_env = {k: str(v) for k, v in prep.env.items()}

    # -- 4. Register SOFA plugin search path ----------------------------------
    sofa_root = (
        str(project.sofa_env.get("SOFA_ROOT", ""))
        or os.environ.get("SOFA_ROOT", "")
    )
    if sofa_root:
        for _pd in [Path(sofa_root) / "bin" / "Release", Path(sofa_root) / "bin"]:
            SofaRuntime.PluginRepository.addFirstPath(str(_pd))

    # -- 5. Build scene graph (env overlay provides params + asset paths) ------
    # OPT_TRIAL_STATE_PATH intentionally absent → is_optimizing=False,
    # so controllers stop the sim with root.animate=False, not os.kill().
    env_overlay = {**extra_env, "OPT_PARAMS_PATH": str(params_path)}

    with _env_overlay(env_overlay):
        scene_spec = importlib.util.spec_from_file_location(
            "_sofa_video_scene", scene_file
        )
        scene_mod = importlib.util.module_from_spec(scene_spec)
        scene_spec.loader.exec_module(scene_mod)

        root = Sofa.Core.Node("root")
        scene_mod.createScene(root)

    # -- 5b. Inject default camera and lights if the scene doesn't define them.
    # runSofa provides these automatically; the Python path does not.
    # Camera and lights must be added before initRoot so they are initialized.
    _inject_camera_and_lights(root)

    # -- 6. Init physics ------------------------------------------------------
    Sofa.Simulation.initRoot(root)
    root.animate = True  # Sofa.Core.Node starts with animate=False; set explicitly

    # -- 7. OpenGL context via pygame -----------------------------------------
    pygame.display.init()
    pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("sofaopt video render")

    # -- 8. Init SOFA visual pipeline -----------------------------------------
    Sofa.SofaGL.glewInit()
    Sofa.Simulation.initVisual(root)
    Sofa.Simulation.initTextures(root)

    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glViewport(0, 0, width, height)

    # -- 9. Build overlay text (computed once, burned into every captured frame) --
    overlay_text: str | None = None
    if text_overlay:
        meta = _overlay_meta or {}
        overlay_text = _build_overlay_text(
            params,
            gen_name=meta.get("gen_name", ""),
            trial_name=meta.get("trial_name", str(trial_dir.name)),
            score=meta.get("score"),
            test_name=test_spec.name if test_name else "",
        )

    # -- 10. Simulation + capture loop ----------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f".{output_path.stem}_frames"
    frame_dir.mkdir(exist_ok=True)

    step = 0
    captured = 0
    try:
        while True:
            if max_steps is not None and step >= max_steps:
                break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            Sofa.Simulation.animate(root, root.dt.value)
            Sofa.Simulation.updateVisual(root)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glEnable(GL_LIGHTING)
            glEnable(GL_DEPTH_TEST)

            # Fallback GL light: SOFA's PositionalLight/DirectionalLight components
            # configure GL lights inside their drawVisual(), but if they weren't
            # injected or failed to initialise, geometry renders black.
            # GL_LIGHT0 here guarantees at least one light regardless.
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
            glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 1.0, 1.0, 0.0])  # directional
            glEnable(GL_LIGHT0)

            # Sofa.SofaGL.draw() renders visual geometry but does NOT configure
            # the projection/view matrices — that's normally done by the SOFA
            # viewer (SofaGLFW, Qt).  Set them up manually from the camera object.
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45.0, width / height, 0.1, 2000.0)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            try:
                cam = root.getObject("camera")
                pos = list(cam.position.value)
                lat = list(cam.lookAt.value)
                gluLookAt(pos[0], pos[1], pos[2], lat[0], lat[1], lat[2], 0.0, 1.0, 0.0)
            except Exception:
                gluLookAt(0.0, 40.0, 200.0, 0.0, 30.0, 0.0, 0.0, 1.0, 0.0)

            Sofa.SofaGL.draw(root)
            pygame.display.flip()

            if step % frame_skip == 0:
                buff = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
                frame = np.frombuffer(buff, dtype=np.uint8).reshape(height, width, 3)
                frame = np.flipud(frame)  # OpenGL origin is bottom-left
                if overlay_text:
                    frame = _add_overlay_text(frame, overlay_text)
                _write_png(frame, frame_dir / f"frame_{captured:06d}.png")
                captured += 1

            step += 1

            # Check after capturing so the final frame (e.g. cube on floor) is included
            try:
                if not root.animate.value:
                    break
            except Exception:
                pass

    finally:
        pygame.quit()

    if captured == 0:
        raise RuntimeError(
            "No frames were captured — the simulation stopped before the first step."
        )

    print(f"[video] Captured {captured} frames ({step} sim steps) at {width}x{height}")

    # -- 11. Encode -----------------------------------------------------------
    _encode_video(frame_dir, output_path, fps=fps, crf=crf, preset=preset)
    shutil.rmtree(frame_dir, ignore_errors=True)

    # Cleanup the entire _video_prep directory (owns all assets the hook generated).
    if video_prep_dir is not None:
        shutil.rmtree(video_prep_dir, ignore_errors=True)

    print(f"[video] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Batch & summary helpers
# ---------------------------------------------------------------------------

def _rank_trials(project) -> list[dict]:
    """Return all completed trial records sorted best→worst by final_score."""
    from sofaopt.dashboard import context as _ctx
    from sofaopt.dashboard.analyze_io import load_all_trials
    _ctx.set_project(project)
    records = load_all_trials()
    return sorted(
        (r for r in records if r.get("is_complete") and r.get("final_score") is not None),
        key=lambda r: r["final_score"],
        reverse=True,
    )


def generate_selected_videos(
    project,
    output_dir: Path | str,
    *,
    top_n: int = 5,
    bottom_n: int = 3,
    score_above: float | None = None,
    score_below: float | None = None,
    test_name: str | None = None,
    **video_kwargs,
) -> list[Path]:
    """Generate videos for the top/bottom N trials (or by score threshold).

    Uses ``analyze_io.load_all_trials()`` to rank trials and calls
    ``generate_trial_video()`` for each selected trial.

    Args:
        project:     The SofaOptProject whose runtime data to scan.
        output_dir:  Directory for output MP4 files.
        top_n:       Generate for this many highest-scoring trials.
        bottom_n:    Generate for this many lowest-scoring trials.
        score_above: Also include all trials with score > this value.
        score_below: Also include all trials with score < this value.
        **video_kwargs: Forwarded to ``generate_trial_video()`` (crf, preset,
                        frame_skip, text_overlay, width, height, fps, ...).

    Returns:
        List of paths to generated MP4 files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = _rank_trials(project)
    if not ranked:
        print("[video] No completed trials found.")
        return []

    selected: list[dict] = []
    seen: set[tuple] = set()

    def _add(r: dict) -> None:
        key = (r["gen_name"], r["trial_name"])
        if key not in seen:
            seen.add(key)
            selected.append(r)

    for r in ranked[:top_n]:
        _add(r)
    for r in ranked[-bottom_n:] if bottom_n else []:
        _add(r)
    if score_above is not None:
        for r in ranked:
            if r["final_score"] > score_above:
                _add(r)
    if score_below is not None:
        for r in ranked:
            if r["final_score"] < score_below:
                _add(r)

    outputs: list[Path] = []
    for i, rec in enumerate(selected, 1):
        score = rec["final_score"]
        fname = f"{rec['gen_name']}_{rec['trial_name']}_score{score:.1f}.mp4"
        out = output_dir / fname
        trial_dir = project.trials_dir / rec["gen_name"] / rec["trial_name"]
        print(f"[video] [{i}/{len(selected)}] {rec['gen_name']}/{rec['trial_name']} score={score:.1f}")
        cached = trial_dir / "trial.mp4"
        if cached.exists():
            print(f"[video]   ↳ using cached recording")
            shutil.copy2(cached, out)
            outputs.append(out)
            continue
        video_kwargs.setdefault("text_overlay", True)
        try:
            generate_trial_video(
                project,
                trial_dir,
                out,
                test_name=test_name,
                _overlay_meta={
                    "gen_name": rec["gen_name"],
                    "trial_name": rec["trial_name"],
                    "score": score,
                },
                **video_kwargs,
            )
            outputs.append(out)
        except Exception as exc:
            print(f"[video]   ↳ failed: {exc}")

    print(f"[video] Generated {len(outputs)}/{len(selected)} videos in {output_dir}")
    return outputs


def generate_summary_video(
    project,
    output_path: Path | str,
    *,
    top_n: int = 3,
    bottom_n: int = 2,
    test_name: str | None = None,
    clip_steps: int | None = None,
    **video_kwargs,
) -> None:
    """Generate a single highlight-reel MP4 from the best and worst trials.

    When cached ``trial.mp4`` recordings exist (produced during the run with
    ``record_frames=True``), uses those directly — fast, no SOFA re-run needed.
    Otherwise re-runs each trial's scene to render the video.

    Text overlay (gen / trial / score / params) is burned in via ffmpeg drawtext
    so the summary clip is self-explanatory without needing the dashboard.

    Args:
        project:     The SofaOptProject whose runtime data to scan.
        output_path: Output MP4 path.
        top_n:       Number of highest-scoring trials to include.
        bottom_n:    Number of lowest-scoring trials to include.
        clip_steps:  Max sim steps per re-rendered clip (mapped to
                     ``generate_trial_video(max_steps=...)``; ignored for
                     cached recordings).
        **video_kwargs: Forwarded to ``generate_trial_video()`` when re-rendering
                        (crf, preset, frame_skip, width, height, fps, text_overlay, ...).
    """
    import tempfile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = _rank_trials(project)
    if not ranked:
        print("[video] No completed trials found.")
        return

    clips: list[dict] = []
    seen: set[tuple] = set()

    def _add(r: dict) -> None:
        key = (r["gen_name"], r["trial_name"])
        if key not in seen:
            seen.add(key)
            clips.append(r)

    for r in ranked[:top_n]:
        _add(r)
    for r in ranked[-bottom_n:] if bottom_n else []:
        _add(r)

    video_kwargs.setdefault("text_overlay", True)
    text_overlay = video_kwargs.pop("text_overlay", True)
    crf = video_kwargs.pop("crf", 28)
    preset = video_kwargs.pop("preset", "fast")

    def _overlay_line1(rec: dict) -> str:
        label = "best" if clips.index(rec) < top_n else "worst"
        return f"{rec['gen_name']} / {rec['trial_name']}  [{label}]  score: {rec['final_score']:.1f}"

    def _overlay_line2(rec: dict) -> str:
        params = rec.get("params") or {}
        return "  ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(params.items())[:6]
        )

    # Fast path: all selected trials have cached in-run recordings.
    # Apply text overlay per-clip (if requested) then concatenate.
    cached_clips = [
        project.trials_dir / r["gen_name"] / r["trial_name"] / "trial.mp4"
        for r in clips
    ]
    if all(p.exists() for p in cached_clips):
        print(f"[video] Using {len(cached_clips)} cached recordings -> {output_path}")
        if text_overlay:
            tmp_dir = Path(tempfile.mkdtemp(prefix="sofaopt_sumovl_"))
            try:
                overlaid: list[Path] = []
                for i, (clip_path, rec) in enumerate(zip(cached_clips, clips)):
                    ovl = tmp_dir / f"clip_{i:02d}.mp4"
                    try:
                        _apply_text_overlay_ffmpeg(
                            clip_path, ovl, _overlay_line1(rec), _overlay_line2(rec),
                            crf=crf, preset=preset,
                        )
                        overlaid.append(ovl)
                    except Exception as exc:
                        print(f"[video]   overlay failed for clip {i}: {exc} -- using raw clip")
                        overlaid.append(clip_path)
                _concat_videos(overlaid, output_path, crf=crf, preset=preset)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            _concat_videos(cached_clips, output_path, crf=crf, preset=preset)
        print(f"[video] Summary saved: {output_path}")
        return

    # Slow path: re-render each trial (no cached recordings).
    tmp_dir = Path(tempfile.mkdtemp(prefix="sofaopt_summary_"))
    clip_paths: list[Path] = []
    try:
        for i, rec in enumerate(clips, 1):
            score = rec["final_score"]
            label = "best" if i <= top_n else "worst"
            trial_dir = project.trials_dir / rec["gen_name"] / rec["trial_name"]
            cached = trial_dir / "trial.mp4"
            print(f"[video] Clip {i}/{len(clips)} ({label}): "
                  f"{rec['gen_name']}/{rec['trial_name']} score={score:.1f}")
            clip_out = tmp_dir / f"clip_{i:02d}_{label}.mp4"
            if cached.exists():
                print(f"[video]   -> using cached recording")
                if text_overlay:
                    try:
                        _apply_text_overlay_ffmpeg(
                            cached, clip_out, _overlay_line1(rec), _overlay_line2(rec),
                            crf=crf, preset=preset,
                        )
                    except Exception:
                        shutil.copy2(cached, clip_out)
                else:
                    shutil.copy2(cached, clip_out)
                clip_paths.append(clip_out)
                continue
            try:
                generate_trial_video(
                    project,
                    trial_dir,
                    clip_out,
                    test_name=test_name,
                    max_steps=clip_steps,
                    crf=crf,
                    preset=preset,
                    text_overlay=text_overlay,
                    _overlay_meta={
                        "gen_name": rec["gen_name"],
                        "trial_name": rec["trial_name"],
                        "score": score,
                    },
                    **video_kwargs,
                )
                clip_paths.append(clip_out)
            except Exception as exc:
                print(f"[video]   -> failed: {exc}")

        if not clip_paths:
            raise RuntimeError("No clips were generated; cannot create summary.")

        print(f"[video] Concatenating {len(clip_paths)} clips -> {output_path}")
        _concat_videos(clip_paths, output_path, crf=crf, preset=preset)
        print(f"[video] Summary saved: {output_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def cleanup_trial_recordings(
    project,
    *,
    keep_top_n: int = 5,
    keep_bottom_n: int = 3,
) -> tuple[int, int]:
    """Delete cached ``trial.mp4`` files, keeping only top/bottom N trials.

    After a long optimization run with ``record_frames=True`` every trial
    produces a ``trial.mp4``. This function deletes the ones that are neither
    in the top ``keep_top_n`` nor the bottom ``keep_bottom_n`` by score,
    freeing disk space while preserving the most informative recordings.

    Args:
        project:      The SofaOptProject whose runtime data to scan.
        keep_top_n:   Keep recordings for this many highest-scoring trials.
        keep_bottom_n: Keep recordings for this many lowest-scoring trials.

    Returns:
        ``(kept, deleted)`` counts.
    """
    ranked = _rank_trials(project)
    if not ranked:
        print("[video] No completed trials found.")
        return 0, 0

    keep_keys: set[tuple[str, str]] = set()
    for r in ranked[:keep_top_n]:
        keep_keys.add((r["gen_name"], r["trial_name"]))
    for r in (ranked[-keep_bottom_n:] if keep_bottom_n else []):
        keep_keys.add((r["gen_name"], r["trial_name"]))

    kept = 0
    deleted = 0
    for rec in ranked:
        mp4 = project.trials_dir / rec["gen_name"] / rec["trial_name"] / "trial.mp4"
        if not mp4.exists():
            continue
        if (rec["gen_name"], rec["trial_name"]) in keep_keys:
            kept += 1
        else:
            mp4.unlink()
            deleted += 1

    print(f"[video] Recordings: kept {kept}, deleted {deleted}.")
    return kept, deleted


def apply_generation_overlays(project, gen: int) -> None:
    """Burn gen/trial/score/params text into each trial.mp4 of a completed generation.

    Called by the orchestrator after run_generation() finishes. FrameRecorder
    streams raw frames at capture time and has no score info, so the overlay
    must be applied here, once trial_state.json has been written.
    Silently skips trials with no trial.mp4 or missing metadata.
    """
    gen_dir = project.trials_dir / f"gen_{gen:04d}"
    if not gen_dir.is_dir():
        return
    for trial_dir in sorted(gen_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        mp4 = trial_dir / "trial.mp4"
        if not mp4.exists():
            continue
        try:
            params: dict = {}
            params_path = trial_dir / "params.json"
            if params_path.exists():
                params = json.loads(params_path.read_text(encoding="utf-8"))
            score: float | None = None
            state_path = trial_dir / "trial_state.json"
            if state_path.exists():
                state_data = json.loads(state_path.read_text(encoding="utf-8"))
                score = state_data.get("final_score")
            gen_name = f"gen_{gen:04d}"
            trial_name = trial_dir.name
            line1 = f"{gen_name} / {trial_name}"
            if score is not None:
                line1 += f"  score: {score:.1f}"
            line2 = "  ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in list(params.items())[:6]
            )
            tmp = mp4.with_name(f"_{mp4.stem}_overlay_tmp.mp4")
            _apply_text_overlay_ffmpeg(mp4, tmp, line1, line2, crf=32, preset="ultrafast")
            tmp.replace(mp4)
        except Exception as exc:
            print(f"[video] Overlay failed for {trial_dir.name}: {exc}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate videos for sofaopt trials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Single trial (default):
    python -m sofaopt.video --project examples/cube_drop/project.py \\
        --trial examples/cube_drop/runtime/trials/gen_0001/trial_01 \\
        --output trial_01.mp4

  Batch (top/bottom N trials):
    python -m sofaopt.video --project examples/cube_drop/project.py \\
        --batch --top 5 --bottom 3 --output-dir runtime/videos \\
        --crf 30 --frame-skip 2

  Summary reel (clips concatenated):
    python -m sofaopt.video --project examples/cube_drop/project.py \\
        --summary --top 3 --bottom 2 --clip-steps 100 \\
        --output runtime/summary.mp4
""",
    )
    parser.add_argument(
        "--project", required=True, type=Path,
        help="Path to project.py (must define PROJECT = SofaOptProject(...))",
    )
    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch",   action="store_true", help="Batch mode: generate for top/bottom N trials")
    mode.add_argument("--summary", action="store_true", help="Summary mode: one highlight-reel MP4")
    mode.add_argument("--cleanup", action="store_true", help="Delete cached trial.mp4s, keeping top/bottom N")

    # Single-trial args
    parser.add_argument("--trial",  type=Path, default=None, help="Trial directory (single-trial mode)")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path")

    # Batch/summary args
    parser.add_argument("--output-dir",  type=Path, default=None, help="Output directory (batch mode)")
    parser.add_argument("--top",         type=int,  default=5,    help="Number of best trials (batch/summary)")
    parser.add_argument("--bottom",      type=int,  default=3,    help="Number of worst trials (batch/summary)")
    parser.add_argument("--score-above", type=float, default=None, help="Include trials scoring above this (batch)")
    parser.add_argument("--score-below", type=float, default=None, help="Include trials scoring below this (batch)")
    parser.add_argument("--clip-steps",  type=int,  default=150,  help="Max sim steps per clip (summary)")

    # Common render args
    parser.add_argument("--test",       default=None, dest="test_name", help="Test name to render")
    parser.add_argument("--width",      type=int,   default=800)
    parser.add_argument("--height",     type=int,   default=600)
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--max-steps",  type=int,   default=None, help="Stop after N sim steps (single trial)")
    parser.add_argument("--crf",        type=int,   default=28,   help="ffmpeg CRF quality (lower=better, default 28)")
    parser.add_argument("--preset",     default="fast",           help="ffmpeg preset (default fast)")
    parser.add_argument("--frame-skip", type=int,   default=16,   help="Capture every Nth frame (default 16)")
    parser.add_argument("--text-overlay", action="store_true",    help="Burn trial ID, score and params into frames")

    args = parser.parse_args()
    project = _load_project(args.project)

    _common = dict(
        test_name=args.test_name,
        width=args.width,
        height=args.height,
        fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        frame_skip=args.frame_skip,
        text_overlay=args.text_overlay,
    )

    if args.cleanup:
        cleanup_trial_recordings(project, keep_top_n=args.top, keep_bottom_n=args.bottom)
    elif args.batch:
        if args.output_dir is None:
            parser.error("--batch requires --output-dir")
        generate_selected_videos(
            project,
            args.output_dir,
            top_n=args.top,
            bottom_n=args.bottom,
            score_above=args.score_above,
            score_below=args.score_below,
            **_common,
        )
    elif args.summary:
        if args.output is None:
            parser.error("--summary requires --output")
        generate_summary_video(
            project,
            args.output,
            top_n=args.top,
            bottom_n=args.bottom,
            clip_steps=args.clip_steps,
            **_common,
        )
    else:
        if args.trial is None or args.output is None:
            parser.error("Single-trial mode requires --trial and --output")
        generate_trial_video(
            project,
            args.trial,
            args.output,
            max_steps=args.max_steps,
            **_common,
        )
