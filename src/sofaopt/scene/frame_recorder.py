"""In-run OpenGL frame recorder for the sofaopt Python runner.

When OPT_RECORD_FRAMES=1 is set in the environment, the runner initialises a
pygame/OpenGL window (hidden on Windows), renders each Nth sim step and pipes
raw RGB bytes to an ffmpeg subprocess producing a fragmented MP4.

Fragmented MP4 (frag_keyframe+empty_moov) stays valid even when the runner
is SIGKILL'd mid-simulation by ScoreWriter after writing the score — only the
in-flight fragment is lost; all prior fragments decode correctly.

This module owns the camera injection and GL render used by BOTH capture
paths: the in-run recorder here and the offline re-render in
``sofaopt.video.single`` import the same ``inject_camera_and_lights`` /
``render_frame_raw``, so recordings and re-renders are framed identically.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared scene-graph helpers (used by runner.py and video.py)
# ---------------------------------------------------------------------------

def inject_camera_and_lights(root) -> None:
    """Add a default camera and lights to *root* if the scene omits them.

    runSofa injects these automatically; the Python path does not.
    Must be called after createScene() and before initRoot().
    """
    try:
        existing = root.getObject("camera")
    except Exception:
        existing = None

    if existing is None:
        try:
            root.addObject(
                "InteractiveCamera",
                name="camera",
                position=[0.0, 40.0, 200.0],
                lookAt=[0.0, 30.0, 0.0],
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


def render_frame_raw(root, sofa_gl, sofa_sim, width: int, height: int) -> bytes:
    """Render the current scene and return raw bottom-up RGB bytes (no flip).

    Returns empty bytes on any error so callers can skip silently.
    """
    try:
        from OpenGL.GL import (
            GL_AMBIENT, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
            GL_DIFFUSE, GL_LIGHT0, GL_LIGHTING, GL_MODELVIEW, GL_POSITION,
            GL_PROJECTION, GL_RGB, GL_UNSIGNED_BYTE,
            glClear, glClearColor, glEnable, glLightfv, glLoadIdentity, glMatrixMode,
            glReadPixels, glViewport,
        )
        from OpenGL.GLU import gluLookAt, gluPerspective

        sofa_sim.updateVisual(root)

        glClearColor(0.18, 0.18, 0.22, 1.0)   # dark blue-grey background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, width, height)

        # Fallback light — SOFA's PositionalLight/DirectionalLight configure GL
        # lights in their drawVisual; this guarantees at least one source.
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 1.0, 1.0, 0.0])
        glEnable(GL_LIGHT0)

        # Sofa.SofaGL.draw() does not configure projection/view matrices.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        try:
            cam  = root.getObject("camera")
            fov   = float(cam.fieldOfView.value)
            znear = float(cam.zNear.value)
            zfar  = float(cam.zFar.value)
        except Exception:
            fov, znear, zfar = 45.0, 0.1, 10000.0
        gluPerspective(fov, width / height, znear, zfar)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        try:
            cam = root.getObject("camera")
            pos = list(cam.position.value)
            lat = list(cam.lookAt.value)
            gluLookAt(pos[0], pos[1], pos[2], lat[0], lat[1], lat[2], 0.0, 0.0, 1.0)
        except Exception:
            gluLookAt(0.0, 40.0, 200.0, 0.0, 30.0, 0.0, 0.0, 0.0, 1.0)

        sofa_gl.draw(root)

        return bytes(glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE))
    except Exception:
        return b""


# ---------------------------------------------------------------------------
# FrameRecorder
# ---------------------------------------------------------------------------

class FrameRecorder:
    """Captures frames during a live SOFA simulation and streams to ffmpeg.

    Frames are piped as raw RGB to an ffmpeg subprocess that writes a
    fragmented MP4.  The format is valid even if the process is SIGKILL'd,
    so this works correctly in the optimizing path where ScoreWriter calls
    ``os.kill(9)`` immediately after writing the score.

    Usage in runner.py::

        recorder = FrameRecorder.from_env()
        if recorder:
            inject_camera_and_lights(root)   # before initRoot
        Sofa.Simulation.initRoot(root)
        root.animate = True
        if recorder:
            recorder.start(root)

        step = 0
        while root.animate.value:
            if recorder:
                recorder.maybe_capture(step, root)
            Sofa.Simulation.animate(root, root.dt.value)
            step += 1

        if recorder:
            recorder.close()   # only reached on non-optimizing (hand-launch) path
    """

    def __init__(
        self,
        output_path: Path,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        frame_skip: int = 4,
        crf: int = 23,
    ) -> None:
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_skip = frame_skip
        self.crf = crf
        self._ffmpeg: subprocess.Popen | None = None
        self._sofa_gl = None
        self._sofa_sim = None
        self._pygame = None

    @classmethod
    def from_env(cls) -> "FrameRecorder | None":
        """Return a recorder configured from environment variables, or None."""
        from sofaopt.core import envkeys

        if os.environ.get(envkeys.RECORD_FRAMES) != "1":
            return None
        output = os.environ.get(envkeys.RECORD_OUTPUT, "")
        if not output:
            return None
        return cls(
            Path(output),
            width=int(os.environ.get(envkeys.RECORD_WIDTH, "640")),
            height=int(os.environ.get(envkeys.RECORD_HEIGHT, "480")),
            fps=int(os.environ.get(envkeys.RECORD_FPS, "30")),
            frame_skip=int(os.environ.get(envkeys.RECORD_FRAME_SKIP, "16")),
        )

    def start(self, root) -> None:
        """Initialise pygame/OpenGL and start the ffmpeg pipe.

        Call this *after* Sofa.Simulation.initRoot(root).
        """
        try:
            import pygame
        except ImportError:
            print("[recorder] pygame not installed -- recording disabled.", flush=True)
            return

        try:
            import Sofa.SofaGL as sofa_gl
            import Sofa.Simulation as sofa_sim
        except ImportError:
            print("[recorder] SOFA not importable -- recording disabled.", flush=True)
            return

        pygame.display.init()
        # Create the GL window HIDDEN so it never flashes on the desktop. Rendering only
        # ever touches the BACK buffer (render_frame_raw does glReadPixels with NO buffer
        # swap), so the window surface is never shown — hidden is fully safe. Previously the
        # window was created shown then hidden via ShowWindow, flashing once per trial; that
        # became a rapid strobe in runs with many <1s trials (e.g. length-cap pre-sim rejects).
        pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME | getattr(pygame, "HIDDEN", 0),
        )
        pygame.display.set_caption("sofaopt recorder")

        # Belt-and-suspenders: also hide via the Win32 handle (no-op if already HIDDEN, and
        # the fallback for any pygame build lacking the HIDDEN flag).
        if os.name == "nt":
            try:
                import ctypes
                hwnd = pygame.display.get_wm_info().get("window", 0)
                if hwnd:
                    ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
            except Exception:
                pass

        sofa_gl.glewInit()
        sofa_sim.initVisual(root)
        sofa_sim.initTextures(root)

        self._pygame = pygame
        self._sofa_gl = sofa_gl
        self._sofa_sim = sofa_sim

        # Pipe raw RGB → ffmpeg → fragmented MP4.
        # vflip: OpenGL pixel origin is bottom-left; ffmpeg expects top-left.
        # frag_keyframe+empty_moov: each keyframe is a self-contained fragment,
        # so the file is valid even if ffmpeg is killed before writing moov.
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-vf", "vflip",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            str(self.output_path),
        ]
        # CREATE_NO_WINDOW prevents ffmpeg from opening a visible console on Windows.
        _flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            self._ffmpeg = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_flags,
            )
            print(
                f"[recorder] {self.width}x{self.height} "
                f"every {self.frame_skip} steps -> {self.output_path}",
                flush=True,
            )
        except FileNotFoundError:
            print("[recorder] ffmpeg not found -- recording disabled.", flush=True)
            if self._pygame:
                self._pygame.quit()
            self._ffmpeg = None

    def maybe_capture(self, step: int, root) -> None:
        """Render and pipe a frame if this is a capture step.

        Call this BEFORE Sofa.Simulation.animate() so the last frame before
        an os.kill(9) is still captured.
        """
        if self._ffmpeg is None or self._sofa_gl is None:
            return
        if step % self.frame_skip != 0:
            return

        # Drain pygame event queue to keep the window responsive.
        if self._pygame:
            try:
                self._pygame.event.pump()
            except Exception:
                pass

        raw = render_frame_raw(root, self._sofa_gl, self._sofa_sim, self.width, self.height)
        if not raw:
            return

        if self._ffmpeg.stdin:
            try:
                self._ffmpeg.stdin.write(raw)
                self._ffmpeg.stdin.flush()
            except (BrokenPipeError, OSError):
                self._ffmpeg = None  # ffmpeg died; stop trying

        if self._pygame:
            try:
                self._pygame.display.flip()
            except Exception:
                pass

    def close(self) -> None:
        """Flush and wait for ffmpeg. Only called on clean (non-SIGKILL) exits."""
        if self._ffmpeg:
            if self._ffmpeg.stdin:
                try:
                    self._ffmpeg.stdin.close()
                except Exception:
                    pass
            try:
                self._ffmpeg.wait(timeout=30)
            except Exception:
                pass
            self._ffmpeg = None
        if self._pygame:
            try:
                self._pygame.quit()
            except Exception:
                pass
            self._pygame = None
