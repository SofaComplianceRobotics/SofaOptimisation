"""Python in-process SOFA runner for sofaopt (launched as a subprocess).

Used when ``SofaOptProject.runner == "python"``. Loads the scene module directly
in-process, runs the SOFA animate loop, and exits when the scene stops itself
(``root.animate = False``) or is killed by the optimizer's timeout watchdog.

The key advantage over ``runSofa`` is that scene code can use the full
``Sofa.Core`` / ``Sofa.Simulation`` Python API — constraint matrices,
``Sofa.Simulation.reset()``, direct node manipulation, etc. — while subprocess
isolation is preserved: a crash in the scene still only kills this worker.

Launch (done automatically by sofa_runner.py when runner == "python"):
    python -m sofaopt.scene.runner <scene.py>
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from sofaopt.core import envkeys
from sofaopt.core.sofa_bootstrap import (
    reconfigure_streams_utf8,
    register_sofa_dll_dirs,
)


def _import_sofa():
    """Import the SOFA bindings, exiting with a diagnosable message when absent."""
    try:
        import SofaRuntime
        import Sofa.Core
        import Sofa.Simulation
    except ImportError as exc:
        raise SystemExit(
            f"[runner] SOFA Python bindings not found. "
            f"Ensure SOFA's site-packages directory is on PYTHONPATH "
            f"and SOFA_ROOT points to the build root. ({exc})"
        ) from exc
    return SofaRuntime, Sofa


def _load_plugins(SofaRuntime) -> None:
    plugins = json.loads(os.environ.get(envkeys.SOFA_PLUGINS, "[]"))
    sofa_root = os.environ.get("SOFA_ROOT", "")
    if sofa_root:
        SofaRuntime.PluginRepository.addFirstPath(str(Path(sofa_root) / "bin"))
    for plugin in plugins:
        try:
            SofaRuntime.importPlugin(plugin)
        except Exception as e:
            print(f"[runner] Warning: could not load plugin {plugin!r}: {e}", flush=True)


def _load_scene_module(scene_path: str):
    spec = importlib.util.spec_from_file_location("_sofaopt_scene", scene_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[runner] Cannot load scene: {scene_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sofaopt_scene"] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "createScene"):
        raise SystemExit(f"[runner] Scene has no createScene() function: {scene_path}")
    return mod


def _setup_recorder(root):
    """FrameRecorder when OPT_RECORD_FRAMES=1, else None.

    Wrapped entirely in try/except: a recording failure must never fail the trial.
    """
    if os.environ.get(envkeys.RECORD_FRAMES) != "1":
        return None
    try:
        from sofaopt.scene.frame_recorder import FrameRecorder, inject_camera_and_lights
        inject_camera_and_lights(root)   # must be before initRoot
        return FrameRecorder.from_env()
    except Exception as exc:
        print(f"[runner] Recording setup failed, disabled: {exc}", flush=True)
        return None


def main(scene_path: str) -> None:
    # Windows console uses cp1252 by default; SOFA and pygame may print non-ASCII.
    reconfigure_streams_utf8()

    # Must run before the first SofaRuntime import (Windows DLL search rules).
    register_sofa_dll_dirs()
    SofaRuntime, Sofa = _import_sofa()
    _load_plugins(SofaRuntime)

    mod = _load_scene_module(scene_path)
    root = Sofa.Core.Node("root")
    mod.createScene(root)

    recorder = _setup_recorder(root)

    Sofa.Simulation.initRoot(root)

    # Sofa.Core.Node starts with animate=False; runSofa sets it True internally
    # before starting its own loop, so we must do the same here.
    root.animate = True

    if recorder:
        try:
            recorder.start(root)   # opens pygame window + ffmpeg pipe
        except Exception as exc:
            print(f"[runner] recorder.start() failed, disabled: {exc}", flush=True)
            recorder = None

    # Capture BEFORE each animate() so the frame preceding any os.kill(9)
    # from ScoreWriter is already in the ffmpeg pipe. Fragmented MP4 means
    # the video is valid even if the process is killed mid-fragment.
    step = 0
    while root.animate.value:
        if recorder:
            recorder.maybe_capture(step, root)
        Sofa.Simulation.animate(root, root.dt.value)
        step += 1

    # Non-optimizing (hand-launched) path: flush and close the recorder cleanly.
    if recorder:
        recorder.close()

    # Skip Python GC / Py_FinalizeEx: SOFA's Python bindings crash during
    # cleanup on Windows.  os._exit() terminates cleanly without running
    # Python destructors (same effect as ScoreWriter's os.kill, but for the
    # non-optimizing / hand-launched path where the kill was not called).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m sofaopt.scene.runner <scene.py>")
    main(sys.argv[1])
