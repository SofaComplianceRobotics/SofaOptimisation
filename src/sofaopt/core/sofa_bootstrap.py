"""Bootstrap helpers for any process that imports SOFA's Python bindings.

Platform-specific setup isolated in one place: the scene
runner, the offline video renderer, and the dashboard's summary-gen child all
face the same three problems and must solve them identically:

1. Windows Python 3.8+ does not search PATH when a ``.pyd`` extension module
   loads its DLL dependencies — SOFA's ``bin`` dir must be registered with
   ``os.add_dll_directory`` *before* the first ``import SofaRuntime``.
2. ``SOFA_ROOT`` may be unset (e.g. the dashboard was launched from an env
   that only has SOFA's site-packages on ``PYTHONPATH``); it can be derived.
3. Windows consoles default to cp1252 — non-ASCII output from SOFA/pygame
   must not crash the process.

stdlib-only: safe to import inside the SofaPython3 interpreter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def reconfigure_streams_utf8() -> None:
    """Make stdout/stderr survive non-ASCII output on cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # non-reconfigurable stream (e.g. pytest capture) — fine


def derive_sofa_root() -> str:
    """Best-effort SOFA build root: env vars first, then a PYTHONPATH scan.

    The scan covers the dashboard-child case: SOFA's site-packages is on
    PYTHONPATH but neither SOFA_ROOT nor SOFAPYTHON3_ROOT is exported.
    """
    root = os.environ.get("SOFA_ROOT", "") or os.environ.get("SOFAPYTHON3_ROOT", "")
    if root:
        return root

    for raw_entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        entry = raw_entry.strip()
        if not entry:
            continue
        p = Path(entry)
        try:
            has_sofa = (p / "Sofa").is_dir() or bool(list(p.glob("Sofa*.pyd"))[:1])
        except (OSError, PermissionError):
            has_sofa = False
        if not has_sofa:
            continue
        # site-packages sits at <root>/lib/python3/site-packages
        for up in (p.parent.parent.parent, p.parent.parent):
            if (up / "bin").is_dir() or (up / "bin" / "Release").is_dir():
                return str(up)
    return ""


def register_sofa_dll_dirs(sofa_root: str | Path | None = None) -> None:
    """Register SOFA's DLL directories (Windows, Python 3.8+).

    Must run before the first ``import SofaRuntime`` — SOFA's own
    ``SofaRuntime/__init__.py`` does this too, but only after the ``.pyd``
    has already loaded, which is too late.
    """
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    root = str(sofa_root or derive_sofa_root())
    if not root:
        return
    for candidate in (Path(root) / "bin" / "Release", Path(root) / "bin"):
        if candidate.is_dir():
            os.add_dll_directory(str(candidate))


def bootstrap_sofa_env() -> str:
    """Full bootstrap for a fresh subprocess that must import Sofa.

    Derives the SOFA root, registers DLL dirs, exports ``SOFA_ROOT`` (SOFA
    needs it to find its INI files and plugins), and puts SOFA's
    site-packages on ``sys.path`` if ``Sofa`` is not yet importable.
    Returns the derived root ('' when none was found).
    """
    root = derive_sofa_root()
    register_sofa_dll_dirs(root or None)
    if root and not os.environ.get("SOFA_ROOT"):
        os.environ["SOFA_ROOT"] = root

    import importlib.util

    if root and importlib.util.find_spec("Sofa") is None:
        for sp in (
            Path(root) / "lib" / "python3" / "site-packages",
            Path(root)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages",
        ):
            if sp.is_dir():
                sys.path.insert(0, str(sp))
                break
    return root
