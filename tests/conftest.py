"""Shared pytest setup: make ``import sofaopt`` work from a plain checkout.

When the package is installed (``pip install -e .``) this is a no-op; when the
repo is used without installing, we put ``src/`` on sys.path so the tests can
still run.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
