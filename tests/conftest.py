"""Shared pytest setup: make ``import sofaopt`` work from a plain checkout.

When the package is installed (``pip install -e .``) this is a no-op; when the
repo is used without installing, we put ``src/`` on sys.path so the tests can
still run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_benchmark_functions():
    """Load ``examples/landscape/benchmark_functions.py`` by path (examples/ is
    not an installed package). Registered in sys.modules so its ``@dataclass``
    field-type resolution works under importlib loading."""
    if "benchmark_functions" in sys.modules:
        return sys.modules["benchmark_functions"]
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "landscape" / "benchmark_functions.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_functions", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_functions"] = mod
    spec.loader.exec_module(mod)
    return mod
