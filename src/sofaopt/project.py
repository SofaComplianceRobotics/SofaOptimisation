"""The sofaopt adapter contract.

A *project* tells the framework three things:

  1. **What to tune**  — a list of :class:`ParamSpec` (name, type, range, default).
  2. **What to run**   — a list of :class:`TestSpec`, each pointing at a SOFA
     ``scene.py`` plus how to score and weight it.
  3. **How to reach SOFA** — the path to ``runSofa`` and any environment a
     scene process needs (``SOFA_ROOT``, ``PYTHONPATH``, plugins, ...).

Everything else (CMA-ES, parallel scheduling, scoring aggregation, gating,
live progress, the dashboard) is provided by the framework and never needs to
know anything about your robot, your geometry, or your SOFA build.

The optimization loop per trial is:

    sample params ─▶ (optional) prepare hook ─▶ launch scene.py via runSofa
                                                        │
                       collect score from trial_state.json ◀─ scene writes score

A *shape*-optimization project supplies a :attr:`SofaOptProject.prepare_trial`
hook that turns the sampled params into a mesh; a project that tunes
stiffness / mass / controller gains supplies no hook at all — the framework
writes the sampled params to ``params.json`` in the trial dir and the scene
reads them directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

ParamType = Literal["float", "int", "bool"]

# A prepare hook receives the sampled params and the per-trial working
# directory, does whatever project-specific preparation is needed (e.g. write
# config + run a geometry generator), and returns a mapping of EXTRA environment
# variables to inject into the scene subprocess (e.g. {"OPT_MESH": "/.../t.stl"}).
# It may raise to mark the whole trial as a hard failure (e.g. invalid geometry).
PrepareHook = Callable[[Mapping[str, Any], Path], Mapping[str, str]]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter.

    Args:
        name: Parameter key. Passed to the scene in ``params.json`` and used as
            the Optuna distribution name, so it must be unique and stable.
        type: ``"float"``, ``"int"`` or ``"bool"``.
        low: Lower bound (inclusive). Ignored for ``bool``.
        high: Upper bound (inclusive). Ignored for ``bool``.
        default: Value used to seed CMA-ES (``x0``) and used verbatim when the
            parameter is *frozen*.

    A float/int parameter with ``low == high`` is **frozen**: it is reported to
    consumers (so the scene still receives it) but is held at ``default``
    instead of being searched. This lets a project list every parameter in one
    place and toggle which are active by widening/narrowing the range.
    """

    name: str
    type: ParamType
    low: float = 0.0
    high: float = 0.0
    default: Any = 0.0

    @property
    def is_frozen(self) -> bool:
        """True when this parameter is fixed at ``default`` rather than searched."""
        if self.type == "bool":
            return False
        return self.low == self.high

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form used internally by the sampler."""
        return {
            "name": self.name,
            "type": self.type,
            "min": self.low,
            "max": self.high,
            "default": self.default,
        }


@dataclass(frozen=True)
class TestSpec:
    """One scenario the candidate is evaluated against.

    A trial may run several tests (and several repeats of each); the per-test
    scores are normalized by ``max_score`` and combined using ``weight``.

    Args:
        name: Unique test id (also the run-state key the scene writes under).
        scene_file: Path to the SOFA ``scene.py`` launched via ``runSofa``.
        label: Human-readable name for the dashboard.
        description: One-line description for the dashboard.
        run_count: Repeats per trial (e.g. randomized scenarios averaged together).
        max_score: Raw score that maps to 1.0 after normalization.
        weight: Relative importance when combining tests (need not sum to 1).
        gated: If True, this test is only run once an *ungated* test has scored
            above zero for the trial — used to skip expensive tests on hopeless
            candidates. Pure quality-of-life; safe to leave False.
        score_aggregation: How repeats are combined: ``"mean"``, ``"median"``,
            ``"min"``, ``"max"``, or ``"sum"``.
        default_selected: Whether the dashboard pre-selects this test.
    """

    name: str
    scene_file: Path
    label: str = ""
    description: str = ""
    run_count: int = 1
    max_score: float = 1.0
    weight: float = 1.0
    gated: bool = False
    score_aggregation: str = "mean"
    default_selected: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_file", Path(self.scene_file).resolve())
        if not self.label:
            object.__setattr__(self, "label", self.name)

    @property
    def display_label(self) -> str:
        if self.description:
            return f"{self.label} — {self.description}"
        return self.label


@dataclass(frozen=True)
class SofaOptProject:
    """Everything the framework needs to optimize one project.

    Construct one of these in your project's ``project.py`` and pass it to
    :func:`sofaopt.run_optimization` (headless) or
    :func:`sofaopt.launch_dashboard` (web UI).
    """

    # --- identity & workspace ---------------------------------------------
    name: str
    work_dir: Path
    """Root for runtime artifacts. Trials, the Optuna DB and progress.json all
    live under ``work_dir/runtime``. Created if missing."""

    # --- what to tune & what to run ---------------------------------------
    params: Sequence[ParamSpec]
    tests: Sequence[TestSpec]

    # --- how to reach SOFA (works with ANY build that ships SofaPython3) ---
    runsofa_exe: Path
    sofa_plugins: Sequence[str] = ("SofaPython3",)
    sofa_env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for scene subprocesses (e.g. ``SOFA_ROOT``,
    ``PYTHONPATH`` so the scene can import your modules and sofaopt). Merged
    over a copy of the current environment."""
    gui_mode: str = "batch"
    """``"batch"`` for headless optimization; ``"qt"``/``"qglviewer"`` to watch."""

    # --- optional per-trial preparation (e.g. geometry generation) ---------
    prepare_trial: PrepareHook | None = None

    # --- optimizer settings (sane defaults) -------------------------------
    n_parallel: int = 5
    n_generations: int = 100
    cmaes_sigma0: float = 1.0
    cmaes_startup_trials: int = 50
    hard_fail_score: float = -3.0
    max_active_sofa_procs: int = 12
    max_run_relaunches: int = 0
    sofa_realtime_timeout: float = 200.0
    prepare_timeout: float = 60.0
    stl_delete_delay: float = 30.0

    # --- optional shape-opt extras ----------------------------------------
    failed_preview_image: Path | None = None
    """Placeholder image shown in the dashboard for trials whose prepare hook
    failed. Only relevant to projects that render per-trial previews."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve())
        object.__setattr__(self, "runsofa_exe", Path(self.runsofa_exe))
        if self.n_parallel < 4:
            raise ValueError("n_parallel must be >= 4 for CMA-ES to remain valid.")
        if not self.params:
            raise ValueError("project.params is empty — nothing to optimize.")
        if not self.tests:
            raise ValueError("project.tests is empty — nothing to evaluate.")

    # --- derived runtime paths --------------------------------------------
    @property
    def runtime_dir(self) -> Path:
        return self.work_dir / "runtime"

    @property
    def trials_dir(self) -> Path:
        return self.runtime_dir / "trials"

    @property
    def previews_dir(self) -> Path:
        return self.trials_dir / "previews"

    @property
    def progress_file(self) -> Path:
        return self.trials_dir / "progress.json"

    @property
    def db_path(self) -> Path:
        return self.runtime_dir / "study.db"

    # --- convenience views -------------------------------------------------
    def test(self, name: str) -> TestSpec:
        for t in self.tests:
            if t.name == name:
                return t
        raise KeyError(f"Unknown test '{name}'. Have: {[t.name for t in self.tests]}")

    @property
    def run_plan(self) -> tuple[tuple[str, int, int], ...]:
        """Flattened (test_name, run_index, run_total) schedule for one trial."""
        return tuple(
            (t.name, run_index, t.run_count)
            for t in self.tests
            for run_index in range(1, t.run_count + 1)
        )

    @property
    def gated_test_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tests if t.gated)

    def scene_env(self) -> dict[str, str]:
        """Base environment for a scene subprocess: current env + project env."""
        env = os.environ.copy()
        env.update({k: str(v) for k, v in self.sofa_env.items()})
        return env


def param_specs_from_dataclass(instance: Any) -> list[ParamSpec]:
    """Build :class:`ParamSpec` list from a dataclass with ``opt`` field metadata.

    Convenience for projects that already describe their parameters as a
    dataclass, annotating each tunable field with
    ``metadata={"opt": {"type": "float", "min": x, "max": y}}``. The field's
    current value becomes the default. Fields without ``opt`` metadata are
    skipped.
    """
    from dataclasses import fields as _fields

    specs: list[ParamSpec] = []
    for f in _fields(instance):
        opt = f.metadata.get("opt")
        if opt is None:
            continue
        specs.append(
            ParamSpec(
                name=f.name,
                type=opt["type"],
                low=opt["min"],
                high=opt["max"],
                default=getattr(instance, f.name),
            )
        )
    return specs
