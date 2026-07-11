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

# A prepare hook receives the sampled params and the per-trial working directory,
# does whatever project-specific preparation is needed (e.g. write a config and
# run a geometry generator), and returns a TrialPrep describing extra scene env,
# files to clean up afterward, and an optional preview image. It may raise to
# mark the whole trial as a hard failure (e.g. invalid geometry). Defined as a
# string forward-ref because TrialPrep is declared below.
PrepareHook = Callable[[Mapping[str, Any], Path], "TrialPrep"]

# A constrain hook receives the freshly sampled params and returns the values
# actually used (e.g. enforce a cross-parameter relationship). The values
# recorded by the optimizer are unchanged; only the used values differ.
ConstrainHook = Callable[[dict], dict]


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
        relaunchable: If True, the scene may run an *iterative probe* across
            several short ``runSofa`` launches, carrying state between them. The
            scene drives this with ``trial.relaunch(carry={...})`` (go again) vs
            ``trial.write_score(...)`` (done), and reads carried state back with
            ``trial.load_carry()``. Relaunches are capped by the project's
            ``max_run_relaunches`` (which must be > 0). Leave False for ordinary
            one-shot scenes.
        score_aggregation: How ``run_count`` repeats are combined into this
            test's score: ``"mean"`` (default), ``"median"``, ``"sum"``, or
            ``"exponential_coverage"`` (sum × 1.5 per additional positive
            repeat — rewards covering many scenarios).
        run_count_min: Adaptive re-evaluation (racing). ``None`` (default)
            runs the fixed ``run_count`` repeats. Set to ``1 <= m <=
            run_count`` to launch only ``m`` repeats per trial and add the
            rest one at a time — only while the trial's 95% confidence
            interval still overlaps the study's best score. Candidates that
            provably cannot beat the incumbent skip their remaining repeats;
            contenders get the full ``run_count``. Requires
            ``score_aggregation="mean"`` (repeats must be noise samples of
            one scenario, not different scenarios). Ignored for
            multi-objective runs (no scalar incumbent to race against).
        default_selected: Whether the dashboard pre-selects this test.
    """

    name: str
    scene_file: Path
    label: str = ""
    description: str = ""
    run_count: int = 1
    max_score: float = 1.0
    weight: float = 1.0
    direction: Literal["maximize", "minimize"] = "maximize"
    gated: bool = False
    relaunchable: bool = False
    score_aggregation: str = "mean"
    run_count_min: int | None = None
    default_selected: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_file", Path(self.scene_file).resolve())
        if not self.label:
            object.__setattr__(self, "label", self.name)
        if self.run_count_min is not None:
            if self.score_aggregation != "mean":
                raise ValueError(
                    f"Test '{self.name}': run_count_min (racing) requires "
                    f"score_aggregation='mean' — with "
                    f"'{self.score_aggregation}' the repeats are distinct "
                    "scenarios, and skipping some would change what the "
                    "score measures."
                )
            if not 1 <= self.run_count_min <= self.run_count:
                raise ValueError(
                    f"Test '{self.name}': run_count_min must be in "
                    f"[1, run_count={self.run_count}], got {self.run_count_min}."
                )

    @property
    def display_label(self) -> str:
        if self.description:
            return f"{self.label} — {self.description}"
        return self.label


@dataclass
class TrialPrep:
    """What a :data:`PrepareHook` returns for one trial.

    Args:
        env: Extra environment variables injected into the scene subprocess
            (e.g. ``{"OPT_MESH": "/.../trial.stl"}``). Values are stringified.
        cleanup: Files to delete once the trial's runs have all finished
            (e.g. the per-trial mesh).
        preview_image: Optional image to show in the dashboard for this trial.
            Either a ready PNG, or an ``.stl`` to be rendered (needs the
            ``preview`` extra).
    """

    env: dict[str, str] = field(default_factory=dict)
    cleanup: list[Path] = field(default_factory=list)
    preview_image: Path | None = None


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
    runsofa_exe: Path | None = None
    """Path to the runSofa executable. Required when runner="runsofa" (the
    default); not needed when runner="python"."""
    sofa_plugins: Sequence[str] = ("SofaPython3",)
    sofa_env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for scene subprocesses (e.g. ``SOFA_ROOT``,
    ``PYTHONPATH`` so the scene can import your modules and sofaopt). Merged
    over a copy of the current environment."""
    gui_mode: str = "batch"
    """``"batch"`` for headless optimization; ``"imgui"``/``"glfw"`` to watch
    (interactive GUI names vary by SOFA build — check ``runSofa --help``)."""
    runner: Literal["runsofa", "python"] = "runsofa"
    """``"runsofa"`` (default) or ``"python"`` to launch an in-process Python
    worker instead. The Python runner imports Sofa directly, giving the scene
    access to ``Sofa.Core.Node``, ``Sofa.Simulation.animate()``, etc. while
    keeping full subprocess isolation."""
    float_step: float | None = None
    """Optional quantization step for float parameter sampling (None = continuous)."""

    # --- optional per-trial preparation (e.g. geometry generation) ---------
    prepare_trial: PrepareHook | None = None
    constrain_params: ConstrainHook | None = None
    """Optional: adjust sampled params before they are used/prepared (e.g.
    enforce a cross-parameter relationship). Does not change what the optimizer
    records — only the values written to params.json and passed to the hook."""
    on_generation_end: Callable[[int, list], None] | None = None
    """Optional: called after each generation finishes with
    ``(gen_index, trial_state_paths)``. Use for cross-generation carryover
    (e.g. seeding the next generation from this one's results)."""

    # --- optimizer settings (sane defaults) -------------------------------
    n_parallel: int = 5
    n_generations: int = 100
    cmaes_sigma0: float = 1.0
    cmaes_startup_trials: int | None = None
    """Number of space-filling startup trials before the model-based sampler
    takes over. ``None`` (default) auto-sizes from the number of searched
    parameters — see :meth:`resolve_startup_trials` — so adding/freezing
    parameters rescales the exploration phase automatically."""
    sampler: Literal["cmaes", "tpe", "random", "gp"] = "cmaes"
    """Optuna sampler: ``"cmaes"`` (default), ``"tpe"`` (Bayesian TPE),
    ``"random"``, or ``"gp"`` (Gaussian-process Bayesian optimization, the
    sample-efficient choice for expensive evaluations in <20-D).
    ``cmaes_sigma0`` / ``cmaes_startup_trials`` are only used when
    ``sampler="cmaes"``; ``cmaes_startup_trials`` also seeds ``"gp"`` startup."""
    cmaes_with_margin: bool = False
    """Use CMA-ES *with Margin* (Hamano et al., GECCO 2022) — keeps
    low-cardinality integer parameters from stagnating under naïve
    discretization. Only applies when ``sampler="cmaes"``."""
    seed_sampler: Literal["random", "sobol"] = "random"
    """Initial-design sampler used for the startup/independent phase of
    ``"cmaes"`` and ``"gp"``. ``"sobol"`` gives a space-filling Sobol' (QMC)
    design that covers parameter interactions evenly before the model-based
    phase begins; ``"random"`` (default) preserves prior behavior."""
    seed_sampler_seed: int = 1234
    """Scramble seed for the Sobol' startup design. Change it to get an
    independent (but equally balanced) exploration — e.g. for a validation
    run that should not revisit the previous run's startup points."""
    multi_objective: bool = False
    """When True each :class:`TestSpec` becomes a separate Pareto objective and
    NSGA-II is used. Set ``TestSpec.direction`` per test to ``"maximize"`` or
    ``"minimize"``. Gating and score weighting are disabled in this mode."""
    dedup_trials: bool = False
    """Skip SOFA for a parameter vector that already completed: the recorded
    score is reused (told to Optuna immediately, trial marked ``cached``).
    Only safe when the objective is DETERMINISTIC — projects that average
    noise over run repeats must keep this off. A converged CMA-ES endgame
    otherwise re-simulates one lattice point for whole generations.
    Ignored for multi-objective studies."""
    stall_generations: int = 0
    """Stop the run early after this many consecutive generations without any
    improvement of the best score (0 = run all ``n_generations``). Post-run
    steps (summary video, report) still execute. Ignored for multi-objective
    studies. With ``cmaes_restarts > 0`` a stall triggers an IPOP restart
    instead of stopping, until the restart budget is spent."""
    cmaes_restarts: int = 0
    """Maximum number of IPOP-style CMA-ES restarts (Auger & Hansen 2005).
    0 (default) keeps the historical behavior: a stall *stops* the run. With
    N > 0 the first N stalls each restart CMA-ES instead — a fresh optimizer
    with population size multiplied by ``cmaes_inc_popsize``, the initial
    ``cmaes_sigma0`` and a uniform-random start point — the standard answer
    to multimodal landscapes. Needs ``stall_generations > 0`` (the trigger);
    only acts when ``sampler="cmaes"`` and single-objective. The run still
    ends at ``n_generations`` regardless of restarts."""
    cmaes_inc_popsize: int = 2
    """Population-size multiplier applied at each IPOP restart (default 2).
    The internal CMA population grows past ``n_parallel``, so one CMA update
    then spans several sofaopt generations — that is expected and fine."""
    hard_fail_score: float = -3.0
    max_active_sofa_procs: int = 12
    max_run_relaunches: int = 0
    sofa_realtime_timeout: float = 200.0
    prepare_timeout: float = 60.0

    # --- dashboard wiring (optional) --------------------------------------
    run_script: Path | None = None
    """Script the dashboard's Run button executes to start a headless
    optimization (typically a one-liner calling ``run_optimization(PROJECT)``).
    If None, the dashboard runs read-only (no Run/Stop)."""
    run_python_exe: Path | None = None
    """Interpreter the dashboard uses to run ``run_script``. Defaults to the
    interpreter serving the dashboard (``sys.executable``) — set it when the
    dashboard may run under a foreign Python whose packages differ from the
    ones the optimization needs (e.g. a SOFA build's bundled Python)."""
    config_file: Path | None = None
    """Optional text/JSON config file to expose in the dashboard's Config tab.
    If None, the Config tab is hidden."""
    title: str = ""
    """Dashboard title. Defaults to ``name`` when empty."""

    # --- in-run frame recording (python runner only) ----------------------
    record_frames: bool = False
    """When True and runner=="python", capture frames during each trial run and
    write a fragmented MP4 to ``trial_dir/trial.mp4``. The video is valid even
    when the runner is killed by ScoreWriter mid-simulation. Use
    ``cleanup_trial_recordings()`` from ``sofaopt.video`` to keep only the
    top/bottom N videos after the run."""
    record_frame_skip: int = 16
    """Capture every Nth simulation step (default 16)."""
    record_frame_size: tuple = (640, 480)
    """(width, height) of the captured video frames."""
    record_keep_top_n: int = 15
    """After the run, auto-cleanup keeps recordings for this many best trials."""
    record_keep_bottom_n: int = 5
    """After the run, auto-cleanup keeps recordings for this many worst trials."""
    record_prune_every_n: int = 20
    """Prune excess trial recordings every N completed trials during the run (0 = only at end).
    With n_parallel=4 and the default of 20, the first prune fires after trial 20 (gen 5),
    the second after trial 40 (gen 10), etc. The first prune rarely deletes anything since
    keep_top_n + keep_bottom_n == 20 by default."""
    record_summary_top_n: int = 5
    """Number of highest-scoring trials to include in the summary video."""
    record_summary_bottom_n: int = 3
    """Number of lowest-scoring trials to include in the summary video."""

    # --- optional shape-opt extras ----------------------------------------
    failed_preview_image: Path | None = None
    """Placeholder image shown in the dashboard for trials whose prepare hook
    failed. Only relevant to projects that render per-trial previews."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_dir", Path(self.work_dir).resolve())
        if self.runsofa_exe is not None:
            object.__setattr__(self, "runsofa_exe", Path(self.runsofa_exe))
        if self.sampler == "cmaes" and not self.multi_objective and self.n_parallel < 4:
            raise ValueError("n_parallel must be >= 4 for CMA-ES to remain valid.")
        if not self.params:
            raise ValueError("project.params is empty — nothing to optimize.")
        if not self.tests:
            raise ValueError("project.tests is empty — nothing to evaluate.")
        if self.cmaes_restarts < 0:
            raise ValueError("cmaes_restarts must be >= 0.")
        if self.cmaes_inc_popsize < 1:
            raise ValueError("cmaes_inc_popsize must be >= 1.")
        if self.cmaes_restarts > 0 and self.stall_generations <= 0:
            raise ValueError(
                "cmaes_restarts > 0 needs stall_generations > 0 — the stall "
                "tracker is what triggers a restart."
            )
        if self.multi_objective and len(self.tests) < 2:
            raise ValueError("multi_objective=True requires at least 2 tests.")
        if self.multi_objective and any(t.gated for t in self.tests):
            import warnings
            warnings.warn(
                "multi_objective=True: gated tests are disabled "
                "(gating is not supported in Pareto mode).",
                stacklevel=2,
            )

    # --- derived runtime paths --------------------------------------------
    def resolve_startup_trials(self) -> int:
        """Startup design size: the explicit value, or auto from dimensionality.

        Auto = the power of two nearest (in log2) to ``k*d``, where ``d`` is the
        number of searched (non-frozen) params and ``k`` is 5 for CMA-ES (only
        needs a coverage map) or 10 for GP-BO (must fit a surrogate; the
        classic 10*d rule). Powers of two because the Sobol' startup design is
        exactly balanced there. Examples (cmaes): d=6 -> 32, d=9 -> 32,
        d=12 -> 64; (gp): d=6 -> 64.
        """
        if self.cmaes_startup_trials is not None:
            return self.cmaes_startup_trials
        import math

        d = max(1, sum(1 for p in self.params if not p.is_frozen))
        k = 10 if self.sampler == "gp" else 5
        return 2 ** max(3, round(math.log2(k * d)))

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


def load_project_file(project_path: Path | str, attr: str = "PROJECT") -> SofaOptProject:
    """Load a :class:`SofaOptProject` from a user ``project.py`` file.

    The file must assign ``PROJECT = SofaOptProject(...)`` (or ``attr``).
    Used by the video CLI and subprocess entry points.
    """
    import importlib.util

    project_path = Path(project_path)
    spec = importlib.util.spec_from_file_location("_sofaopt_project", project_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load project file: {project_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, attr):
        raise AttributeError(
            f"{project_path} does not define a {attr} variable. "
            f"Make sure the file assigns: {attr} = SofaOptProject(...)"
        )
    return getattr(mod, attr)


def _spec_to_jsonable(spec: Any) -> dict[str, Any]:
    from dataclasses import fields as _fields

    return {
        f.name: (str(v) if isinstance(v, Path) else v)
        for f in _fields(spec)
        for v in [getattr(spec, f.name)]
    }


def project_to_jsonable(project: SofaOptProject) -> dict[str, Any]:
    """JSON-safe dict for handing a project to a subprocess.

    Hooks (``prepare_trial``, ``constrain_params``, ``on_generation_end``)
    cannot cross a process boundary and are dropped. This is the sanctioned
    replacement for pickling the project (§9: no pickle for IPC).
    """
    from dataclasses import fields as _fields

    out: dict[str, Any] = {}
    for f in _fields(project):
        v = getattr(project, f.name)
        if v is not None and callable(v):
            continue  # hook — not serializable, receiver runs hook-free
        if isinstance(v, Path):
            v = str(v)
        elif f.name in ("params", "tests"):
            v = [_spec_to_jsonable(s) for s in v]
        elif isinstance(v, Mapping):
            v = {k: str(val) for k, val in v.items()}
        elif isinstance(v, tuple):
            v = list(v)
        out[f.name] = v
    return out


def project_from_jsonable(data: Mapping[str, Any]) -> SofaOptProject:
    """Rebuild a (hook-free) :class:`SofaOptProject` from :func:`project_to_jsonable`."""
    kwargs = dict(data)
    kwargs["params"] = [ParamSpec(**p) for p in kwargs.get("params", [])]
    kwargs["tests"] = [TestSpec(**t) for t in kwargs.get("tests", [])]
    for key in ("work_dir", "runsofa_exe", "run_script", "config_file", "failed_preview_image"):
        if kwargs.get(key):
            kwargs[key] = Path(kwargs[key])
    if "record_frame_size" in kwargs:
        kwargs["record_frame_size"] = tuple(kwargs["record_frame_size"])
    return SofaOptProject(**kwargs)


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
