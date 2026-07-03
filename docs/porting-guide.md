# Porting guide: optimize *your* SOFA project with sofaopt

This guide takes you from an existing SOFA scene to a fully optimized,
dashboard-driven project. You bring the simulation and the score; sofaopt
brings CMA-ES, parallel execution, aggregation, gating, and the web UI.


---

## 0. Mental model

The framework runs one loop:

```
sample params ─▶ (optional) prepare hook ─▶ launch scene.py via runSofa
                                                                      │
        collect score from trial_state.json ◀─ scene writes score  ◀─
```

So you implement **three things**:

1. a **`SofaOptProject`** — what to tune, what to run, how to reach SOFA;
2. one or more **scenes** that read the sampled params and write a score;
3. *(only for shape optimization)* a **prepare hook** that turns params into an
   asset (e.g. a mesh) before the scene launches.

Everything else is provided.

> A complete, runnable example lives in [`examples/cube_drop/`](../examples/cube_drop).
> Read it alongside this guide — it is a small project + scene that exercises the
> whole pipeline (including the prepare hook).

---

## 1. Install

```bash
pip install -e /c/dev/sofaopt[dashboard]   # add ,preview for STL previews
```

sofaopt does **not** install SOFA. It launches whatever `runSofa` you point it
at, so any build with the `SofaPython3` plugin works (emiolabs build, an
official binary release, a source build — all fine).

---

## 2. Make your scene speak the contract

In each scene's `createScene(root)`, read the trial and report a score. This is
the *entire* scene-side API:

```python
from sofaopt.scene import open_trial

def createScene(root):
    trial = open_trial(root)            # params + run metadata from the env
    stiffness = trial.params["stiffness"]
    mesh = trial.env.get("OPT_MESH")    # only if a prepare hook produced one

    # ... build your SOFA graph using those values ...

    root.addObject(MyController(trial=trial))   # see below
    return root
```

A controller (or any code) reports exactly **one** outcome per run:

```python
class MyController(Sofa.Core.Controller):
    def onAnimateEndEvent(self, e):
        # optional: live progress for the dashboard
        trial.write_status({"state": "running",
                            "current_frame": self.step,
                            "total_frames": self.horizon},
                           min_interval=0.2)
        if done:
            trial.write_score(score, reason="held 4.2 s")   # writes + stops
        # or trial.prune("unstable")  to discard the run unscored
```

Key properties:

- `trial.write_score(...)` is **idempotent** and **terminates** the process —
  call it once when the run has produced its result.
- When the scene is opened **outside** the optimizer (e.g. `runSofa scene.py`
  to debug), `trial.params == {}` and `trial.is_optimizing == False`, so guard
  scoring with `if trial.is_optimizing:` and your scene still runs interactively.

### What the scene receives (env, set automatically)

| `trial.` field | from env key | meaning |
|----------------|--------------|---------|
| `params` | `OPT_PARAMS_PATH` (a `params.json`) | this trial's sampled values |
| `run_slot` | `OPT_RUN_SLOT` | which slot to write the score into |
| `test_name` | `OPT_TEST_NAME` | which test is running |
| `test_run_index`/`test_run_total` | `OPT_TEST_RUN_INDEX`/`OPT_TEST_RUN_TOTAL` | repeat i of N for this test |
| `gen`/`trial`/`run` | `OPT_GEN`/`OPT_TRIAL`/`OPT_RUN` | identifiers |
| `trial_state_path` | `OPT_TRIAL_STATE_PATH` | where scores are written; absent ⇒ `is_optimizing == False` |
| `env["..."]` | prepare-hook env | e.g. `OPT_MESH` |

The full key list (including the Python-runner recording keys
`OPT_SOFA_PLUGINS` / `OPT_RECORD_*` and the dashboard override keys) lives in
one place: [`src/sofaopt/core/envkeys.py`](../src/sofaopt/core/envkeys.py).
Always import from there instead of retyping string literals.

---

## 3. Describe the project

Create `project.py` (anywhere — your repo, not sofaopt's):

```python
import os
from pathlib import Path
from sofaopt import SofaOptProject, ParamSpec, TestSpec

HERE = Path(__file__).resolve().parent

PROJECT = SofaOptProject(
    name="my_robot",
    work_dir=HERE,                      # runtime/ (trials, db, progress) goes here
    params=[
        ParamSpec("stiffness", "float", 1e3, 1e6, 1e4), #name, type, min, max, default
        ParamSpec("n_layers",  "int",   1,   6,   3),
        ParamSpec("use_brace", "bool",  default=True),#name, type, default
    ],

    tests=[
        TestSpec("reach", scene_file=HERE / "scenes/reach.py",
                 max_score=100, weight=2, default_selected=True),
        TestSpec("hold",  scene_file=HERE / "scenes/hold.py",
                 max_score=100, weight=1, gated=True),
    ],
    # add more tests with max_score=, weight=, gated=, run_count=, default_selected=

    runsofa_exe=Path(os.environ.get("RUNSOFA_EXE", "runSofa")),
    sofa_env={k: os.environ[k] for k in ("SOFA_ROOT",) if k in os.environ},
    n_parallel=5,
    n_generations=100,
    run_script=HERE / "run.py",         # enables the dashboard's Run button
)
```

### Parameters (`ParamSpec`)

- types: `"float"`, `"int"`, `"bool"`.
- a float/int with `low == high` is **frozen** (reported to the scene but not
  searched) — handy to keep every parameter in one list and toggle which are live.
- already have a dataclass with `metadata={"opt": {...}}` fields? Use
  `sofaopt.param_specs_from_dataclass(instance)`.

### Tests (`TestSpec`)

- `scene_file` is launched via `runSofa`.
- `run_count` — how many times to run this test per trial (e.g. several
  randomized scenarios). `score_aggregation` is how those repeats are combined
  into the test's score: `"mean"` (default), `"median"`, `"sum"`, or
  `"exponential_coverage"` (sum × 1.5 per additional *positive* repeat —
  rewards candidates that succeed across many scenarios rather than excelling
  in one). Both are fields on the `TestSpec`:

  ```python
  TestSpec("reach", scene_file=..., run_count=3, score_aggregation="mean", max_score=100)
  ```

- `max_score` normalizes the test to `[0,1]`; `weight` combines tests.
- `gated=True`: only run this test once an *ungated* test has
  scored above zero for the candidate.

### How scores combine (exact order)

Per trial the pipeline is, in this order and nowhere else:

1. each test's repeat scores → one per-test aggregate via `score_aggregation`;
2. per-test aggregate → normalized by `max_score`, clamped at 1.0;
3. normalized tests → combined by `weight`, renormalized over the tests
   actually counted (a gated test that never unlocked is excluded);
4. the result (0–100) is the study objective and is recorded in
   `trial_state.json` (`final_score`) — the dashboard/videos read it verbatim.

Failure semantics: prepare-hook exceptions and all-runs-crashed report
`hard_fail_score` (default −3.0) to the sampler as a *real* observation;
`sofa_realtime_timeout` kills report the trial as *pruned* (a wedge, not bad
parameters). Crashed repeats among successful ones count as 0.0 within their
test rather than failing the trial.

---

## 4. Reaching your SOFA build

Three things must agree (same build): `runsofa_exe`, the plugin tree under
`SOFA_ROOT`, and the SofaPython3 site-packages the scene imports. Set them via
`runsofa_exe` + `sofa_env`:

```python
runsofa_exe=Path(r"C:/sofa/bin/runSofa.exe"),
sofa_env={
    "SOFA_ROOT": r"C:/sofa",
    "PYTHONPATH": r"C:/sofa/lib/python3/site-packages;C:/path/to/your/modules",
},
```

`PYTHONPATH` must let the scene `import sofaopt` and your own modules. With an
editable `pip install -e` of sofaopt into the same interpreter SofaPython3 uses,
`import sofaopt` already works.

---

## 5. Shape optimization? Add a prepare hook

If a parameter changes **geometry**, generate it per trial with a hook that
returns a `TrialPrep`:

```python
from sofaopt import TrialPrep

def prepare(params, trial_dir):
    mesh = trial_dir / "shape.stl"
    build_mesh(params, out=mesh)        # your generator (CadQuery, gmsh, a CLI…)
    return TrialPrep(
        env={"OPT_MESH": str(mesh)},    # injected into the scene process
        cleanup=[mesh],                 # deleted after the trial's runs finish
    )

PROJECT = SofaOptProject(..., prepare_trial=prepare)
```

The scene then loads `trial.env["OPT_MESH"]`. Raise inside the hook to hard-fail
a candidate (e.g. invalid geometry). Parameter-only projects (stiffness, mass,
gains, …) need **no hook at all** — just read `trial.params` in the scene.

Other optional hooks: `constrain_params(params)->params` (enforce
cross-parameter relationships before use) and
`on_generation_end(gen, paths)` (cross-generation carryover).

---

## 6. Tuning the search (CMA-ES)

How thoroughly and how fast the optimizer searches is controlled entirely by a
few fields on `SofaOptProject`. You set them when you build the project:

```python
PROJECT = SofaOptProject(
    ...,
    n_parallel=6,             # population size  (also = concurrent runSofa procs)
    n_generations=120,        # how many CMA-ES update steps
    cmaes_startup_trials=24,  # random trials before CMA-ES takes over
    cmaes_sigma0=0.3,         # initial search spread
    max_active_sofa_procs=12, # hard cap on concurrent SOFA processes
)
```

### The budget — how many simulations you're committing to

```
trials evaluated      = n_parallel × n_generations
runSofa launches      = trials × Σ(run_count over selected tests)
```

So 6 × 120 = 720 candidates; if your one test has `run_count=3`, that's ~2160
`runSofa` runs. Multiply by a typical scene's wall-time to estimate the run, and
size `n_generations` to the time you actually have.

### `n_parallel` — population size (λ)

The number of candidates CMA-ES draws **per generation**, launched concurrently
as separate `runSofa` processes. Two effects:

- **Search quality:** CMA-ES estimates its next step from a whole population, so
  bigger populations give a steadier, more robust update (better on noisy or
  rugged landscapes) — at the cost of more simulations per generation.
- **Parallelism:** it's also how many scenes run at once. Match it to the CPU
  cores you can spare. Must be **≥ 4** (the framework enforces this — CMA-ES is
  ill-defined below that).

### `n_generations` — how long it refines

The number of CMA-ES update steps. More generations = more refinement of the
distribution toward good regions. This is your main "search longer" dial.

### Where it starts — `x0` (your `ParamSpec` defaults)

CMA-ES does **not** start from a random point: its initial mean is each
parameter's `default`. So **set your defaults to your best-known / baseline
design** and the search begins there and improves outward. Frozen params
(`min == max`) are held fixed and excluded from the search.

### `cmaes_startup_trials` — the random startup phase

The **first `cmaes_startup_trials` completed trials are sampled uniformly at
random** within each parameter's bounds; only afterwards does the CMA-ES
algorithm take over. CMA-ES needs a handful of evaluated points before its
covariance estimate means anything — this warm-up provides them.

**Default is now `None` = auto-sized from dimensionality**
(`project.resolve_startup_trials()`): the power of two nearest to `5·d`
(CMA-ES) or `10·d` (GP-BO), with `d` = searched (non-frozen) params — powers
of two because the Sobol' design is exactly balanced there. Adding or freezing
parameters rescales the exploration phase automatically; set an explicit int
to override.

- It's effectively *"how many random trials first."* Because a generation is
  `n_parallel` trials, setting it to `K × n_parallel` gives roughly **K fully
  random generations** before CMA-ES engages.
- **Rule of thumb:** at least one population (`≥ n_parallel`); a small multiple
  (2–4×) for rugged or higher-dimensional problems. The framework default of
  **50** suits a real project with many parameters; a 2-parameter toy is fine
  with ~8.
- Bigger = more upfront exploration (less likely to commit early to a poor
  basin); smaller = converges sooner.

### `cmaes_sigma0` — initial spread

The initial standard deviation of the search distribution, in Optuna's
internally-normalized parameter space (each range mapped to ~`[0, 1]`). So:

- `1.0` (default) is **broad** — the first CMA-ES samples spread across most of
  each parameter's range.
- `0.2–0.3` starts **local**, clustered near your defaults (`x0`) — good when
  you trust the baseline and want refinement rather than a global hunt.

As the search proceeds CMA-ES adapts this spread automatically; `sigma0` only
sets the starting width.

### `max_active_sofa_procs` — concurrency safety cap

Distinct from `n_parallel`: a single trial can launch several runs (multiple
tests / repeats), so the number of *in-flight* SOFA processes can exceed the
population. This caps the total concurrently, throttling new launches until
others finish. Set it to roughly your core count (default 12).

### Recipes

- **Quick smoke test:** `n_parallel=4, n_generations=8, cmaes_startup_trials=8`.
- **Real run:** `n_parallel=`cores-you-can-spare, `n_generations=100+`,
  `cmaes_startup_trials=50`, `cmaes_sigma0=1.0`.
- **Trust your baseline, want refinement:** keep defaults sharp, lower
  `cmaes_sigma0` to ~`0.2` and `cmaes_startup_trials` to ~`n_parallel`.
- **Rugged / many parameters:** raise `cmaes_startup_trials` and keep
  `cmaes_sigma0` near `1.0` for wider exploration.

---

## 7. Run it

Headless (`run.py`):

```python
from sofaopt import run_optimization
from project import PROJECT
run_optimization(PROJECT)
```

Dashboard (`dashboard.py`): select tests + weights, Run/Stop, live progress,
leaderboard, parameter-bounds heatmap:

```python
from sofaopt import launch_dashboard
from project import PROJECT
launch_dashboard(PROJECT, port=8050)
```

Artifacts land under `work_dir/runtime/` (`trials/gen_XXXX/trial_YY/…`,
`study.db`, `trials/progress.json`).

**Archiving.** Starting a **fresh** run (no existing `study.db` to resume) no
longer wipes `runtime/` — it *moves* it to `work_dir/archives/<timestamp>_auto/`
first, so a new run can never destroy a previous one. Archive explicitly with
`sofaopt.archive_run(PROJECT, name=..., notes=...)` or the dashboard's
**Archives** tab, which also restores/deletes archives and compares runs
(overlaid best-so-far curves + best-params diff). Restoring moves the archive
back to `runtime/` — its `study.db` is intact, so the restored run can be
resumed. Delete `work_dir/archives/` entries you don't need; they are plain
directories.

---

## 8. Checklist

- [ ] `pip install -e sofaopt[dashboard]` into the interpreter SofaPython3 uses.
- [ ] `runsofa_exe` + `sofa_env` point at one consistent SOFA build with SofaPython3.
- [ ] Each scene calls `open_trial(root)` and writes exactly one `write_score`/`prune`.
- [ ] `params` and `tests` declared; `max_score`/`weight` set per test.
- [ ] Shape project only: `prepare_trial` writes the asset + returns its env.
- [ ] `run_script` set so the dashboard can Run.
- [ ] Sanity-check a single scene first: `runSofa -l SofaPython3 -g imgui scenes/reach.py`.

---

## 9. Troubleshooting

- **Scene exits immediately / plugin ABI errors** — `runsofa_exe`, `SOFA_ROOT`
  and the SofaPython3 site-packages are from different builds. Make them one build.
- **`ModuleNotFoundError: sofaopt` inside the scene** — add sofaopt (and your
  modules) to `sofa_env["PYTHONPATH"]`, or install sofaopt into SOFA's Python.
- **All trials hard-fail** — open one scene by hand under `runSofa`; the per-run
  log is `runtime/trials/gen_*/trial_*/sofa_run*.log`.
- **Dashboard Run does nothing** — set `run_script` on the project.
- **A run never ends** — ensure every code path eventually calls `write_score`
  or `prune`; `sofa_realtime_timeout` prunes a stuck run as a backstop.
